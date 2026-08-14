"""
EDELINE Retrieval Module

Episodic retrieval system adapted from STORM's retrieval.py for EDELINE's 
diffusion-based world model architecture. Uses SimHash (LSH) with PCA on 
RecurrentEmbeddingEncoder features for similarity search, and TD-error based
anchor triggering to identify surprising transitions worth revisiting.

Key differences from STORM's retrieval.py:
- Uses (episode_id, timestep) as index keys instead of (pointer, env_idx)
- Embedding source: RecurrentEmbeddingEncoder (CNN, pre-Mamba)
- Dataset access: Episode-based with in-memory cache
- Returns SegmentId objects for BatchSampler priority queue injection
"""

import torch
import random
import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Dict, Any

from data import SegmentId


class FastHashBucket:
    """O(1) dictionary-based hash bucket supporting random sampling.
    Copied from STORM's retrieval.py without modification."""
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = []
        self.data_map = {}

    def add_or_update(self, index):
        if index in self.data_map:
            return None

        if len(self.items) < self.max_size:
            list_idx = len(self.items)
            self.items.append(index)
            self.data_map[index] = list_idx
            return None
        else:
            replace_list_idx = random.randrange(self.max_size)
            old_index = self.items[replace_list_idx]
            del self.data_map[old_index]

            self.items[replace_list_idx] = index
            self.data_map[index] = replace_list_idx
            return old_index

    def remove(self, index):
        if index not in self.data_map:
            return
        list_idx = self.data_map[index]
        last_index = self.items[-1]

        self.items[list_idx] = last_index
        self.data_map[last_index] = list_idx

        self.items.pop()
        del self.data_map[index]

    def sample(self, k, exclude=None):
        if not self.items:
            return []

        pool = self.items
        if exclude is not None and exclude in self.data_map:
            pool = [x for x in self.items if x != exclude]

        k = min(k, len(pool))
        if k == 0:
            return []

        return random.sample(pool, k)

    def __len__(self):
        return len(self.items)


class EDELINERetrievalManager:
    """
    Retrieval context manager adapted for EDELINE's architecture.
    
    Uses RecurrentEmbeddingEncoder output (CNN features, pre-Mamba) for LSH hashing,
    and stores (episode_id, timestep) pairs as index keys. Integrates with EDELINE's
    episode-based Dataset through in-memory cache access.
    """
    def __init__(self, config: Dict[str, Any], latent_dim: int, device: str = "cuda"):
        self.config = config
        self.device = device
        self.enabled = bool(config.get("enable", False))
        self.latent_dim = latent_dim

        # Trigger config
        self.trigger_mode = config.get("trigger_mode", "z_score")
        self.z_score_threshold = float(config.get("z_score_threshold", 3.5))
        self.anchor_offset = int(config.get("anchor_offset", -2))
        self.ema_alpha = float(config.get("ema_alpha", 0.01))

        # EMA statistics for z-score normalization (single environment)
        self.ema_mean = 0.0
        self.ema_var = 1.0
        self.ema_vd_mean = 0.0
        self.ema_vd_var = 1.0

        # Hashing config
        self.hash_bits = int(config.get("hash_bits", 10))
        self.use_pca = config.get("use_pca", True)
        self.max_pca_samples = int(config.get("max_pca_samples", 100000))
        self.max_bucket_size = int(config.get("max_bucket_size", 1000000000))

        # Hash projection matrix (random init, updated by PCA)
        proj = torch.randn(latent_dim, self.hash_bits, dtype=torch.float32, device=device)
        self.hash_proj = proj
        self.hash_mean = None

        bit_values = 2 ** torch.arange(self.hash_bits, dtype=torch.int64, device=device)
        self.hash_bit_values = bit_values

        # Hash memory: key -> FastHashBucket of (episode_id, timestep) tuples
        self.hash_memory = {}

        # Reverse mapping: (episode_id, timestep) -> hash_key
        self.index_to_bucket = {}

        # Active anchors queue: list of ((episode_id, timestep), hash_key)
        self.active_anchors = deque()

        # Retrieval config
        self.max_anchors = int(config.get("max_anchors", 16))
        self.multiplier = int(config.get("multiplier", 16))
        self.target = int(config.get("target", 16))
        self.max_contexts = int(config.get("max_contexts", 256))
        self.warmup_epochs = int(config.get("warmup_epochs", 5))
        self.global_rebuild_every = int(config.get("global_rebuild_every", 10))

        # Track which episodes have been indexed
        self._indexed_episode_count = 0

    def _hash_keys(self, latent: torch.Tensor) -> List[int]:
        """Compute LSH hash keys from latent vectors. Same logic as STORM."""
        if latent.numel() == 0:
            return []

        latent_f = latent.float()
        if self.hash_mean is not None:
            latent_f = latent_f - self.hash_mean.float()

        scores = latent_f @ self.hash_proj.float()
        bits = scores > 0
        keys = (bits.to(torch.int64) * self.hash_bit_values).sum(dim=-1)
        return keys.detach().cpu().tolist()

    def _insert_into_bucket(self, episode_id: int, timestep: int, key: int):
        """Insert (episode_id, timestep) into the hash bucket for the given key."""
        idx_tuple = (episode_id, timestep)
        old_key = self.index_to_bucket.get(idx_tuple, -1)

        if old_key != -1 and old_key != key:
            old_queue = self.hash_memory.get(old_key)
            if old_queue is not None:
                old_queue.remove(idx_tuple)

        queue = self.hash_memory.get(key)
        if queue is None:
            queue = FastHashBucket(max_size=self.max_bucket_size)
            self.hash_memory[key] = queue

        evicted_idx = queue.add_or_update(idx_tuple)
        if evicted_idx is not None:
            if evicted_idx in self.index_to_bucket:
                del self.index_to_bucket[evicted_idx]

        self.index_to_bucket[idx_tuple] = key

    @torch.no_grad()
    def build_hash_index(self, dataset, world_model, device, chunk_size: int = 512):
        """
        Scan all episodes in the dataset, encode each frame with 
        RecurrentEmbeddingEncoder, and build the hash index.
        """
        if not self.enabled:
            return

        self.hash_memory.clear()
        self.index_to_bucket.clear()

        all_latents = []
        all_indices = []

        for episode_id in range(dataset.num_episodes):
            episode = dataset.load_episode(episode_id)
            num_frames = len(episode)

            # Process frames in chunks to limit GPU memory usage
            for start in range(0, num_frames, chunk_size):
                end = min(start + chunk_size, num_frames)
                # episode.obs: (T, C, H, W) in [-1, 1]
                obs_chunk = episode.obs[start:end].to(device)
                latent = world_model.encode_obs_for_hash(obs_chunk)  # (N, latent_dim)
                all_latents.append(latent)
                for t in range(start, end):
                    all_indices.append((episode_id, t))

        if len(all_latents) == 0:
            self._indexed_episode_count = dataset.num_episodes
            return

        full_latents = torch.cat(all_latents, dim=0)

        # PCA projection update
        if self.use_pca and full_latents.shape[0] > self.hash_bits:
            self._update_pca(full_latents)

        # Bulk hashing and insertion
        keys = self._hash_keys(full_latents)
        for (ep_id, ts), key in zip(all_indices, keys):
            self._insert_into_bucket(ep_id, ts, key)

        self._indexed_episode_count = dataset.num_episodes
        print(f"[Retrieval] Hash index built: {len(all_indices)} frames from {dataset.num_episodes} episodes (PCA: {self.use_pca})")

    @torch.no_grad()
    def update_new_episodes(self, dataset, world_model, device, chunk_size: int = 512):
        """Incrementally index episodes added since the last index build."""
        if not self.enabled:
            return

        new_count = dataset.num_episodes - self._indexed_episode_count
        if new_count <= 0:
            return

        for episode_id in range(self._indexed_episode_count, dataset.num_episodes):
            episode = dataset.load_episode(episode_id)
            num_frames = len(episode)

            for start in range(0, num_frames, chunk_size):
                end = min(start + chunk_size, num_frames)
                obs_chunk = episode.obs[start:end].to(device)
                latent = world_model.encode_obs_for_hash(obs_chunk)

                keys = self._hash_keys(latent)
                for i, t in enumerate(range(start, end)):
                    self._insert_into_bucket(episode_id, t, keys[i])

        self._indexed_episode_count = dataset.num_episodes

    def _update_pca(self, full_latents: torch.Tensor):
        """Update hash projection using PCA on the given latents."""
        pca_input = full_latents
        if pca_input.shape[0] > self.max_pca_samples:
            indices = torch.randperm(pca_input.shape[0], device=pca_input.device)[:self.max_pca_samples]
            pca_input = pca_input[indices]

        mean_latent = pca_input.mean(dim=0, keepdim=True)
        centered_input = pca_input - mean_latent

        U, S, V = torch.pca_lowrank(centered_input, q=self.hash_bits, center=False)

        if not torch.isnan(V).any():
            self.hash_proj = V.to(self.device)
            self.hash_mean = mean_latent.to(self.device)

    @torch.no_grad()
    def rebuild_hash_index(self, dataset, world_model, device, chunk_size: int = 512):
        """
        Global Rebuild: clear all hash buckets, recompute PCA, and rehash 
        all transitions. Fixes representation drift.
        """
        if not self.enabled:
            return

        print("[Retrieval] Starting Global Rebuild...")
        self.build_hash_index(dataset, world_model, device, chunk_size)

    def add_batch_transitions(
        self,
        values: torch.Tensor,      # (B, T)
        rewards: torch.Tensor,     # (B, T)
        ends: torch.Tensor,        # (B, T)
        gamma: float,
        segment_ids: List[SegmentId],
        mask: torch.Tensor,        # (B, T) bool mask
        is_warmup: bool = False,
    ) -> int:
        """
        Compute TD-error from value estimates and trigger anchors for 
        surprising transitions.
        
        Args:
            values: Value estimates from ActorCritic, shape (B, T)
            rewards: Rewards from batch, shape (B, T)
            ends: End flags from batch, shape (B, T)
            gamma: Discount factor
            segment_ids: List of SegmentId for each batch element
            mask: Padding mask, shape (B, T)
            is_warmup: If True, only update EMA stats, don't trigger anchors
        
        Returns:
            Number of triggered anchors
        """
        if not self.enabled:
            return 0

        B, T = values.shape
        if T < 3:
            return 0

        # Compute TD-error: |r + γV(s_{t+1})(1-d) - V(s_t)|
        v_curr = values[:, :-1]   # (B, T-1)
        v_next = values[:, 1:]    # (B, T-1)
        r_curr = rewards[:, :-1]  # (B, T-1)
        d_curr = ends[:, :-1].float()  # (B, T-1)
        m_curr = mask[:, :-1] & mask[:, 1:]  # (B, T-1) valid mask

        # Value difference (for combined metric)
        v_prev = values[:, :-2]          # (B, T-2)
        v_curr_for_vd = values[:, 1:-1]  # (B, T-2)
        value_diff = v_curr_for_vd - v_prev  # (B, T-2)

        target_v = r_curr + gamma * v_next * (1.0 - d_curr)
        delta_v_raw = target_v - v_curr  # (B, T-1)
        abs_delta_v = torch.abs(delta_v_raw)

        # Update EMA statistics
        valid_mask_1d = m_curr.any(dim=1)  # (B,) which batch elements have valid data
        if valid_mask_1d.any():
            valid_abs_delta = abs_delta_v[m_curr]
            if valid_abs_delta.numel() > 0:
                batch_mean = valid_abs_delta.mean().item()
                batch_var = valid_abs_delta.var(unbiased=False).item() if valid_abs_delta.numel() > 1 else 0.0

                diff = batch_mean - self.ema_mean
                self.ema_mean += self.ema_alpha * diff
                self.ema_var = (1 - self.ema_alpha) * (self.ema_var + self.ema_alpha * (batch_var + diff ** 2))

            # Value diff EMA (uses shorter sequence: T-2)
            m_vd = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
            valid_vd = value_diff[m_vd]
            if valid_vd.numel() > 0:
                vd_mean = valid_vd.mean().item()
                vd_var = valid_vd.var(unbiased=False).item() if valid_vd.numel() > 1 else 0.0

                diff_vd = vd_mean - self.ema_vd_mean
                self.ema_vd_mean += self.ema_alpha * diff_vd
                self.ema_vd_var = (1 - self.ema_alpha) * (self.ema_vd_var + self.ema_alpha * (vd_var + diff_vd ** 2))

        if is_warmup:
            return 0

        # Compute combined z-score metric for anchor triggering
        # We use the T-2 length (intersection of td-error and value_diff)
        td_for_metric = abs_delta_v[:, :-1]  # (B, T-2) align with value_diff
        m_metric = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]

        z_td = (td_for_metric - self.ema_mean) / (np.sqrt(self.ema_var) + 1e-8)
        z_vd = (value_diff - self.ema_vd_mean) / (np.sqrt(self.ema_vd_var) + 1e-8)

        combined_metric = torch.relu(z_td) * torch.nn.functional.softplus(z_vd)

        # Mask invalid positions
        combined_metric[~m_metric] = float('-inf')

        # Find max metric per batch element
        max_val, max_idx = combined_metric.max(dim=1)  # (B,)
        triggered_b = max_val >= self.z_score_threshold

        num_trig = 0
        if triggered_b.any():
            b_indices = triggered_b.nonzero(as_tuple=True)[0]
            t_indices = max_idx[b_indices]

            for i in range(len(b_indices)):
                b_idx = b_indices[i].item()
                t_idx = t_indices[i].item()

                seg_id = segment_ids[b_idx]
                # Map batch-internal timestep to actual episode timestep
                actual_timestep = max(0, seg_id.start) + t_idx + self.anchor_offset
                if actual_timestep < 0:
                    continue

                ep_id = seg_id.episode_id
                anchor_key = self.index_to_bucket.get((ep_id, actual_timestep), -1)
                if anchor_key != -1:
                    anchor = (ep_id, actual_timestep)
                    self.active_anchors.append((anchor, anchor_key))
                    num_trig += 1

        return num_trig

    @torch.no_grad()
    def retrieve_segments(
        self,
        dataset,
        context_length: int,
        world_model,
        device,
    ) -> List[SegmentId]:
        """
        Pop anchors from the queue, find similar segments via hash lookup,
        and return SegmentId objects for BatchSampler injection.
        
        Args:
            dataset: EDELINE Dataset instance
            context_length: Segment length for WorldModelEnv (num_steps_conditioning)
            world_model: WorldModel instance for lazy recomputation
            device: Torch device
        
        Returns:
            List of SegmentId objects to be prioritized in BatchSampler
        """
        if not self.enabled or len(self.active_anchors) == 0:
            return []

        popped_anchors = []
        for _ in range(min(self.max_anchors, len(self.active_anchors))):
            popped_anchors.append(self.active_anchors.popleft())

        result_segment_ids = []
        total_retrieved = 0

        for (anchor_ep, anchor_ts), anchor_key in popped_anchors:
            if total_retrieved >= self.max_contexts:
                break

            queue = self.hash_memory.get(anchor_key)
            if not queue:
                continue

            # Sample candidates from the same hash bucket
            sampled_indices = queue.sample(
                k=self.multiplier * (self.target - 1),
                exclude=(anchor_ep, anchor_ts)
            )

            # Lazy recomputation: re-encode candidates to verify hash match
            valid_candidates = []
            if sampled_indices:
                obs_list = []
                valid_sampled = []
                for (ep_id, ts) in sampled_indices:
                    if ep_id >= dataset.num_episodes:
                        continue
                    episode = dataset.load_episode(ep_id)
                    if ts >= len(episode):
                        continue
                    obs_list.append(episode.obs[ts:ts + 1].to(device))
                    valid_sampled.append((ep_id, ts))

                if obs_list:
                    obs_tensor = torch.cat(obs_list, dim=0)  # (N, C, H, W)
                    encoded = world_model.encode_obs_for_hash(obs_tensor)
                    current_keys = self._hash_keys(encoded)

                    for i, k in enumerate(current_keys):
                        if k == anchor_key:
                            valid_candidates.append(valid_sampled[i])
                            if len(valid_candidates) >= self.target - 1:
                                break

            # Build segment IDs: anchor first, then matched candidates
            all_matches = [(anchor_ep, anchor_ts)] + valid_candidates
            for (ep_id, ts) in all_matches:
                if total_retrieved >= self.max_contexts:
                    break

                # Construct SegmentId for this match
                # The segment should end at or near the matched timestep
                episode = dataset.load_episode(ep_id)
                ep_len = len(episode)
                stop = min(ts + 1, ep_len)
                start = stop - context_length

                # Validate: segment must be within episode bounds
                # (make_segment handles padding, so negative start is OK)
                if stop <= 0:
                    continue

                seg_id = SegmentId(episode_id=ep_id, start=start, stop=stop)
                result_segment_ids.append(seg_id)
                total_retrieved += 1

        return result_segment_ids

    def state_dict(self) -> dict:
        """Save retrieval manager state for checkpointing."""
        return {
            "ema_mean": self.ema_mean,
            "ema_var": self.ema_var,
            "ema_vd_mean": self.ema_vd_mean,
            "ema_vd_var": self.ema_vd_var,
            "hash_proj": self.hash_proj.cpu(),
            "hash_mean": self.hash_mean.cpu() if self.hash_mean is not None else None,
            "indexed_episode_count": self._indexed_episode_count,
        }

    def load_state_dict(self, state_dict: dict):
        """Restore retrieval manager state from checkpoint."""
        self.ema_mean = state_dict["ema_mean"]
        self.ema_var = state_dict["ema_var"]
        self.ema_vd_mean = state_dict["ema_vd_mean"]
        self.ema_vd_var = state_dict["ema_vd_var"]
        self.hash_proj = state_dict["hash_proj"].to(self.device)
        hm = state_dict.get("hash_mean")
        self.hash_mean = hm.to(self.device) if hm is not None else None
        self._indexed_episode_count = state_dict.get("indexed_episode_count", 0)
