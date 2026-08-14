from typing import Generator, List, Optional

import numpy as np
import torch

from .dataset import Dataset
from .segment import SegmentId


class BatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        seq_length: int,
        sample_weights: Optional[List] = None,
        can_sample_beyond_end: bool = False,
    ) -> None:
        try:
            super().__init__(dataset)
        except TypeError:
            super().__init__()
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.can_sample_beyond_end = can_sample_beyond_end
        self._priority_queue: List[SegmentId] = []

    def push_priority_segments(self, segment_ids: List[SegmentId]) -> None:
        """Register segments to be sampled with priority (e.g., from retrieval)."""
        self._priority_queue.extend(segment_ids)

    def __len__(self):
        raise NotImplementedError

    def __iter__(self) -> Generator[List[SegmentId], None, None]:
        while True:
            yield self.sample()

    def sample(self) -> List[SegmentId]:
        # Consume priority segments first
        priority_batch: List[SegmentId] = []
        while self._priority_queue and len(priority_batch) < self.batch_size:
            priority_batch.append(self._priority_queue.pop(0))

        remaining = self.batch_size - len(priority_batch)
        if remaining > 0:
            random_batch = self._sample_random(remaining)
        else:
            random_batch = []

        return priority_batch + random_batch

    def _sample_random(self, count: int) -> List[SegmentId]:
        """Sample `count` random segments from the dataset (original logic)."""
        num_episodes = self.dataset.num_episodes

        if (self.sample_weights is None) or num_episodes < len(self.sample_weights):
            weights = self.dataset.lengths / self.dataset.num_steps
        else:
            weights = self.sample_weights
            num_weights = len(self.sample_weights)
            assert all([0 <= x <= 1 for x in weights]) and sum(weights) == 1
            sizes = [
                num_episodes // num_weights + (num_episodes % num_weights) * (i == num_weights - 1)
                for i in range(num_weights)
            ]
            weights = [w / s for (w, s) in zip(weights, sizes) for _ in range(s)]

        episode_ids = np.random.choice(np.arange(num_episodes), size=count, replace=True, p=weights)
        timesteps = np.random.randint(low=0, high=self.dataset.lengths[episode_ids])

        # padding allowed, both before start and after end
        if self.can_sample_beyond_end:
            starts = timesteps - np.random.randint(0, self.seq_length, len(timesteps))
            stops = starts + self.seq_length

        # padding allowed only before start
        else:
            stops = np.minimum(
                self.dataset.lengths[episode_ids], timesteps + 1 + np.random.randint(0, self.seq_length, len(timesteps))
            )
            starts = stops - self.seq_length

        return [SegmentId(*x) for x in zip(episode_ids, starts, stops)]

