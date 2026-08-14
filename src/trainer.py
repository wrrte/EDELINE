from datetime import datetime
from functools import partial
from pathlib import Path
import shutil
import time
from typing import Tuple

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from agent import Agent
from edeline_retrieval import EDELINERetrievalManager
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, collate_segments_to_batch, Dataset, DatasetTraverser
from envs import make_env, WorldModelEnv
from utils import (
    CommonTools,
    configure_opt,
    count_parameters,
    get_lr_sched,
    keep_agent_copies_every,
    Logs,
    process_confusion_matrices_if_any_and_compute_classification_metrics,
    save_info_for_import_script,
    save_with_backup,
    set_seed,
    StateDictMixin,
    try_until_no_except,
    wandb_log,
)
from rl_plotter.logger import Logger


class Trainer(StateDictMixin):
    def __init__(self, cfg: DictConfig) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        OmegaConf.resolve(cfg)
        self._cfg = cfg

        # Seed
        if cfg.common.seed is None:
            cfg.common.seed = int(datetime.now().timestamp()) % 10**5
        set_seed(cfg.common.seed)

        # Init wandb
        wandb_key_path = Path(".wandb_api_key")
        if wandb_key_path.exists():
            with open(wandb_key_path, "r") as f:
                wandb.login(key=f.read().strip())
                
        try_until_no_except(
            partial(wandb.init, config=OmegaConf.to_container(cfg, resolve=True), reinit=True, resume=True, **cfg.wandb)
        )

        # Flags
        self._is_static_dataset = cfg.collection.path_to_static_dataset is not None
        self._is_model_free = cfg.training.model_free
        self._use_cuda = "cuda" in cfg.common.device

        # Device
        self._device = torch.device(cfg.common.device)
        if self._use_cuda:
            torch.cuda.set_device(self._device)  # fix compilation error on multi-gpu nodes

        # Checkpointing
        self._path_ckpt_dir = Path("checkpoints")
        self._path_state_ckpt = self._path_ckpt_dir / "state.pt"
        self._keep_agent_copies = partial(
            keep_agent_copies_every,
            every=cfg.checkpointing.save_agent_every,
            path_ckpt_dir=self._path_ckpt_dir,
            num_to_keep=cfg.checkpointing.num_to_keep,
        )
        self._save_info_for_import_script = partial(
            save_info_for_import_script, run_name=cfg.wandb.name, path_ckpt_dir=self._path_ckpt_dir
        )

        # First time, init files hierarchy
        if not cfg.common.resume:
            self._path_ckpt_dir.mkdir(exist_ok=False, parents=False)
            path_config = Path("config") / "trainer.yaml"
            path_config.parent.mkdir(exist_ok=False, parents=False)
            shutil.move(".hydra/config.yaml", path_config)
            wandb.save(str(path_config))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")

        # Datasets
        num_workers = cfg.training.num_workers_data_loaders
        use_manager = cfg.training.cache_in_ram and (num_workers > 0)
        p = Path(cfg.collection.path_to_static_dataset) if self._is_static_dataset else Path("dataset")
        self.train_dataset = Dataset(p / "train", "train_dataset", cfg.training.cache_in_ram, use_manager)
        self.test_dataset = Dataset(p / "test", "test_dataset", cache_in_ram=True)
        self.train_dataset.load_from_default_path()
        self.test_dataset.load_from_default_path()

        # Logger
        self._logger = Logger(exp_name="LucidDream", env_name=self._cfg.env.train.id)

        # Envs
        train_env = make_env(num_envs=cfg.collection.train.num_envs, device=self._device, cfg=cfg.env.train)
        test_env = make_env(num_envs=cfg.collection.test.num_envs, device=self._device, cfg=cfg.env.test)

        # Create models
        num_actions = int(test_env.num_actions)
        self.agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(self._device)

        if cfg.initialization.path_to_ckpt is not None:
            self.agent.load(**cfg.initialization)

        # Collectors
        self._train_collector = make_collector(
            train_env, self.agent.actor_critic, self.train_dataset, cfg.collection.train.epsilon
        )
        self._test_collector = make_collector(
            test_env, self.agent.actor_critic, self.test_dataset, cfg.collection.test.epsilon, reset_every_collect=True
        )

        ######################################################

        # Optimizers and LR schedulers

        def build_opt(name: str) -> torch.optim.AdamW:
            return configure_opt(getattr(self.agent, name), **getattr(cfg, name).optimizer)

        def build_lr_sched(name: str) -> torch.optim.lr_scheduler.LambdaLR:
            return get_lr_sched(self.opt.get(name), getattr(cfg, name).training.lr_warmup_steps)

        self._model_names = ["world_model", "actor_critic"]
        self.opt = CommonTools(*map(build_opt, self._model_names))
        self.lr_sched = CommonTools(*map(build_lr_sched, self._model_names))

        # Data loaders

        make_data_loader = partial(
            DataLoader,
            dataset=self.train_dataset,
            collate_fn=collate_segments_to_batch,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=self._use_cuda,
            pin_memory_device=str(self._device) if self._use_cuda else "",
        )

        c = cfg.world_model.training
        bs = BatchSampler(self.train_dataset, c.batch_size, c.seq_length, c.sample_weights, can_sample_beyond_end=True)
        dl_world_model_train = make_data_loader(batch_sampler=bs)
        dl_world_model_test = DatasetTraverser(self.test_dataset, c.batch_size, c.seq_length)

        self._data_loader_train = CommonTools(dl_world_model_train, None)
        self._data_loader_test = CommonTools(dl_world_model_test, None)

        # RL env

        if self._is_model_free:
            rl_env = make_env(num_envs=cfg.actor_critic.training.batch_size, device=self._device, cfg=cfg.env.train)

        else:
            c = cfg.actor_critic.training
            sl = cfg.agent.world_model.denoiser.inner_model.num_steps_conditioning
            bs = BatchSampler(self.train_dataset, c.batch_size, sl, c.sample_weights)
            self._ac_batch_sampler = bs  # store reference for retrieval injection
            dl_actor_critic = make_data_loader(batch_sampler=bs)
            wm_env_cfg = instantiate(cfg.world_model_env)
            rl_env = WorldModelEnv(self.agent.world_model, dl_actor_critic, wm_env_cfg)

            if cfg.training.compile_wm:
                rl_env.predict_next_obs = torch.compile(rl_env.predict_next_obs, mode="reduce-overhead")
                rl_env.predict_rew_end = torch.compile(rl_env.predict_rew_end, mode="reduce-overhead")

        # Setup training
        sigma_distribution_cfg = instantiate(cfg.denoiser.sigma_distribution)
        actor_critic_loss_cfg = instantiate(cfg.actor_critic.actor_critic_loss)
        self.agent.setup_training(sigma_distribution_cfg, actor_critic_loss_cfg, rl_env)

        # Retrieval setup
        retrieval_cfg = OmegaConf.to_container(cfg.retrieval, resolve=True) if hasattr(cfg, 'retrieval') else {}
        if retrieval_cfg.get('enable', False):
            enc_cfg = cfg.agent.world_model.recurrent_embedding_module
            img_size = enc_cfg.img_size
            total_down = sum(enc_cfg.down)
            spatial = img_size // (2 ** total_down)
            latent_dim = enc_cfg.channels[-1] * spatial * spatial
            self.retrieval_manager = EDELINERetrievalManager(
                config=retrieval_cfg, latent_dim=latent_dim, device=str(self._device)
            )
            print(f"[Retrieval] Enabled. latent_dim={latent_dim}, hash_bits={retrieval_cfg.get('hash_bits', 10)}")
        else:
            self.retrieval_manager = None

        # Training state (things to be saved/restored)
        self.epoch = 0
        self.num_epochs_collect = None
        self.num_episodes_test = 0
        self.num_batch_train = CommonTools(0, 0)
        self.num_batch_test = CommonTools(0, 0)

        if cfg.common.resume:
            self.load_state_checkpoint()
        else:
            self.save_checkpoint()

        for name in self._model_names:
            print(f"{count_parameters(getattr(self.agent, name))} parameters in {name}")
        print(self.train_dataset)
        print(self.test_dataset)

    def run(self) -> None:
        to_log = []

        if self.epoch == 0:
            if self._is_model_free or self._is_static_dataset:
                self.num_epochs_collect = 0
            else:
                self.num_epochs_collect, to_log_ = self.collect_initial_dataset()
                to_log += to_log_

            # Build initial hash index after first collection
            if self.retrieval_manager is not None and self.train_dataset.num_episodes > 0:
                self.retrieval_manager.build_hash_index(
                    self.train_dataset, self.agent.world_model, self._device
                )

        num_epochs = self.num_epochs_collect + self._cfg.training.num_final_epochs

        while self.epoch < num_epochs:
            self.epoch += 1

            print(f"\nEpoch {self.epoch} / {num_epochs}\n")
            start_time = time.time()

            # Training
            should_collect_train = (
                not self._is_model_free and not self._is_static_dataset and self.epoch <= self.num_epochs_collect
            )

            if should_collect_train:
                c = self._cfg.collection.train
                to_log += self._train_collector.send(NumToCollect(steps=c.steps_per_epoch))

                # Incrementally index new episodes
                if self.retrieval_manager is not None:
                    self.retrieval_manager.update_new_episodes(
                        self.train_dataset, self.agent.world_model, self._device
                    )

            if self._cfg.training.should:
                to_log += self.train_agent()

            # Periodic global rebuild (PCA update + rehash)
            if (self.retrieval_manager is not None
                    and self.retrieval_manager.enabled
                    and self.epoch % self.retrieval_manager.global_rebuild_every == 0
                    and self.train_dataset.num_episodes > 0):
                self.retrieval_manager.rebuild_hash_index(
                    self.train_dataset, self.agent.world_model, self._device
                )
                to_log.append({"retrieval/global_rebuild": 1.0})

            # Evaluation
            should_test = self._cfg.evaluation.should and (self.epoch % self._cfg.evaluation.every == 0)
            should_collect_test = should_test and not self._is_static_dataset

            if should_collect_test:
                to_log += self.collect_test()

            if should_test and not self._is_model_free:
                to_log += self.test_agent()

            # Logging
            to_log.append({"duration": (time.time() - start_time) / 3600})
            wandb_log(to_log, self.epoch)
            to_log = []

            # Checkpointing
            self.save_checkpoint()

        # Last collect
        if not self._is_static_dataset:
            wandb_log(self.collect_test(final=True), self.epoch)

    def collect_initial_dataset(self) -> Tuple[int, Logs]:
        print("\nInitial collect\n")
        to_log = []
        c = self._cfg.collection.train
        min_steps = c.first_epoch.min
        steps_per_epoch = c.steps_per_epoch
        max_steps = c.first_epoch.max
        threshold_rew = c.first_epoch.threshold_rew
        assert min_steps % steps_per_epoch == 0

        steps = min_steps
        while True:
            to_log += self._train_collector.send(NumToCollect(steps=steps))
            num_steps = self.train_dataset.num_steps
            total_minority_rew = sum(sorted(self.train_dataset.counts_rew)[:-1])
            if total_minority_rew >= threshold_rew:
                break
            if (max_steps is not None) and num_steps >= max_steps:
                print("Reached the specified maximum for initial collect")
                break
            print(f"Minority reward: {total_minority_rew}/{threshold_rew} -> Keep collecting\n")
            steps = steps_per_epoch

        print("\nSummary of initial collect:")
        print(f"Num steps: {num_steps} / {c.num_steps_total}")
        print(f"Reward counts: {dict(self.train_dataset.counter_rew)}")

        remaining_steps = c.num_steps_total - num_steps
        assert remaining_steps % c.steps_per_epoch == 0
        num_epochs_collect = remaining_steps // c.steps_per_epoch

        return num_epochs_collect, to_log

    def collect_test(self, final: bool = False) -> Logs:
        c = self._cfg.collection.test
        episodes = c.num_final_episodes if final else c.num_episodes
        td = self.test_dataset
        td.clear()
        to_log = self._test_collector.send(NumToCollect(episodes=episodes))
        key_ep_id = f"{td.name}/episode_id"
        to_log = [{k: v + self.num_episodes_test if k == key_ep_id else v for k, v in x.items()} for x in to_log]

        print(f"\nSummary of {'final' if final else 'test'} collect: {td.num_episodes} episodes ({td.num_steps} steps)")
        keys = [key_ep_id, "return", "length"]
        to_log_episodes = [x for x in to_log if set(x.keys()) == set(keys)]
        episode_ids, returns, lengths = [[d[k] for d in to_log_episodes] for k in keys]
        for i, (ep_id, ret, length) in enumerate(zip(episode_ids, returns, lengths)):
            print(f"  Episode {ep_id}: return = {ret} length = {length}\n", end="\n" if i == episodes - 1 else "")

        # Logger for plotting
        num_steps = self.train_dataset.num_steps
        mean_return = np.mean(returns)
        self._logger.update(score=[mean_return], total_steps=num_steps)
        to_log.append({"eval_return": mean_return})

        self.num_episodes_test += episodes

        if final:
            to_log.append({"final_return_mean": np.mean(returns), "final_return_std": np.std(returns)})
            print(to_log[-1])

        return to_log

    def train_agent(self) -> Logs:
        self.agent.train()
        self.agent.zero_grad()
        to_log = []
        model_names = ["actor_critic"] if self._is_model_free else self._model_names
        for name in model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                steps = cfg.steps_first_epoch if self.epoch == 1 else cfg.steps_per_epoch

                if name == "actor_critic" and self.retrieval_manager is not None:
                    # Inject retrieved segments before actor-critic training
                    self._inject_retrieved_segments()

                to_log += self.train_component(name, steps)
        return to_log

    def _inject_retrieved_segments(self) -> None:
        """Retrieve similar segments and push them to the AC BatchSampler's priority queue."""
        if self.retrieval_manager is None or not self.retrieval_manager.enabled:
            return
        if not hasattr(self, '_ac_batch_sampler'):
            return

        is_warmup = self.epoch <= self.retrieval_manager.warmup_epochs
        if is_warmup:
            return

        sl = self._cfg.agent.world_model.denoiser.inner_model.num_steps_conditioning
        retrieved_ids = self.retrieval_manager.retrieve_segments(
            self.train_dataset,
            context_length=sl,
            world_model=self.agent.world_model,
            device=self._device,
        )

        if retrieved_ids:
            self._ac_batch_sampler.push_priority_segments(retrieved_ids)
            print(f"[Retrieval] Injected {len(retrieved_ids)} segments into AC BatchSampler")

    @torch.no_grad()
    def test_agent(self) -> Logs:
        self.agent.eval()
        to_log = []
        model_names = [] if self._is_model_free else self._model_names[:-1]
        for name in model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                to_log += self.test_component(name)
        return to_log

    def train_component(self, name: str, steps: int) -> Logs:
        cfg = getattr(self._cfg, name).training
        model = getattr(self.agent, name)
        opt = self.opt.get(name)
        lr_sched = self.lr_sched.get(name)
        data_loader = self._data_loader_train.get(name)

        model.train()
        opt.zero_grad()
        data_iterator = iter(data_loader) if data_loader is not None else None
        to_log = []

        num_steps = cfg.grad_acc_steps * steps

        # Retrieval: determine if we should trigger anchors during WM training
        should_trigger = (
            name == "world_model"
            and self.retrieval_manager is not None
            and self.retrieval_manager.enabled
        )
        is_retrieval_warmup = self.epoch <= self.retrieval_manager.warmup_epochs if should_trigger else True
        total_triggered = 0

        for i in trange(num_steps, desc=f"Training {name}"):
            batch = next(data_iterator).to(self._device) if data_iterator is not None else None
            loss, metrics = model.compute_loss(batch) if batch is not None else model.compute_loss()
            loss.backward()

            # Retrieval: TD-error based anchor triggering during WM training
            if should_trigger and batch is not None:
                with torch.no_grad():
                    ac = self.agent.actor_critic
                    b, t, c, h, w = batch.obs.shape
                    hx = torch.zeros(b, ac.lstm_dim, device=self._device)
                    cx = torch.zeros(b, ac.lstm_dim, device=self._device)
                    values = []
                    for ti in range(t):
                        _, val, (hx, cx) = ac(batch.obs[:, ti], (hx, cx))
                        values.append(val)
                    values = torch.stack(values, dim=1)  # (B, T)

                    num_trig = self.retrieval_manager.add_batch_transitions(
                        values=values,
                        rewards=batch.rew,
                        ends=batch.end,
                        gamma=self._cfg.actor_critic.actor_critic_loss.gamma,
                        segment_ids=batch.segment_ids,
                        mask=batch.mask_padding,
                        is_warmup=is_retrieval_warmup,
                    )
                    total_triggered += num_trig

            num_batch = self.num_batch_train.get(name)
            metrics[f"num_batch_train_{name}"] = num_batch
            self.num_batch_train.set(name, num_batch + 1)

            if (i + 1) % cfg.grad_acc_steps == 0:
                if cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    metrics["grad_norm_before_clip"] = grad_norm

                opt.step()
                opt.zero_grad()

                if lr_sched is not None:
                    metrics["lr"] = lr_sched.get_last_lr()[0]
                    lr_sched.step()

            to_log.append(metrics)

        # Log retrieval stats
        if should_trigger:
            to_log.append({
                "retrieval/triggered_anchors_epoch": total_triggered,
                "retrieval/active_anchors_queue": len(self.retrieval_manager.active_anchors),
                "retrieval/td_error_ema_mean": self.retrieval_manager.ema_mean,
                "retrieval/td_error_ema_var": self.retrieval_manager.ema_var,
                "retrieval/num_hash_buckets": len(self.retrieval_manager.hash_memory),
            })

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"{name}/train/{k}": v for k, v in d.items()} for d in to_log]
        return to_log

    @torch.no_grad()
    def test_component(self, name: str) -> Logs:
        model = getattr(self.agent, name)
        data_loader = self._data_loader_test.get(name)
        model.eval()
        to_log = []
        for batch in tqdm(data_loader, desc=f"Evaluating {name}"):
            batch = batch.to(self._device)
            _, metrics = model.compute_loss(batch)
            num_batch = self.num_batch_test.get(name)
            metrics[f"num_batch_test_{name}"] = num_batch
            self.num_batch_test.set(name, num_batch + 1)
            to_log.append(metrics)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"{name}/test/{k}": v for k, v in d.items()} for d in to_log]
        return to_log

    def load_state_checkpoint(self) -> None:
        self.load_state_dict(torch.load(self._path_state_ckpt, map_location=self._device))

    def save_checkpoint(self) -> None:
        save_with_backup(self.state_dict(), self._path_state_ckpt)
        self.train_dataset.save_to_default_path()
        self.test_dataset.save_to_default_path()
        self._keep_agent_copies(self.agent.state_dict(), self.epoch)
        self._save_info_for_import_script(self.epoch)