import hydra
from omegaconf import DictConfig, OmegaConf

from trainer import Trainer
from utils import skip_if_run_is_over


OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../config", config_name="trainer", version_base="1.3")
def main(cfg: DictConfig):
    import torch.multiprocessing as mp
    import resource
    try:
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))
    except (ValueError, OSError):
        pass

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    try:
        mp.set_sharing_strategy('file_system')
    except RuntimeError:
        pass
    run(cfg)


@skip_if_run_is_over
def run(cfg):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
