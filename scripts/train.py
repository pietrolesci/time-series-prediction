import logging
import time
from pathlib import Path

import hydra
import torch
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.datamodule import DataloaderConfig, DataModule
from src.loggers import TensorBoardLogger
from src.models import MODEL_CONFIG
from src.module import LOBModel, OptimCofig
from src.utilities import conf_to_dict, instantiate_from_conf

SEP_LINE = f"{'=' * 80}"

# Configure the logger and configure colorlog
logger = logging.getLogger("hydra")


@hydra.main(version_base=None, config_path="../conf", config_name="train_conf")
def main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()

    # Config
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, "./hparams.yaml")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}\n{SEP_LINE}")

    # Load datamodule
    dataloader_config = DataloaderConfig(**conf_to_dict(cfg.data))
    datamodule = DataModule(dataloader_config=dataloader_config, **conf_to_dict(cfg.datamodule))
    datamodule.setup()

    # Load module
    config: MODEL_CONFIG = instantiate_from_conf(
        cfg.model, num_targets=datamodule.num_targets, num_levels=datamodule.num_levels
    )  # type: ignore
    logger.info(f"Model config:\n{config.to_dict()}")  # type: ignore

    # Rename run folder in light of the model config
    cwd = Path.cwd()
    new_name = cwd.parent / f"{config.name}_{cwd.name}"
    logger.info(f"Renaming run folder from: {cwd.name} to {new_name.name}")
    cwd.rename(new_name)

    # Maybe compile
    module = LOBModel(config, OptimCofig(**conf_to_dict(cfg.optim)))
    if cfg.torch_compile:
        module.forward = torch.compile(module.forward)

    # Load trainer
    loggers, callbacks = instantiate_from_conf([cfg.get(i) for i in ("loggers", "callbacks")])
    trainer = Trainer(**conf_to_dict(cfg.trainer), logger=loggers, callbacks=callbacks)

    # Train
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")
    trainer.fit(model=module, datamodule=datamodule, ckpt_path=cfg.resume_from_checkpoint)
    logger.info(f"Training total time: {(time.perf_counter() - start_time) / 60:.1f} minutes")

    # Test
    trainer.test(datamodule=datamodule, ckpt_path="best")

    # TODO: rename folder with {model_name}-{num_params}-{tok_name} automatically
    # f"{cfg.model}-{model.num_parameters() / 1e6:.0f}M-{tok_path.name}"
    # Rename current working directory to "cur_dir"
    # cur_dir = Path.cwd()
    # new_dir = cur_dir.with_name("cur_dir")
    # cur_dir.rename(new_dir)
    for log in trainer.loggers:
        if isinstance(log, TensorBoardLogger):
            log.save_to_parquet("tb_logs.parquet")


if __name__ == "__main__":
    main()
