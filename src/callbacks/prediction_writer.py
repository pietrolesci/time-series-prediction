from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torch import Tensor

from src.module import RunningStage


class PredictionWriter(Callback):
    def __init__(
        self,
        out_dir: str | Path,
        save_on_train: bool = False,
        save_on_validation: bool = True,
        save_on_test: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.save_on_train = save_on_train
        self.save_on_validation = save_on_validation
        self.save_on_test = save_on_test

        self.predictions = defaultdict(list)

    def on_batch_end(self, stage: str | RunningStage, outputs: dict[str, Tensor], suffix: str | None = None) -> None:
        out = {k: v.cpu().numpy().tolist() for k, v in outputs.items() if k in ("preds", "y", "idx")}
        attr = f"{stage}_preds{suffix or ''}"
        self.predictions[attr].extend(zip(*out.values(), strict=True))

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: dict[str, Tensor], batch: Any, batch_idx: int
    ) -> None:
        if self.save_on_train:
            return self.on_batch_end(RunningStage.TRAIN, outputs)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.save_on_validation:
            dl_name = RunningStage.VALIDATION if dataloader_idx == 0 else RunningStage.TEST
            return self.on_batch_end(RunningStage.VALIDATION, outputs, f"_{dl_name}")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.save_on_test:
            return self.on_batch_end(RunningStage.TEST, outputs)

    def on_epoch_end(self, stage: str | RunningStage, trainer: Trainer) -> None:
        for k in self.predictions:
            if not k.startswith(stage):
                continue

            # write to disk
            self.out_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.out_dir / f"{k}.tsv"
            (
                pd.DataFrame(self.predictions[k], columns=("preds", "y", "idx"))
                .assign(step=trainer.global_step)
                .to_csv(file_path, sep="\t", mode="a", index=False, header=not file_path.exists())
            )

            self.predictions[k] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self.on_epoch_end(RunningStage.TRAIN, trainer)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self.on_epoch_end(RunningStage.VALIDATION, trainer)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self.on_epoch_end(RunningStage.TEST, trainer)
