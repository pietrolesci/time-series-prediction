from copy import copy
from dataclasses import dataclass
from functools import partial
from os import cpu_count
from pathlib import Path
from typing import Any, Literal

import torch
from lightning.pytorch import LightningDataModule
from torch.nn.functional import pad
from torchdata.stateful_dataloader import StatefulDataLoader

from src.dataset import LOBDataset, OrderFlowDataset
from src.module import RunningStage
from src.utilities import DictConfig, get_logger

logger = get_logger("data")


@dataclass
class DataloaderConfig(DictConfig):
    batch_size: int | None = None
    eval_batch_size: int | None = None
    num_workers: int | None = cpu_count()
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = False
    multiprocessing_context: str | None = None
    shuffle: bool = False
    prefetch_factor: int | None = None
    snapshot_every_n_steps: int | None = None

    def get_train_kwargs(self) -> dict:
        kwargs = copy(self.to_dict())
        kwargs.pop("eval_batch_size")
        return kwargs

    def get_val_kwargs(self) -> dict:
        kwargs = copy(self.to_dict())
        kwargs["batch_size"] = kwargs.pop("eval_batch_size")
        kwargs["shuffle"] = False
        kwargs["drop_last"] = False
        return kwargs


class DataModule(LightningDataModule):
    train_ds: LOBDataset | OrderFlowDataset
    val_ds: LOBDataset | OrderFlowDataset
    test_ds: LOBDataset | OrderFlowDataset | None

    _should_mask_padding: bool | None = None

    def __init__(
        self,
        dataloader_config: DataloaderConfig,
        window_size: int,
        num_levels: int,
        train_data_path: str | Path,
        val_data_path: str | Path,
        test_data_path: str | Path | None = None,
        data_repr: Literal["lob", "orderflow"] = "lob",
    ) -> None:
        super().__init__()
        self.train_data_path = Path(train_data_path) if train_data_path else train_data_path
        self.val_data_path = Path(val_data_path) if val_data_path else val_data_path
        self.test_data_path = Path(test_data_path) if test_data_path else test_data_path

        self.dataloader_config = dataloader_config
        self.data_kwargs = {"window_size": window_size, "num_levels": num_levels}
        self.collator = partial(collate_fn, max_len=window_size)
        self.dataset_class = LOBDataset if data_repr == "lob" else OrderFlowDataset

        self.save_hyperparameters()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None) -> None:
        self.train_ds = self.dataset_class(self.train_data_path, **self.data_kwargs)
        self.val_ds = self.dataset_class(self.val_data_path, **self.data_kwargs)

        logger.info(f"Train dataset loaded: {len(self.train_ds)=}")
        logger.info(f"Validation dataset loaded: {len(self.val_ds)=}")

        if self.test_data_path:
            self.test_ds = self.dataset_class(self.test_data_path, **self.data_kwargs)
            logger.info(f"Test dataset loaded: {len(self.test_ds)=}")

    def train_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(self.train_ds, **self.dataloader_config.get_train_kwargs(), collate_fn=self.collator)

    def val_dataloader(self) -> list[StatefulDataLoader]:
        out = [StatefulDataLoader(self.val_ds, **self.dataloader_config.get_val_kwargs(), collate_fn=self.collator)]
        tdl = self.test_dataloader()
        if tdl:
            out.append(tdl)
        return out

    def test_dataloader(self) -> StatefulDataLoader | None:
        if self.test_ds:
            return StatefulDataLoader(self.test_ds, **self.dataloader_config.get_val_kwargs(), collate_fn=self.collator)

    def transfer_batch_to_device(
        self, batch: dict[str, torch.Tensor], device: torch.device, dataloader_idx: int
    ) -> Any:
        # Keep idx on cpu
        idx = batch.pop("idx")
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        batch["idx"] = idx
        return batch

    def show_batch(self, stage: str | RunningStage) -> dict[str, torch.Tensor]:
        return next(iter(getattr(self, f"{stage}_dataloader")()))

    @property
    def num_targets(self) -> int:
        return getattr(self.train_ds, "num_targets", 1)

    @property
    def num_features(self) -> int:
        return self.train_ds.num_features

    @property
    def window_size(self) -> int:
        return self.data_kwargs["window_size"]

    @property
    def num_levels(self) -> int:
        return self.data_kwargs["num_levels"]


def collate_fn(batch: list[dict[str, torch.Tensor]], max_len: int) -> dict:
    padded_X = []
    for item in batch:
        x = item["X"]
        pad_size = max_len - x.shape[0]
        if pad_size > 0:
            # pad the past such that the most recent observation is at the end
            x = pad(x, (0, 0, pad_size, 0), "constant", 0)  # e.g., [[0, 0, 0], [0, 0, 0], [x, x, x]]
        padded_X.append(x)

    return {
        "y": torch.stack([item["y"] for item in batch]),
        "X": torch.stack(padded_X),
        "idx": torch.stack([item["idx"] for item in batch]),
    }

    # # Pad each instance to the maximum length
    # new_batch = ld_to_dl(batch)
    # print(new_batch["y"])
    # # Pad the temporal dimension

    # # Stack the padded instances
    # return {"X": torch.concat(padded_X), "y": torch.cat(new_batch["y"]), "idx": torch.cat(new_batch["idx"])}
