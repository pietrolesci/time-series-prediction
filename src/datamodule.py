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

from src.dataset import LOBDataset
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

    def get_train_kwargs(self) -> dict:
        kwargs = copy(self.to_dict())
        kwargs.pop("eval_batch_size")
        return kwargs

    def get_val_kwargs(self) -> dict:
        kwargs = copy(self.to_dict())
        kwargs["batch_size"] = kwargs.pop("eval_batch_size")
        kwargs["shuffle"] = False
        return kwargs


class DataModule(LightningDataModule):
    train_ds: LOBDataset
    val_ds: LOBDataset
    test_ds: LOBDataset | None

    _should_mask_padding: bool | None = None

    def __init__(
        self,
        dataloader_config: DataloaderConfig,
        window_size: int,
        num_levels: int,
        use_prev_y: bool,
        is_classification: bool,
        train_data_path: str | Path,
        val_data_path: str | Path,
        test_data_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.train_data_path = Path(train_data_path) if train_data_path else train_data_path
        self.val_data_path = Path(val_data_path) if val_data_path else val_data_path
        self.test_data_path = Path(test_data_path) if test_data_path else test_data_path

        self.dataloader_config = dataloader_config
        self.data_kwargs = {
            "window_size": window_size,
            "num_levels": num_levels,
            "use_prev_y": use_prev_y,
            "is_classification": is_classification,
        }

        self.collator = partial(collate_fn, max_len=window_size)

        self.save_hyperparameters()

    @property
    def should_mask_padding(self) -> bool:
        if self._should_mask_padding is None:
            self._should_mask_padding = not self.data_kwargs["is_classification"] and self.data_kwargs["use_prev_y"]
        return self._should_mask_padding

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        self.train_ds = LOBDataset(self.train_data_path, **self.data_kwargs)
        self.val_ds = LOBDataset(self.val_data_path, **self.data_kwargs)

        logger.info(f"Train dataset loaded: {len(self.train_ds)=}")
        logger.info(f"Validation dataset loaded: {len(self.val_ds)=}")

        if self.test_data_path:
            self.test_ds = LOBDataset(self.test_data_path, **self.data_kwargs)
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


def collate_fn(batch: list[dict[str, torch.Tensor]], max_len: int) -> dict:
    # Find the maximum length in the batch
    # max_len = max(item["X"].shape[0] for item in batch)

    # Pad each instance to the maximum length
    padded_X = []
    padded_y = []
    padded_idx = []
    for item in batch:
        X = item["X"]
        y = item["y"]
        idx = item["idx"]
        pad_size = max_len - X.shape[0]
        if pad_size > 0:
            X = pad(X, (0, 0, pad_size, 0), "constant", 0)  # e.g., [[0, 0, 0], [0, 0, 0], [x, x, x]]
            idx = pad(idx, (pad_size, 0), "constant", -100)  # e.g., [-100, -100, idx]
            if y.dim() > 0:
                y = pad(y, (pad_size, 0), "constant", -100)  # e.g., [-100, -100, y]

        # pad the past such that the most recent observation is at the end
        padded_X.append(X)
        padded_y.append(y)
        padded_idx.append(idx)

    # Stack the padded instances
    out = {"X": torch.stack(padded_X), "y": torch.stack(padded_y), "idx": torch.stack(padded_idx)}

    return out
