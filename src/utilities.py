# import json
import copy
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import colorlog
import numpy as np
import polars as pl
import srsly
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# NOTE: we hard-code the number of possible classes to 41 (-5, 5, 0.25)
TOTAL_NUMBER_OF_CLASSES = 41
TOTAL_NUMBER_OF_LEVELS = 15


@torch.no_grad()
def continuous_to_class(y: torch.Tensor) -> torch.Tensor:
    classes = (y + 5.0) / 0.25
    return classes.long()


@torch.no_grad()
def class_to_continuous(class_idx: torch.Tensor) -> torch.Tensor:
    return class_idx * 0.25 - 5


@torch.no_grad()
def logits_to_continuous(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1) * 0.25 - 5


@dataclass
class DictConfig:
    """Dataclass which is subscriptable like a dict"""

    def to_dict(self) -> dict[str, Any]:
        out = copy.deepcopy(self.__dict__)
        return out

    def __getitem__(self, k: str) -> Any:
        return self.__dict__[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)


def get_logger(name: str, level: Literal["error", "warning", "info", "debug"] = "info") -> logging.Logger:
    # Convert the level string to the corresponding logging level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure the logger and configure colorlog
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"  # noqa: E501
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def flatten(x: list[list]) -> list:
    return [i for j in x for i in j]


def remove_file(path: str | Path) -> None:
    path = Path(path)
    path.unlink(missing_ok=True)


def jsonl2parquet(filepath: str | Path, out_dir: str | Path) -> None:
    filepath = Path(filepath)
    assert filepath.name.endswith(".jsonl"), "Not a jsonl file"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fl = srsly.read_jsonl(filepath)
    df = pl.DataFrame({k: flatten(v) for k, v in ld_to_dl(line).items()} for line in fl)  # type: ignore
    df = df.explode(df.columns)

    df.write_parquet(out_dir / f"{filepath.name.removesuffix('.jsonl')}.parquet")


def ld_to_dl(ld: list[dict]) -> dict[str, list]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def conf_to_dict(x: DictConfig | None) -> dict:
    if x is not None:
        return OmegaConf.to_container(x)  # type: ignore
    return {}


def instantiate_from_conf(cfgs: DictConfig | list[DictConfig]) -> Any | list[Any]:
    if isinstance(cfgs, list):
        return [list(instantiate(cfg).values()) if cfg is not None else None for cfg in cfgs]
    return instantiate(cfgs)


def create_memmap_from_polars(file_path: str | Path, df: pl.DataFrame) -> None:
    file_path = Path(file_path)

    # Convert to NumPy array
    data = df.to_numpy()

    # Save to memmap
    memmap = np.memmap(str(file_path.with_suffix(".npy")), dtype=data.dtype, mode="w+", shape=data.shape)
    memmap[:] = data[:]
    memmap.flush()

    # Save metadata
    metadata = {"dtype": str(data.dtype), "shape": data.shape, "columns": df.columns}
    srsly.write_json(file_path.with_suffix(".meta"), metadata)


def read_memmap(file_path: str | Path) -> tuple[np.memmap, dict]:
    file_path = Path(file_path)

    # Load metadata
    meta: dict = srsly.read_json(file_path.with_suffix(".meta"))  # type: ignore

    # Load memmap
    memmap = np.memmap(str(file_path.with_suffix(".npy")), dtype=meta["dtype"], mode="r", shape=tuple(meta["shape"]))

    return memmap, meta
