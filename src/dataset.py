from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utilities import TOTAL_NUMBER_OF_LEVELS, read_memmap


class LOBDataset(Dataset):
    def __init__(self, file_path: str | Path, window_size: int = 100, num_levels: int = TOTAL_NUMBER_OF_LEVELS) -> None:
        super().__init__()
        self.data, self.metadata = read_memmap(file_path)
        self.window_size = window_size
        self._num_levels = num_levels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start_idx = max(0, index - self.window_size + 1)
        data_window = self.data[start_idx : index + 1]

        max_levels = (self.num_levels * 4) + 1  # 4 features per level
        X = torch.tensor(data_window[:, 1:max_levels], dtype=torch.float32)  # Exclude "y" column
        y = torch.tensor(data_window[-1, 0], dtype=torch.float32)  # Only "y" column
        out = {"X": X, "y": y, "idx": torch.arange(start_idx, index + 1, 1)}
        return out

    def __len__(self) -> int:
        return len(self.data)

    @property
    def num_levels(self) -> int:
        return self._num_levels

    @property
    def num_features(self) -> int:
        # askOrderFlow, bidOrderFlow per each level
        return self.num_levels * 4


class OrderFlowDataset(Dataset):
    def __init__(self, file_path: str | Path, window_size: int = 100, num_levels: int = TOTAL_NUMBER_OF_LEVELS) -> None:
        super().__init__()
        self.data, self.metadata = read_memmap(file_path)
        self._num_targets = sum(i.startswith("y") for i in self.metadata["columns"])

        self.window_size = window_size
        self._num_levels = num_levels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start_idx = max(0, index - self.window_size + 1)
        data_window = self.data[start_idx : index + 1, :]

        y = data_window[-1, : self._num_targets]
        X = data_window[:, self._num_targets : self._num_targets + self.num_features]

        return {
            "y": torch.tensor(y, dtype=torch.float32),
            "X": torch.tensor(X, dtype=torch.float32),
            "idx": torch.tensor(index, dtype=torch.int64),
        }

    def __len__(self) -> int:
        return len(self.data)

    @property
    def num_levels(self) -> int:
        return self._num_levels

    @property
    def num_targets(self) -> int:
        return self._num_targets

    @property
    def num_features(self) -> int:
        # askOrderFlow, bidOrderFlow per each level
        return self.num_levels * 2
