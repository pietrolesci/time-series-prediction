from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utilities import TOTAL_NUMBER_OF_LEVELS, continuous_to_class, read_memmap


class LOBDataset(Dataset):
    def __init__(
        self,
        file_path: str | Path,
        window_size: int = 100,
        num_levels: int = TOTAL_NUMBER_OF_LEVELS,
        use_prev_y: bool = False,
        is_classification: bool = False,
    ) -> None:
        super().__init__()
        self.data, self.metadata = read_memmap(file_path)

        self._window_size = window_size
        self._num_levels = num_levels

        # This is typically fixed throughout training
        self.use_prev_y = use_prev_y
        self.is_classification = is_classification

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start_idx = max(0, index - self._window_size + 1)
        data_window = self.data[start_idx : index + 1]

        # Extract features (X) and target (y)
        max_levels = (self.num_levels * 4) + 1  # 4 features per level
        X = torch.tensor(data_window[:, 1:max_levels], dtype=torch.float32)  # Exclude "y" column

        y = torch.tensor(data_window[:, 0], dtype=torch.float32)  # Only "y" column
        if not self.use_prev_y:
            y = y[-1]
        if self.is_classification:
            y = continuous_to_class(y)

        out = {"X": X, "y": y, "idx": torch.arange(start_idx, index + 1, 1)}
        return out

    def __len__(self) -> int:
        return len(self.data)

    @property
    def num_levels(self) -> int:
        return self._num_levels
