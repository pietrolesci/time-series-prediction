from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.deeplob import DeepLOBConfig


@dataclass
class DeepVolConfig(DeepLOBConfig):
    @property
    def name(self) -> str:
        return f"deepvol-{self.num_levels}"

    def get_model(self) -> nn.Module:
        return DeepVol(self)


class DeepVol(nn.Module):
    """Expects OrderFlow data at each level and predicts the midprice return up to 100 steps ahead.

    [batch_size, T, 2L, 1] -> [batch_size, T, 1]
    """

    def __init__(self, config: DeepVolConfig) -> None:
        super().__init__()

        # Convolution blocks
        self.conv2 = self._get_conv_block(1, config.conv_out_channels, (1, 2), (1, 2))
        self.conv3 = self._get_conv_block(
            config.conv_out_channels, config.conv_out_channels, (1, config.num_levels), (1, 1)
        )

        # Inception blocks
        self.inp1 = self._get_inception_block(config.conv_out_channels, config.incep_out_channels, (3, 1))
        self.inp2 = self._get_inception_block(config.conv_out_channels, config.incep_out_channels, (5, 1))
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(config.conv_out_channels, config.incep_out_channels, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(config.incep_out_channels),
        )

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=config.incep_out_channels * 3,
            hidden_size=config.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Sequential(nn.Linear(config.lstm_hidden_size, config.num_targets, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimention
        # x: batch_size, n_channels, seq_len, num_levels * 4
        x = x.unsqueeze(1)

        # Convolution blocks
        # x: batch_size, n_channels, seq_len, num_levels * 2
        x = self.conv2(x)
        x = self.conv3(x)

        # Inception blocks
        # batch_size, incep_out_channels, seq_len, 1
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        # Concatenate inception blocks
        # batch_size, incep_out_channels * 3, seq_len, 1
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        # Reshape to prepare for LSTM
        # batch_size, seq_len, incep_out_channels * 3
        # Changed with a simpler to understand version
        # >>> x = x.permute(0, 2, 1, 3)
        # >>> x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x = x.squeeze(-1).permute(0, 2, 1)

        # batch_size, seq_len, lstm_hidden_size
        x, _ = self.lstm(x)

        # Take the last seq
        # batch_size, lstm_hidden_size
        x = x[:, -1, :]

        return self.fc(x)

    def _get_conv_block(
        self, in_channels: int, out_channels: int, first_kernel_size: tuple[int, int], first_stride: tuple[int, int]
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=first_kernel_size, stride=first_stride),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),
        )

    def _get_inception_block(
        self, in_channels: int, out_channels: int, mid_kernel_size: tuple[int, int]
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=mid_kernel_size, padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),
        )
