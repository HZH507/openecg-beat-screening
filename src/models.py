"""Neural network definitions used in the project."""
from __future__ import annotations

import torch
from torch import nn

from config import NUM_CLASSES


class ConvBlock1d(nn.Module):
    """Conv -> BatchNorm -> ReLU -> max-pool (stride 2)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECGCNN(nn.Module):
    """Compact four-stage 1-D CNN for 256-sample ECG beats.

    Input:  (B, 1, 256)
    Output: (B, 4) logits
    """

    def __init__(self, num_classes: int = NUM_CLASSES, width: int = 32,
                 dropout: float = 0.4, n_extra: int = 3) -> None:
        super().__init__()
        self.n_extra = n_extra
        self.features = nn.Sequential(
            ConvBlock1d(1, width, 7),           # 256 -> 128
            ConvBlock1d(width, width * 2, 5),   # 128 -> 64
            ConvBlock1d(width * 2, width * 4, 5),  # 64 -> 32
            ConvBlock1d(width * 4, width * 4, 3),  # 32 -> 16
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(dropout))
        if n_extra > 0:
            self.head = nn.Sequential(
                nn.Linear(width * 4 + n_extra, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )
        else:
            self.head = nn.Linear(width * 4, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                extra: torch.Tensor | None = None) -> torch.Tensor:
        z = self.pool(self.features(x))
        if self.n_extra > 0:
            if extra is None:
                extra = torch.zeros((len(z), self.n_extra), dtype=z.dtype,
                                    device=z.device)
            z = torch.cat([z, extra], dim=1)
        return self.head(z)


class ECGCNNLight(nn.Module):
    """A small, BN-free 1-D CNN used as the final lightweight classifier.

    Batch normalisation made the tiny inter-patient model unstable in pilot
    runs (validation collapse after a few epochs); removing BN and reducing
    width gives a much more robust optimisation path, at the cost of a little
    capacity.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                extra: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.features(x))


class MLPBaseline(nn.Module):
    """Parameter-matched-ish fully connected baseline on raw samples."""

    def __init__(self, num_classes: int = NUM_CLASSES, length: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(length, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DnCNN1d(nn.Module):
    """One-dimensional DnCNN that predicts the noise component.

    The output is ``x - noise`` so that the network only has to model the
    additive noise, which is easier to optimise than the full ECG waveform.
    """

    def __init__(self, channels: int = 64, depth: int = 7, kernel: int = 15) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(1, channels, kernel, padding=kernel // 2, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv1d(channels, channels, kernel, padding=kernel // 2,
                          bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv1d(channels, 1, kernel, padding=kernel // 2,
                                bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
