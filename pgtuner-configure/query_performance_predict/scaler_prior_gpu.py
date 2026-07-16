# -*- coding: utf-8 -*-
"""Feature scaler for QPP when the head is a variable-width prior encoding + dataset tail."""

from __future__ import annotations

import torch


class Scaler_minmax_prior_gpu:
    """
    First ``head_dim`` columns: min–max with fixed ``head_min`` / ``head_max`` (broadcastable).
    Remaining columns: z-normalize using ``fit`` statistics (same contract as ``Scaler_minmax_new_gpu``).
    """

    def __init__(self, head_dim: int, head_min: torch.Tensor, head_max: torch.Tensor, device: torch.device):
        self.head_dim = int(head_dim)
        self.head_min = head_min.to(device=device, dtype=torch.float32)
        self.head_max = head_max.to(device=device, dtype=torch.float32)
        self.device = device
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> None:
        if self.head_dim != 0:
            tail = data[:, self.head_dim :]
            self.mean = torch.mean(tail, dim=0)
            self.std = torch.std(tail, dim=0) + 1e-8
        else:
            self.mean = torch.mean(data, dim=0)
            self.std = torch.std(data, dim=0) + 1e-8

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.head_dim != 0:
            head = (data[:, : self.head_dim] - self.head_min) / (self.head_max - self.head_min + 1e-12)
            std_safe = self.std + 1e-8
            tail = (data[:, self.head_dim :] - self.mean) / std_safe
            return torch.cat((head, tail), dim=1)
        std_safe = self.std + 1e-8
        return (data - self.mean) / std_safe

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.head_dim != 0:
            head = data[:, : self.head_dim] * (self.head_max - self.head_min + 1e-12) + self.head_min
            tail = data[:, self.head_dim :] * self.std + self.mean
            return torch.cat((head, tail), dim=1)
        return data * self.std + self.mean

    def save_parameters(self, _minmax_path: str | None, standard_path: str) -> None:
        torch.save({"mean": self.mean, "std": self.std}, standard_path)

    def load_parameters(self, _minmax_path: str | None, standard_path: str, device: torch.device) -> None:
        standard_params = torch.load(standard_path, map_location=device)
        self.mean = standard_params["mean"].to(device)
        self.std = standard_params["std"].to(device)
