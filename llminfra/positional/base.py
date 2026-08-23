"""Abstract base class shared by all positional encodings."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BasePositionalEncoding(nn.Module, ABC):
    """Base class for position encoders used by attention implementations."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional information to ``x``."""
