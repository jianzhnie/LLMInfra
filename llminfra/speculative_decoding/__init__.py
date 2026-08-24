"""Speculative decoding methods grouped by drafting strategy."""

from .base import SpeculativeDecoder
from .dflash import BlockDiffusionDrafter, DFlashDecoder, dflash_loss
from .dspark import DSparkDecoder, DSparkScheduler
from .eagle import Eagle1Speculator, Eagle2Speculator, Eagle3Speculator, EagleSpeculator
from .medusa import MedusaHead, medusa_loss
from .mtp import MTPDecoder, MultiTokenPredictionHead, mtp_loss
from .ngram import NGramSpeculator

__all__ = [
    "BlockDiffusionDrafter",
    "DFlashDecoder",
    "DSparkDecoder",
    "DSparkScheduler",
    "Eagle1Speculator",
    "Eagle2Speculator",
    "Eagle3Speculator",
    "EagleSpeculator",
    "MTPDecoder",
    "MedusaHead",
    "MultiTokenPredictionHead",
    "NGramSpeculator",
    "SpeculativeDecoder",
    "dflash_loss",
    "medusa_loss",
    "mtp_loss",
]
