"""Educational positional encoding and long-context scaling utilities.

This package provides PyTorch implementations of the position-related
components referenced by mainstream attention architectures:

- `rope`: Rotary Position Embedding (RoPE) core
- `scaling`: YaRN / dynamic-NTK / partial / interpolation / LongRoPE variants
- `alibi`: Attention with Linear Biases (ALiBi)
- `two_d`: two-dimensional block position embedding
- `factory`: name-based `build_positional_encoding` factory

The implementations are intended for teaching and small-scale experiments.
Production deployments should compare against the official kernels and
Transformers implementations, especially for exact YaRN coefficients.
"""

from .alibi import ALiBiBias
from .base import BasePositionalEncoding
from .classic import (
    LearnedAbsolutePositionEmbedding,
    NoPositionEncoding,
    SinusoidalPositionEmbedding,
    T5RelativePositionBias,
)
from .factory import build_positional_encoding, list_positional_encodings
from .multimodal_rope import MultiModalRotaryPositionEmbedding
from .rope_scaling import (
    LONGROPE_PRESETS,
    DynamicNTKRotaryEmbedding,
    LongRoPEPreset,
    LongRoPEScaledRotaryEmbedding,
    PartialRotaryPositionEmbedding,
    PositionInterpolation,
    YaRNParameters,
    YaRNScaledRotaryEmbedding,
    get_longrope_preset,
    register_longrope_preset,
)
from .rotary import RotaryPositionEmbedding, apply_rotary_pos_emb
from .two_dimensional import TwoDimensionalPositionEmbedding

__all__ = [
    "LONGROPE_PRESETS",
    "ALiBiBias",
    "BasePositionalEncoding",
    "DynamicNTKRotaryEmbedding",
    "LearnedAbsolutePositionEmbedding",
    "LongRoPEPreset",
    "LongRoPEScaledRotaryEmbedding",
    "MultiModalRotaryPositionEmbedding",
    "NoPositionEncoding",
    "PartialRotaryPositionEmbedding",
    "PositionInterpolation",
    "RotaryPositionEmbedding",
    "SinusoidalPositionEmbedding",
    "T5RelativePositionBias",
    "TwoDimensionalPositionEmbedding",
    "YaRNParameters",
    "YaRNScaledRotaryEmbedding",
    "apply_rotary_pos_emb",
    "build_positional_encoding",
    "get_longrope_preset",
    "list_positional_encodings",
    "register_longrope_preset",
]
