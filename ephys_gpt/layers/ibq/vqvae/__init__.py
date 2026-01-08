from .quantize import (
    VectorQuantizer,
    VectorQuantizer2,
    VectorQuantizerWithWeakTrick,
    VectorQuantizerWithKld,
    GumbelQuantize,
    VectorQuantizerCosine,
    IndexPropagationQuantize,
)

__all__ = [
    "VectorQuantizer",
    "VectorQuantizer2",
    "VectorQuantizerWithWeakTrick",
    "VectorQuantizerWithKld",
    "GumbelQuantize",
    "VectorQuantizerCosine",
    "IndexPropagationQuantize",
]
