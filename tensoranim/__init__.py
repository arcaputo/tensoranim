"""
tensoranim/__init__.py
──────────────────────
Public API for tensoranim.

Quick-start
───────────
    import numpy as np
    import tensoranim as ta

    # render a reshape animation to an MP4 file
    ta.animate_reshape(
        source_data  = np.arange(12).reshape(3, 4),
        target_shape = (2, 6),
        output_path  = "reshape.mp4",
        quality      = "medium_quality",   # low / medium / high / production
    )

    # render a matmul animation
    ta.animate_matmul(
        A = np.random.randn(3, 4),
        B = np.random.randn(4, 2),
        output_path = "matmul.mp4",
    )

Or use scene classes directly with Manim's CLI:
    manim -pql my_script.py ReshapeScene
"""

from .core import TensorGrid, PALETTE, make_label, make_arrow, shape_tag
from .scenes import (
    ReshapeScene,
    TransposeScene,
    SliceScene,
    MatMulScene,
    ConcatScene,
    BroadcastScene,
    SoftmaxScene,
    AttentionScene,
)
from .api import (
    animate_reshape,
    animate_transpose,
    animate_slice,
    animate_matmul,
    animate_concat,
    animate_broadcast,
    animate_softmax,
    animate_attention,
    render_scene,
)

__all__ = [
    # primitives
    "TensorGrid",
    "PALETTE",
    "make_label",
    "make_arrow",
    "shape_tag",
    # scene classes
    "ReshapeScene",
    "TransposeScene",
    "SliceScene",
    "MatMulScene",
    "ConcatScene",
    "BroadcastScene",
    "SoftmaxScene",
    "AttentionScene",
    # convenience API
    "animate_reshape",
    "animate_transpose",
    "animate_slice",
    "animate_matmul",
    "animate_concat",
    "animate_broadcast",
    "animate_softmax",
    "animate_attention",
    "render_scene",
]
