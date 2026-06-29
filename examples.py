"""
examples.py
───────────
Run any scene from the command line with manim:

    manim -pql examples.py ReshapeScene
    manim -pql examples.py TransposeScene
    manim -pql examples.py SliceScene
    manim -pql examples.py MatMulScene
    manim -pql examples.py ConcatScene
    manim -pql examples.py BroadcastScene
    manim -pql examples.py SoftmaxScene
    manim -pql examples.py AttentionScene

Or from Python:
    import tensoranim as ta
    ta.animate_reshape(np.arange(12).reshape(3,4), (2,6), "out/reshape.mp4")
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from tensoranim.scenes import (
    ReshapeScene,
    TransposeScene,
    SliceScene,
    MatMulScene,
    ConcatScene,
    BroadcastScene,
    SoftmaxScene,
    AttentionScene,
)

# Manim only detects scenes whose __module__ matches the file being executed.
for _cls in (ReshapeScene, TransposeScene, SliceScene, MatMulScene,
             ConcatScene, BroadcastScene, SoftmaxScene, AttentionScene):
    _cls.__module__ = __name__

# ── configure each example ────────────────────────────────────────────────────

# reshape 3×4 → 4×3
ReshapeScene.source_data  = np.arange(12).reshape(3, 4)
ReshapeScene.target_shape = (4, 3)
ReshapeScene.show_values  = True

# transpose 2×3
TransposeScene.source_data = np.arange(1, 7).reshape(2, 3)
TransposeScene.show_values = True

# slice row 1 from a 4×5 matrix
SliceScene.source_data = np.arange(20).reshape(4, 5)
SliceScene.row_slice   = 1
SliceScene.col_slice   = None
SliceScene.show_values = True

# matrix multiply (2×3) @ (3×2)
MatMulScene.A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
MatMulScene.B = np.array([[1, 4], [2, 5], [3, 6]], dtype=float)
MatMulScene.show_values = True

# concatenate two tensors along dim 0
ConcatScene.tensors = [
    np.ones((2, 4)),
    np.ones((3, 4)) * 2,
]
ConcatScene.axis = 0

# broadcast (1,4) → (4,4)
BroadcastScene.vector_data = np.array([10, 20, 30, 40], dtype=float)
BroadcastScene.n_rows      = 4

# softmax on a 3×5 logit matrix
SoftmaxScene.source_data = np.array([
    [2.0,  1.0, 0.1, 3.0, -1.0],
    [0.5,  2.5, 1.0, 0.2,  0.8],
    [1.0,  0.0, 4.0, 0.5, -0.3],
])

# attention: Q, K, V of shape (4, 6)  → seq_len=4, head_dim=6
np.random.seed(42)
AttentionScene.Q = np.random.randn(4, 6)
AttentionScene.K = np.random.randn(4, 6)
AttentionScene.V = np.random.randn(4, 6)
