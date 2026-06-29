"""
tensoranim — animated tensor visualization library
"""

from .core import TensorGrid, ThreeDTensorGrid, TensorScene3D, PALETTE, make_label, make_arrow, shape_tag
from .scenes import (
    ReshapeScene, TransposeScene, SliceScene, MatMulScene,
    ConcatScene, BroadcastScene, SoftmaxScene, AttentionScene,
)
from .scenes3d import (
    Tensor3DScene, DepthSliceScene, Permute3DScene,
    BMMScene, Conv2DScene, EinsumScene,
)
from .ops import (
    TensorState, TensorGraph,
    ReshapeOp, TransposeOp, LinearOp, MatMulOp,
    SoftmaxOp, LayerNormOp, DropoutOp, EmbeddingOp, ResidualOp,
    OpBlock,
)
from .api import (
    animate_reshape, animate_transpose, animate_slice,
    animate_matmul, animate_concat, animate_broadcast,
    animate_softmax, animate_attention,
    render_scene, render_model,
)

__all__ = [
    # core primitives
    "TensorGrid", "ThreeDTensorGrid", "TensorScene3D",
    "PALETTE", "make_label", "make_arrow", "shape_tag",
    # 2-D scenes
    "ReshapeScene", "TransposeScene", "SliceScene", "MatMulScene",
    "ConcatScene", "BroadcastScene", "SoftmaxScene", "AttentionScene",
    # 3-D scenes
    "Tensor3DScene", "DepthSliceScene", "Permute3DScene",
    "BMMScene", "Conv2DScene", "EinsumScene",
    # composable ops
    "TensorState", "TensorGraph", "OpBlock",
    "ReshapeOp", "TransposeOp", "LinearOp", "MatMulOp",
    "SoftmaxOp", "LayerNormOp", "DropoutOp", "EmbeddingOp", "ResidualOp",
    # api
    "animate_reshape", "animate_transpose", "animate_slice",
    "animate_matmul", "animate_concat", "animate_broadcast",
    "animate_softmax", "animate_attention",
    "render_scene", "render_model",
]
