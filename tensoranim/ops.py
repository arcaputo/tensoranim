"""
tensoranim/ops.py
─────────────────
Composable, reusable operation blocks for model-level animation.

Design
──────
Each ML operation is represented as an OpBlock — a plain Python object (not a
Scene) that:
  - Holds its configuration (kernel size, target shape, etc.)
  - Accepts TensorState inputs and produces a TensorState output
  - Can animate itself onto any scene via .animate(scene, inputs, output)

This lets you build a list (or DAG) of OpBlocks describing a model's forward
pass and hand that list to ModelScene, which renders a single coherent video.

Quick example
─────────────
    import numpy as np
    import tensoranim as ta
    from tensoranim.ops import (
        TensorState, ReshapeOp, LinearOp, SoftmaxOp, ModelScene
    )

    x = TensorState(np.random.randn(4, 8),  name="x",      color=ta.PALETTE["fill_0"])
    W = TensorState(np.random.randn(8, 16), name="W",      color=ta.PALETTE["fill_1"])

    ops = [
        LinearOp(name="fc1"),
        ReshapeOp(target_shape=(2, 32), name="reshape"),
        SoftmaxOp(name="softmax"),
    ]

    ta.render_model(x, ops, weights=[W], output_path="model.mp4")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Dict, Any, Sequence

from manim import (
    VGroup, Text, Arrow, Rectangle,
    FadeIn, FadeOut, FadeTransform, Transform, AnimationGroup,
    Create, Write, ReplacementTransform,
    LEFT, RIGHT, UP, DOWN, ORIGIN,
    SMALL_BUFF, MED_SMALL_BUFF, MED_LARGE_BUFF, LARGE_BUFF,
    PI,
)

from .core import (
    TensorGrid, ThreeDTensorGrid, TensorScene3D,
    make_label, make_arrow, shape_tag,
    PALETTE, _resolve_color,
)


# ══════════════════════════════════════════════════════════════════════════════
# TensorState — the object that flows between ops
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TensorState:
    """
    A named tensor with display metadata.

    This is what flows between OpBlocks in a model graph — the "edge" in the
    computation graph, carrying both the actual numpy data and the visual style
    to use when rendering it.

    Parameters
    ----------
    data   : np.ndarray  — the actual tensor values
    name   : str         — label shown in the animation (e.g. "x", "h1", "logits")
    color  : str         — hex color for the grid fill
    shape  : tuple       — read-only shorthand for data.shape
    """
    data:  np.ndarray
    name:  str   = "tensor"
    color: str   = PALETTE["fill_0"]

    def __post_init__(self):
        self.data = np.asarray(self.data)

    @property
    def shape(self) -> tuple:
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def make_grid(self, cell_size: float = 0.5, show_values: bool = False,
                  **kwargs) -> TensorGrid:
        """Build a TensorGrid for this state (handles 1-D/2-D/3-D)."""
        return TensorGrid(
            self.data,
            cell_size=cell_size,
            fill_color=self.color,
            show_values=show_values,
            **kwargs,
        )

    def make_3d_grid(self, cell_size: float = 0.44, **kwargs) -> ThreeDTensorGrid:
        """Build a ThreeDTensorGrid (requires 3-D data)."""
        assert self.ndim == 3, "make_3d_grid requires 3-D data"
        return ThreeDTensorGrid(
            self.data,
            cell_size=cell_size,
            fill_color=self.color,
            **kwargs,
        )

    def reshaped(self, new_shape: tuple, name: str = None,
                 color: str = None) -> "TensorState":
        return TensorState(
            data=self.data.reshape(new_shape),
            name=name or self.name,
            color=color or self.color,
        )

    def __repr__(self):
        return f"TensorState({self.name!r}, shape={self.shape})"


# ══════════════════════════════════════════════════════════════════════════════
# Layout helpers
# ══════════════════════════════════════════════════════════════════════════════

# Maximum cell size for a grid given available screen width/height
def _auto_cell_size(shape: tuple, max_w: float = 3.0, max_h: float = 2.5) -> float:
    """Compute cell size so the grid fits in (max_w x max_h) Manim units."""
    ndim = len(shape)
    cols = shape[-1] if ndim >= 1 else 1
    rows = shape[-2] if ndim >= 2 else 1
    return min(max_w / max(cols, 1), max_h / max(rows, 1), 0.6)


def _grid_for_state(state: TensorState, cell_size: float = None,
                    show_values: bool = False) -> TensorGrid:
    cs = cell_size or _auto_cell_size(state.shape)
    return state.make_grid(cell_size=cs, show_values=show_values)


# ══════════════════════════════════════════════════════════════════════════════
# OpBlock — base class for all operations
# ══════════════════════════════════════════════════════════════════════════════

class OpBlock:
    """
    Base class for all animatable tensor operations.

    Subclasses must implement:
      - forward(*inputs: TensorState) -> TensorState
      - animate(scene, inputs, output, in_grids, out_grid)

    `animate` receives ready-built grids and a reference to the scene so it
    can call self.play(), self.wait(), etc.  It should NOT reposition the grids
    — ModelScene handles layout.  It SHOULD handle highlighting, arrows, and
    step subtitles.
    """

    def __init__(self, name: str = "", output_color: str = None):
        self.name = name
        self.output_color = output_color  # if None, inherit from first input

    # ── subclass interface ────────────────────────────────────────────────────

    def forward(self, *inputs: TensorState) -> TensorState:
        raise NotImplementedError

    def animate(self, scene, inputs: List[TensorState], output: TensorState,
                in_grids: List[TensorGrid], out_grid: TensorGrid):
        """Animate this op onto `scene`. Default: simple arrow + fade-in."""
        if in_grids:
            arrow = make_arrow(in_grids[-1], out_grid,
                               label=self.name or self.__class__.__name__)
            scene.play(Create(arrow), FadeIn(out_grid), run_time=0.6)
        else:
            scene.play(FadeIn(out_grid), run_time=0.4)

    # ── helpers available to subclasses ──────────────────────────────────────

    def _subtitle(self, scene, text: str) -> Text:
        t = Text(text, font_size=17, color=PALETTE["dim_tag"])
        t.to_edge(DOWN, buff=0.35)
        return t

    def _output_color(self, inputs: List[TensorState]) -> str:
        if self.output_color:
            return self.output_color
        return inputs[0].color if inputs else PALETTE["fill_0"]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"


# ══════════════════════════════════════════════════════════════════════════════
# Concrete OpBlocks
# ══════════════════════════════════════════════════════════════════════════════

class ReshapeOp(OpBlock):
    """tensor.reshape(target_shape)"""

    def __init__(self, target_shape: tuple, name: str = "reshape",
                 output_color: str = None):
        super().__init__(name=name, output_color=output_color)
        self.target_shape = target_shape

    def forward(self, x: TensorState) -> TensorState:
        return TensorState(
            data=x.data.reshape(self.target_shape),
            name=self.name or x.name,
            color=self._output_color([x]),
        )

    def animate(self, scene, inputs, output, in_grids, out_grid):
        src_grid = in_grids[0]
        flat = inputs[0].data.flatten()
        src_cells = [src_grid.cell(*np.unravel_index(i, inputs[0].shape))
                     for i in range(flat.size)]
        tgt_cells = [out_grid.cell(*np.unravel_index(i, output.shape))
                     for i in range(flat.size)]

        arrow = make_arrow(src_grid, out_grid, label="reshape")
        scene.play(Create(arrow), FadeIn(out_grid), run_time=0.4)

        anims = [Transform(s.copy(), t)
                 for s, t in zip(src_cells, tgt_cells)]
        scene.play(AnimationGroup(*anims, lag_ratio=0.05), run_time=0.8)


class TransposeOp(OpBlock):
    """tensor.T  (2-D only)"""

    def __init__(self, name: str = ".T", output_color: str = None):
        super().__init__(name=name, output_color=output_color)

    def forward(self, x: TensorState) -> TensorState:
        assert x.ndim == 2
        return TensorState(data=x.data.T, name=self.name or x.name,
                           color=self._output_color([x]))

    def animate(self, scene, inputs, output, in_grids, out_grid):
        src = in_grids[0]; rows, cols = inputs[0].shape
        arrow = make_arrow(src, out_grid, label=".T")
        scene.play(Create(arrow), run_time=0.3)
        anims = []
        for r in range(rows):
            for c in range(cols):
                anims.append(Transform(src.cell(r, c).copy(), out_grid.cell(c, r)))
        scene.play(AnimationGroup(*anims, lag_ratio=0.04), FadeIn(out_grid),
                   run_time=0.7)


class LinearOp(OpBlock):
    """
    Fully-connected linear layer:  output = input @ W.T  (+ bias)

    weight : TensorState  shape (out_features, in_features)
    bias   : TensorState  shape (out_features,)  or None
    """

    def __init__(self, weight: TensorState, bias: TensorState = None,
                 name: str = "linear", output_color: str = PALETTE["fill_1"]):
        super().__init__(name=name, output_color=output_color)
        self.weight = weight
        self.bias   = bias

    def forward(self, x: TensorState) -> TensorState:
        out = x.data @ self.weight.data.T
        if self.bias is not None:
            out = out + self.bias.data
        return TensorState(data=out, name=self.name,
                           color=self._output_color([x]))

    def animate(self, scene, inputs, output, in_grids, out_grid):
        x_grid = in_grids[0]
        W      = self.weight
        x_in   = inputs[0]

        # show W as a small grid between input and output
        cs  = _auto_cell_size(W.shape, max_w=2.0, max_h=1.5)
        w_grid = _grid_for_state(W, cell_size=cs)
        w_lbl  = make_label(f"W {W.shape}", 14)

        w_grid.move_to(x_grid.get_center() + RIGHT *
                       (x_grid.get_width() / 2 + out_grid.get_width() / 2 + 0.8) / 2
                       + UP * 1.0)
        w_lbl.next_to(w_grid, UP, buff=SMALL_BUFF)

        scene.play(FadeIn(w_grid), FadeIn(w_lbl), run_time=0.3)

        m = x_in.shape[0] if x_in.ndim >= 2 else 1
        k = x_in.shape[-1]
        n = W.shape[0]

        # highlight row/col pairs for first output position, then sweep
        arr1 = make_arrow(x_grid, out_grid, label=self.name)
        scene.play(Create(arr1), FadeIn(out_grid), run_time=0.4)

        for i in range(min(m, 3)):   # animate first 3 rows
            row_hi = [x_grid.cell(i if x_in.ndim == 2 else 0, kk).animate.set_fill(
                          color=PALETTE["fill_2"], opacity=0.9)
                      for kk in range(min(k, 8))]
            scene.play(AnimationGroup(*row_hi, lag_ratio=0.04), run_time=0.3)
            for j in range(min(n, 4)):
                scene.play(
                    out_grid.cell(i if output.ndim == 2 else 0, j).animate.set_fill(
                        color=self.output_color or PALETTE["fill_1"], opacity=0.9),
                    run_time=0.12,
                )
            reset = [x_grid.cell(i if x_in.ndim == 2 else 0, kk).animate.set_fill(
                         color=x_in.color, opacity=0.85)
                     for kk in range(min(k, 8))]
            scene.play(AnimationGroup(*reset, lag_ratio=0), run_time=0.1)

        scene.play(FadeOut(w_grid), FadeOut(w_lbl), run_time=0.2)


class MatMulOp(OpBlock):
    """A @ B = C  (explicit second tensor)"""

    def __init__(self, B: TensorState, name: str = "matmul",
                 output_color: str = PALETTE["fill_4"]):
        super().__init__(name=name, output_color=output_color)
        self.B = B

    def forward(self, A: TensorState) -> TensorState:
        C = A.data @ self.B.data
        return TensorState(data=C, name=self.name, color=self.output_color)

    def animate(self, scene, inputs, output, in_grids, out_grid):
        a_grid = in_grids[0]
        A, B   = inputs[0], self.B
        C_data = output.data
        m, k   = A.shape[-2], A.shape[-1]
        k2, n  = B.shape

        cs = _auto_cell_size(B.shape, max_w=1.8, max_h=2.0)
        b_grid = _grid_for_state(B, cell_size=cs)
        b_lbl  = make_label(f"B {B.shape}", 14)
        b_grid.next_to(a_grid, RIGHT, buff=0.5)
        b_lbl.next_to(b_grid, UP, buff=SMALL_BUFF)
        scene.play(FadeIn(b_grid), FadeIn(b_lbl), run_time=0.3)

        arrow = make_arrow(b_grid, out_grid, label="@")
        scene.play(Create(arrow), FadeIn(out_grid), run_time=0.35)

        for i in range(min(m, 3)):
            for j in range(min(n, 3)):
                hi_r = [a_grid.cell(i, kk).animate.set_fill(
                            color=PALETTE["fill_2"], opacity=0.9)
                        for kk in range(min(k, 8))]
                hi_c = [b_grid.cell(kk, j).animate.set_fill(
                            color=PALETTE["fill_3"], opacity=0.9)
                        for kk in range(min(k, 8))]
                scene.play(AnimationGroup(*(hi_r + hi_c), lag_ratio=0.02),
                           run_time=0.25)
                scene.play(
                    out_grid.cell(i, j).animate.set_fill(
                        color=self.output_color, opacity=0.9),
                    run_time=0.15)
                reset = ([a_grid.cell(i, kk).animate.set_fill(
                              color=A.color, opacity=0.85)
                          for kk in range(min(k, 8))] +
                         [b_grid.cell(kk, j).animate.set_fill(
                              color=B.color, opacity=0.85)
                          for kk in range(min(k, 8))])
                scene.play(AnimationGroup(*reset, lag_ratio=0), run_time=0.1)

        scene.play(FadeOut(b_grid), FadeOut(b_lbl), run_time=0.2)


class SoftmaxOp(OpBlock):
    """Row-wise softmax with heat-map transition."""

    def __init__(self, dim: int = -1, name: str = "softmax",
                 output_color: str = None):
        super().__init__(name=name, output_color=output_color)
        self.dim = dim

    def forward(self, x: TensorState) -> TensorState:
        data = x.data
        shifted = data - data.max(axis=self.dim, keepdims=True)
        e = np.exp(shifted)
        probs = e / e.sum(axis=self.dim, keepdims=True)
        return TensorState(data=probs, name=self.name or x.name,
                           color=self._output_color([x]))

    def animate(self, scene, inputs, output, in_grids, out_grid):
        src = in_grids[0]; probs = output.data

        def prob_color(idx, val):
            t = float(np.clip(val, 0, 1))
            r = int(np.clip(20 + t * 30,  0, 255))
            g = int(np.clip(80 + t * 170, 0, 255))
            b = int(np.clip(60 + t * 20,  0, 255))
            return f"#{r:02x}{g:02x}{b:02x}"

        arrow = make_arrow(src, out_grid, label="softmax")
        scene.play(Create(arrow), FadeIn(out_grid), run_time=0.4)

        rows = probs.shape[0] if probs.ndim >= 2 else 1
        cols = probs.shape[-1]
        for r in range(rows):
            anims = [
                out_grid.cell(r, c).animate.set_fill(
                    color=prob_color((r, c), probs[r, c]), opacity=0.88)
                for c in range(cols)
            ]
            scene.play(AnimationGroup(*anims, lag_ratio=0.06), run_time=0.35)


class LayerNormOp(OpBlock):
    """
    Layer normalisation: normalise the last dimension, show mean/variance
    as an animated sweep before the final rescaled output.
    """

    def __init__(self, eps: float = 1e-5, name: str = "layer_norm",
                 output_color: str = PALETTE["fill_3"]):
        super().__init__(name=name, output_color=output_color)
        self.eps = eps

    def forward(self, x: TensorState) -> TensorState:
        mu  = x.data.mean(axis=-1, keepdims=True)
        std = x.data.std(axis=-1, keepdims=True)
        out = (x.data - mu) / (std + self.eps)
        return TensorState(data=out, name=self.name, color=self.output_color)

    def animate(self, scene, inputs, output, in_grids, out_grid):
        src  = in_grids[0]
        rows = inputs[0].shape[0] if inputs[0].ndim >= 2 else 1
        cols = inputs[0].shape[-1]

        arrow = make_arrow(src, out_grid, label="LN")
        scene.play(Create(arrow), FadeIn(out_grid), run_time=0.4)

        # sweep each row: highlight in amber (mean step), then fill output
        for r in range(rows):
            hi = [src.cell(r, c).animate.set_fill(
                      color=PALETTE["fill_2"], opacity=0.9)
                  for c in range(min(cols, 12))]
            scene.play(AnimationGroup(*hi, lag_ratio=0.04), run_time=0.25)

            fill = [out_grid.cell(r, c).animate.set_fill(
                        color=self.output_color, opacity=0.88)
                    for c in range(min(cols, 12))]
            scene.play(AnimationGroup(*fill, lag_ratio=0.04), run_time=0.25)

            restore = [src.cell(r, c).animate.set_fill(
                           color=inputs[0].color, opacity=0.85)
                       for c in range(min(cols, 12))]
            scene.play(AnimationGroup(*restore, lag_ratio=0), run_time=0.1)


class DropoutOp(OpBlock):
    """
    Dropout: randomly zero out cells, then scale survivors.
    Visualises the mask being applied.
    """

    def __init__(self, p: float = 0.5, name: str = "dropout",
                 output_color: str = None, seed: int = 0):
        super().__init__(name=name, output_color=output_color)
        self.p    = p
        self.seed = seed

    def forward(self, x: TensorState) -> TensorState:
        rng  = np.random.default_rng(self.seed)
        mask = rng.random(x.shape) > self.p
        out  = x.data * mask / (1 - self.p)
        return TensorState(data=out, name=self.name or x.name,
                           color=self._output_color([x]))

    def animate(self, scene, inputs, output, in_grids, out_grid):
        src = in_grids[0]
        x   = inputs[0]
        rng  = np.random.default_rng(self.seed)
        mask = rng.random(x.shape) > self.p

        arrow = make_arrow(src, out_grid, label=f"drop p={self.p}")
        scene.play(Create(arrow), FadeIn(out_grid), run_time=0.4)

        rows = x.shape[0] if x.ndim >= 2 else 1
        cols = x.shape[-1]

        # flash dropped cells dark, keep survivors bright
        drop_anims = []
        keep_anims = []
        for r in range(rows):
            for c in range(cols):
                idx = (r, c) if x.ndim >= 2 else (0, c)
                cell = out_grid.cell(*idx)
                m = bool(mask[r, c]) if x.ndim >= 2 else bool(mask[c])
                if m:
                    keep_anims.append(
                        cell.animate.set_fill(color=self._output_color([x]),
                                             opacity=0.9))
                else:
                    drop_anims.append(
                        cell.animate.set_fill(color=PALETTE["cell"], opacity=0.25))

        scene.play(
            AnimationGroup(*(drop_anims + keep_anims), lag_ratio=0.01),
            run_time=0.5,
        )


class EmbeddingOp(OpBlock):
    """
    Embedding lookup: integer indices → dense vectors.
    Shows indices being "looked up" in a weight table.

    weight : TensorState  shape (vocab_size, embed_dim)
    """

    def __init__(self, weight: TensorState, name: str = "embedding",
                 output_color: str = PALETTE["fill_0"]):
        super().__init__(name=name, output_color=output_color)
        self.weight = weight

    def forward(self, indices: TensorState) -> TensorState:
        idx = indices.data.astype(int).flatten()
        out = self.weight.data[idx]
        return TensorState(data=out, name=self.name, color=self.output_color)

    def animate(self, scene, inputs, output, in_grids, out_grid):
        idx_grid = in_grids[0]
        W        = self.weight
        indices  = inputs[0].data.astype(int).flatten()

        cs = _auto_cell_size(W.shape, max_w=2.5, max_h=2.5)
        w_grid = _grid_for_state(W, cell_size=cs)
        w_lbl  = make_label(f"embed table {W.shape}", 14)

        # place weight table between idx and output
        mid_x = (idx_grid.get_right()[0] + out_grid.get_left()[0]) / 2
        w_grid.move_to([mid_x, 0.8, 0])
        w_lbl.next_to(w_grid, UP, buff=SMALL_BUFF)
        scene.play(FadeIn(w_grid), FadeIn(w_lbl), run_time=0.3)

        arrow = make_arrow(w_grid, out_grid, label="lookup")
        scene.play(Create(arrow), FadeIn(out_grid), run_time=0.4)

        # highlight each looked-up row in the table
        for seq_pos, row_idx in enumerate(indices[:min(len(indices), 6)]):
            if row_idx < W.shape[0]:
                row_hi = [w_grid.cell(int(row_idx), c).animate.set_fill(
                              color=PALETTE["fill_2"], opacity=0.95)
                          for c in range(min(W.shape[1], 12))]
                scene.play(AnimationGroup(*row_hi, lag_ratio=0.03), run_time=0.25)

                out_row_hi = [out_grid.cell(seq_pos, c).animate.set_fill(
                                  color=self.output_color, opacity=0.9)
                              for c in range(min(output.shape[-1], 12))]
                scene.play(AnimationGroup(*out_row_hi, lag_ratio=0.03),
                           run_time=0.2)

                restore = [w_grid.cell(int(row_idx), c).animate.set_fill(
                               color=W.color, opacity=0.85)
                           for c in range(min(W.shape[1], 12))]
                scene.play(AnimationGroup(*restore, lag_ratio=0), run_time=0.1)

        scene.play(FadeOut(w_grid), FadeOut(w_lbl), run_time=0.2)


class ResidualOp(OpBlock):
    """
    Residual / skip connection:  output = x + residual
    Shows two grids summing element-wise.
    """

    def __init__(self, residual: TensorState, name: str = "residual",
                 output_color: str = PALETTE["fill_4"]):
        super().__init__(name=name, output_color=output_color)
        self.residual = residual

    def forward(self, x: TensorState) -> TensorState:
        out = x.data + self.residual.data
        return TensorState(data=out, name=self.name, color=self.output_color)

    def animate(self, scene, inputs, output, in_grids, out_grid):
        x_grid = in_grids[0]
        res    = self.residual

        cs = _auto_cell_size(res.shape, max_w=2.0, max_h=2.0)
        r_grid = _grid_for_state(res, cell_size=cs)
        r_lbl  = make_label(f"residual {res.shape}", 14)

        r_grid.next_to(x_grid, UP, buff=0.4)
        r_lbl.next_to(r_grid, UP, buff=SMALL_BUFF)
        scene.play(FadeIn(r_grid), FadeIn(r_lbl), run_time=0.3)

        plus = Text("+", font_size=28, color=PALETTE["title"])
        plus.move_to(
            (x_grid.get_center() + r_grid.get_center()) / 2 + RIGHT * 0.4
        )
        scene.play(Write(plus), run_time=0.2)

        arrow = make_arrow(x_grid, out_grid, label="add")
        scene.play(Create(arrow), FadeIn(out_grid), run_time=0.4)

        # element-wise flash
        rows = output.shape[0] if output.ndim >= 2 else 1
        cols = output.shape[-1]
        anims = [out_grid.cell(r, c).animate.set_fill(
                     color=self.output_color, opacity=0.9)
                 for r in range(rows) for c in range(min(cols, 12))]
        scene.play(AnimationGroup(*anims, lag_ratio=0.01), run_time=0.4)
        scene.play(FadeOut(r_grid), FadeOut(r_lbl), FadeOut(plus), run_time=0.2)


# ══════════════════════════════════════════════════════════════════════════════
# TensorGraph — declare a model as a DAG of ops
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphNode:
    """A node in the TensorGraph DAG."""
    op:      OpBlock
    inputs:  List[str]          # names of input TensorStates
    output:  str                # name of output TensorState


class TensorGraph:
    """
    Declarative computation graph for model-level animation.

    Usage
    ─────
        g = TensorGraph()
        g.input("x",      TensorState(np.random.randn(4, 8),  name="x"))
        g.input("W1",     TensorState(np.random.randn(8, 16), name="W1"))
        g.op(LinearOp(weight=g["W1"]), inputs=["x"],  output="h1")
        g.op(LayerNormOp(),             inputs=["h1"], output="h1_ln")
        g.op(SoftmaxOp(),               inputs=["h1_ln"], output="logits")
        g.render("model.mp4")
    """

    def __init__(self):
        self._states: Dict[str, TensorState] = {}
        self._nodes:  List[GraphNode]         = []

    def input(self, name: str, state: TensorState):
        """Register an input (or constant) TensorState."""
        state.name = name
        self._states[name] = state
        return self

    def op(self, op_block: OpBlock, inputs: List[str], output: str,
           output_color: str = None):
        """Add an op to the graph. `inputs` and `output` are state names."""
        if output_color:
            op_block.output_color = output_color
        self._nodes.append(GraphNode(op=op_block, inputs=inputs, output=output))
        return self

    def __getitem__(self, name: str) -> TensorState:
        return self._states[name]

    def run(self) -> Dict[str, TensorState]:
        """Execute the forward pass and populate all intermediate states."""
        for node in self._nodes:
            in_states = [self._states[n] for n in node.inputs]
            out_state = node.op.forward(*in_states)
            out_state.name = node.output
            self._states[node.output] = out_state
        return self._states

    def render(self, output_path: str = "model.mp4",
               quality: str = "medium_quality", fps: int = 15,
               cell_size: float = None, show_values: bool = False):
        """Run forward pass and render the full animation."""
        from .model_scene import ModelScene
        self.run()
        ModelScene.graph      = self
        ModelScene.cell_size  = cell_size
        ModelScene.show_values = show_values

        from .api import render_scene
        render_scene(ModelScene, output_path=output_path,
                     quality=quality, fps=fps)
