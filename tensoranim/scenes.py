"""
tensoranim/scenes.py
────────────────────
Ready-made Manim Scene subclasses for common tensor transformations.

Each scene can be used standalone or composed into a longer animation.

Available scenes
────────────────
  ReshapeScene       – tensor.reshape(new_shape)
  TransposeScene     – tensor.T  /  tensor.permute(...)
  SliceScene         – tensor[i, :, :]  and similar
  MatMulScene        – A @ B = C  (2-D only)
  ConcatScene        – torch.cat([A, B], dim=...)
  BroadcastScene     – broadcasting a (1, N) tensor over a (M, N) grid
  SoftmaxScene       – row-wise softmax with heat-map transition
  AttentionScene     – scaled dot-product attention Q K V
"""

from __future__ import annotations

import numpy as np
from manim import (
    Scene, VGroup, Text, Rectangle, Arrow, MathTex,
    FadeIn, FadeOut, FadeTransform, Transform, AnimationGroup,
    Create, Write, DrawBorderThenFill, Indicate, Flash,
    Wait, ReplacementTransform,
    LEFT, RIGHT, UP, DOWN, ORIGIN,
    SMALL_BUFF, MED_SMALL_BUFF, MED_LARGE_BUFF, LARGE_BUFF,
    WHITE, BLACK, GREY, config,
    rate_functions,
)

from .core import (
    TensorGrid, make_label, make_arrow, shape_tag,
    PALETTE,
)


# ── base ─────────────────────────────────────────────────────────────────────

class TensorScene(Scene):
    """Base class: dark background + title helper."""

    def setup(self):
        self.camera.background_color = PALETTE["bg"]

    def title(self, text: str) -> Text:
        t = Text(text, font_size=28, color=PALETTE["title"])
        t.to_edge(UP, buff=0.3)
        return t

    def subtitle(self, text: str) -> Text:
        t = Text(text, font_size=18, color=PALETTE["dim_tag"])
        t.to_edge(DOWN, buff=0.35)
        return t


# ── ReshapeScene ─────────────────────────────────────────────────────────────

class ReshapeScene(TensorScene):
    """
    Animate tensor.reshape(new_shape).

    The cells of the source tensor fly one-by-one (in row-major order) into
    their new positions in the target shape.

    Parameters (set as class attributes before rendering)
    ──────────────────────────────────────────────────────
    source_data   : np.ndarray   (1-D or 2-D)
    target_shape  : tuple
    fill_color    : str or callable
    show_values   : bool
    """

    source_data:  np.ndarray = np.arange(12).reshape(3, 4)
    target_shape: tuple = (4, 3)
    fill_color:   str   = PALETTE["fill_0"]
    show_values:  bool  = True

    def construct(self):
        src = np.asarray(self.source_data)
        tgt_shape = self.target_shape
        flat = src.flatten()
        assert flat.size == np.prod(tgt_shape), \
            "reshape: element count must match"

        tgt = flat.reshape(tgt_shape)

        title = self.title(f"reshape  {src.shape}  →  {tgt_shape}")
        self.play(Write(title))

        # source grid
        src_grid = TensorGrid(
            src, fill_color=self.fill_color, show_values=self.show_values
        )
        src_grid.center().shift(LEFT * 2.5)
        src_shape_tag = shape_tag(src.shape)
        src_shape_tag.next_to(src_grid, DOWN, buff=SMALL_BUFF)

        self.play(FadeIn(src_grid), FadeIn(src_shape_tag))
        self.wait(0.4)

        # ghost target grid (empty, just borders) — shown immediately as a guide
        tgt_grid = TensorGrid(
            tgt, fill_color=self.fill_color, fill_opacity=0.0,
            show_values=False, stroke_width=0.8
        )
        tgt_grid.center().shift(RIGHT * 2.5)
        tgt_shape_tag = shape_tag(tgt_shape)
        tgt_shape_tag.next_to(tgt_grid, DOWN, buff=SMALL_BUFF)

        arrow = make_arrow(src_grid, tgt_grid, label="reshape")
        self.play(Create(arrow), FadeIn(tgt_grid), FadeIn(tgt_shape_tag))
        self.wait(0.3)

        # filled target grid (same color + labels) — used only as Transform targets
        tgt_grid_filled = TensorGrid(
            tgt, fill_color=self.fill_color,
            show_values=self.show_values, stroke_width=0.8
        )
        tgt_grid_filled.center().shift(RIGHT * 2.5)

        # animate each cell flying to its new position
        # Each ghost must include the text label (stored separately from the rect)
        anims = []
        for i in range(flat.size):
            src_idx = np.unravel_index(i, src.shape)
            tgt_idx = np.unravel_index(i, tgt_shape)

            src_rect = src_grid.cell(*src_idx)
            tgt_rect = tgt_grid_filled.cell(*tgt_idx)
            src_txt = src_grid._texts.get(src_idx)
            tgt_txt = tgt_grid_filled._texts.get(tgt_idx)

            if src_txt is not None and tgt_txt is not None:
                ghost = VGroup(src_rect.copy(), src_txt.copy())
                anims.append(Transform(ghost, VGroup(tgt_rect, tgt_txt)))
            else:
                anims.append(Transform(src_rect.copy(), tgt_rect))

        self.play(AnimationGroup(*anims, lag_ratio=0.06))
        self.wait(0.8)

        done = Text("✓ reshape complete", font_size=20, color=PALETTE["fill_1"])
        done.next_to(tgt_grid, UP, buff=MED_SMALL_BUFF)
        self.play(FadeIn(done))
        self.wait(1)


# ── TransposeScene ───────────────────────────────────────────────────────────

class TransposeScene(TensorScene):
    """
    Animate matrix.T — show how rows become columns.

    source_data : 2-D np.ndarray
    """

    source_data: np.ndarray = np.arange(6).reshape(2, 3)
    fill_color:  str        = PALETTE["fill_4"]
    show_values: bool       = True

    def construct(self):
        src = np.asarray(self.source_data)
        assert src.ndim == 2, "TransposeScene requires a 2-D tensor."
        tgt = src.T

        title = self.title(f"transpose  {src.shape}  →  {tgt.shape}")
        self.play(Write(title))

        src_grid = TensorGrid(src, fill_color=self.fill_color, show_values=self.show_values)
        src_grid.center().shift(LEFT * 2.8)
        src_lbl = make_label(f"A  {src.shape}", font_size=20)
        src_lbl.next_to(src_grid, DOWN, buff=SMALL_BUFF)

        tgt_grid = TensorGrid(tgt, fill_color=self.fill_color, show_values=self.show_values)
        tgt_grid.center().shift(RIGHT * 2.8)
        tgt_lbl = make_label(f"Aᵀ  {tgt.shape}", font_size=20)
        tgt_lbl.next_to(tgt_grid, DOWN, buff=SMALL_BUFF)

        self.play(FadeIn(src_grid), FadeIn(src_lbl))
        self.wait(0.4)

        arrow = make_arrow(src_grid, tgt_grid, label=".T")
        self.play(Create(arrow))

        rows, cols = src.shape
        anims = []
        for r in range(rows):
            for c in range(cols):
                src_rect = src_grid.cell(r, c)
                tgt_rect = tgt_grid.cell(c, r)
                src_txt = src_grid._texts.get((r, c))
                tgt_txt = tgt_grid._texts.get((c, r))

                if src_txt is not None and tgt_txt is not None:
                    ghost = VGroup(src_rect.copy(), src_txt.copy())
                    anims.append(Transform(ghost, VGroup(tgt_rect, tgt_txt)))
                else:
                    anims.append(Transform(src_rect.copy(), tgt_rect))

        self.play(
            AnimationGroup(*anims, lag_ratio=0.05),
            FadeIn(tgt_lbl),
        )
        self.wait(1)


# ── SliceScene ───────────────────────────────────────────────────────────────

class SliceScene(TensorScene):
    """
    Highlight a slice of a 2-D tensor and extract it.

    source_data : 2-D np.ndarray
    row_slice   : int or None   (None = all rows)
    col_slice   : int or None   (None = all cols)
    """

    source_data: np.ndarray = np.arange(20).reshape(4, 5)
    row_slice:   object     = 1          # e.g. 1  → row 1
    col_slice:   object     = None       # None → all columns
    fill_color:  str        = PALETTE["fill_0"]
    show_values: bool       = True

    def construct(self):
        src = np.asarray(self.source_data)
        assert src.ndim == 2

        rs = self.row_slice
        cs = self.col_slice

        # build slice mask
        row_idx = range(src.shape[0]) if rs is None else [rs]
        col_idx = range(src.shape[1]) if cs is None else [cs]

        slice_expr = (
            f"[{rs if rs is not None else ':'}, "
            f"{cs if cs is not None else ':'}]"
        )
        title = self.title(f"slice  tensor{slice_expr}")
        self.play(Write(title))

        grid = TensorGrid(src, fill_color=self.fill_color, show_values=self.show_values)
        grid.center()
        self.play(FadeIn(grid))
        self.wait(0.4)

        # highlight selected cells
        highlight_anims = []
        for r in row_idx:
            for c in col_idx:
                cell = grid.cell(r, c)
                highlight_anims.append(
                    cell.animate.set_fill(color=PALETTE["fill_2"], opacity=0.95)
                )
        self.play(AnimationGroup(*highlight_anims, lag_ratio=0.04))
        self.wait(0.3)

        # extract: copy highlighted cells to the right
        r_idx = rs if rs is not None else slice(None)
        c_idx = cs if cs is not None else slice(None)
        sliced = src[r_idx, c_idx]
        if np.isscalar(sliced):
            sliced = np.array([[sliced]])
        sliced = np.atleast_2d(sliced)

        out_grid = TensorGrid(
            sliced, fill_color=PALETTE["fill_2"], show_values=self.show_values
        )
        out_grid.next_to(grid, RIGHT, buff=LARGE_BUFF)
        out_lbl = shape_tag(sliced.shape)
        out_lbl.next_to(out_grid, DOWN, buff=SMALL_BUFF)

        arrow = make_arrow(grid, out_grid, label=slice_expr)
        self.play(Create(arrow), FadeIn(out_grid), FadeIn(out_lbl))
        self.wait(1)


# ── MatMulScene ──────────────────────────────────────────────────────────────

class MatMulScene(TensorScene):
    """
    Animate A @ B = C for 2-D matrices.

    Highlights each (row of A, col of B) pair that contributes to a result
    cell in C, then reveals the output grid.
    """

    A: np.ndarray = np.arange(6).reshape(2, 3).astype(float)
    B: np.ndarray = np.arange(6).reshape(3, 2).astype(float)
    show_values: bool = True

    def construct(self):
        A = np.asarray(self.A)
        B = np.asarray(self.B)
        assert A.shape[1] == B.shape[0], "inner dims must match"
        C = A @ B

        title = self.title(
            f"matmul  {A.shape} @ {B.shape} = {C.shape}"
        )
        self.play(Write(title))

        a_grid = TensorGrid(A, fill_color=PALETTE["fill_0"], show_values=self.show_values)
        b_grid = TensorGrid(B, fill_color=PALETTE["fill_1"], show_values=self.show_values)
        c_grid = TensorGrid(C, fill_color=PALETTE["fill_4"],
                            show_values=self.show_values, fill_opacity=0.0,
                            stroke_width=0.8)

        a_lbl = make_label(f"A {A.shape}", 20); b_lbl = make_label(f"B {B.shape}", 20)
        c_lbl = make_label(f"C {C.shape}", 20)
        at_sym = Text("@", color=PALETTE["title"], font_size=36)
        eq_sym = Text("=", color=PALETTE["title"], font_size=36)

        # lay out A  @  B  =  C
        group = VGroup(a_grid, at_sym, b_grid, eq_sym, c_grid)
        group.arrange(RIGHT, buff=0.4)
        group.center()

        for lbl, grid in [(a_lbl, a_grid), (b_lbl, b_grid), (c_lbl, c_grid)]:
            lbl.next_to(grid, DOWN, buff=SMALL_BUFF)

        self.play(
            FadeIn(a_grid), FadeIn(b_grid), FadeIn(c_grid),
            Write(at_sym), Write(eq_sym),
            FadeIn(a_lbl), FadeIn(b_lbl), FadeIn(c_lbl),
        )
        self.wait(0.5)

        # step through each output cell
        m, k = A.shape; k2, n = B.shape
        cs = a_grid.cell_size

        for i in range(m):
            for j in range(n):
                row_anims, col_anims = [], []
                for kk in range(k):
                    row_anims.append(
                        a_grid.cell(i, kk).animate.set_fill(
                            color=PALETTE["fill_2"], opacity=0.9)
                    )
                    col_anims.append(
                        b_grid.cell(kk, j).animate.set_fill(
                            color=PALETTE["fill_3"], opacity=0.9)
                    )
                self.play(AnimationGroup(*(row_anims + col_anims), lag_ratio=0.02),
                          run_time=0.4)

                # reveal C[i,j]
                c_cell = c_grid.cell(i, j)
                self.play(
                    c_cell.animate.set_fill(color=PALETTE["fill_4"], opacity=0.88),
                    run_time=0.25,
                )

                # reset row/col highlight
                reset = []
                for kk in range(k):
                    reset.append(
                        a_grid.cell(i, kk).animate.set_fill(
                            color=PALETTE["fill_0"], opacity=0.85)
                    )
                    reset.append(
                        b_grid.cell(kk, j).animate.set_fill(
                            color=PALETTE["fill_1"], opacity=0.85)
                    )
                self.play(AnimationGroup(*reset, lag_ratio=0), run_time=0.15)

        done = Text("✓", font_size=28, color=PALETTE["fill_1"])
        done.next_to(c_grid, UP, buff=SMALL_BUFF)
        self.play(FadeIn(done))
        self.wait(1)


# ── ConcatScene ──────────────────────────────────────────────────────────────

class ConcatScene(TensorScene):
    """
    Animate torch.cat([A, B], dim=axis).

    tensors : list of np.ndarray (must be 2-D and compatible)
    axis    : 0 (stack rows) or 1 (stack columns)
    """

    tensors: list = [
        np.ones((2, 4)),
        np.ones((3, 4)) * 2,
    ]
    axis: int = 0

    def construct(self):
        tensors = [np.asarray(t) for t in self.tensors]
        axis = self.axis
        result = np.concatenate(tensors, axis=axis)

        colors = [PALETTE["fill_0"], PALETTE["fill_1"], PALETTE["fill_2"],
                  PALETTE["fill_3"]]

        title = self.title(f"cat(tensors, dim={axis})  →  {result.shape}")
        self.play(Write(title))

        grids = []
        for i, t in enumerate(tensors):
            g = TensorGrid(t, fill_color=colors[i % len(colors)])
            grids.append(g)

        grp = VGroup(*grids)
        grp.arrange(RIGHT, buff=0.6)
        grp.center().shift(LEFT * 2)

        labels = []
        for i, (g, t) in enumerate(zip(grids, tensors)):
            lbl = make_label(f"T{i} {t.shape}", 18)
            lbl.next_to(g, DOWN, buff=SMALL_BUFF)
            labels.append(lbl)

        self.play(FadeIn(grp), *[FadeIn(l) for l in labels])
        self.wait(0.4)

        res_grid = TensorGrid(
            result,
            fill_color=lambda idx, v: colors[
                self._which_tensor(tensors, axis, idx[0] if axis == 0 else idx[1])
            ],
        )
        res_grid.center().shift(RIGHT * 3)
        res_lbl = make_label(f"cat  {result.shape}", 20)
        res_lbl.next_to(res_grid, DOWN, buff=SMALL_BUFF)

        arrow = make_arrow(grp, res_grid, label=f"cat dim={axis}")
        self.play(Create(arrow))

        # animate each source grid flying to its final position
        anims = []
        for src_g in grids:
            ghost = src_g.copy()
            anims.append(FadeTransform(ghost, res_grid))
        self.play(AnimationGroup(*anims, lag_ratio=0.3))
        self.play(FadeIn(res_grid), FadeIn(res_lbl))
        self.wait(1)

    @staticmethod
    def _which_tensor(tensors, axis, idx):
        cum = 0
        for i, t in enumerate(tensors):
            cum += t.shape[axis]
            if idx < cum:
                return i
        return len(tensors) - 1


# ── BroadcastScene ───────────────────────────────────────────────────────────

class BroadcastScene(TensorScene):
    """
    Visualise broadcasting a (1, N) vector across M rows to produce (M, N).

    vector_data : 1-D or (1, N) np.ndarray
    n_rows      : M (number of times to broadcast)
    """

    vector_data: np.ndarray = np.array([1, 2, 3, 4], dtype=float)
    n_rows:      int        = 3

    def construct(self):
        vec = np.asarray(self.vector_data).flatten()
        N = vec.size
        M = self.n_rows
        result = np.tile(vec, (M, 1))

        title = self.title(f"broadcast  (1,{N})  →  ({M},{N})")
        self.play(Write(title))

        vec_grid = TensorGrid(
            vec[np.newaxis, :], fill_color=PALETTE["fill_2"], show_values=True
        )
        vec_grid.center().shift(UP * 1.5 + LEFT * 2.5)
        vec_lbl = make_label(f"(1, {N})", 18)
        vec_lbl.next_to(vec_grid, LEFT, buff=SMALL_BUFF)

        self.play(FadeIn(vec_grid), FadeIn(vec_lbl))
        self.wait(0.3)

        out_grid = TensorGrid(result, fill_color=PALETTE["fill_2"],
                              show_values=True, fill_opacity=0.15,
                              stroke_width=0.8)
        out_grid.center().shift(RIGHT * 2)
        out_lbl = make_label(f"({M}, {N})", 18)
        out_lbl.next_to(out_grid, DOWN, buff=SMALL_BUFF)

        arrow = make_arrow(vec_grid, out_grid, label="broadcast")
        self.play(Create(arrow), FadeIn(out_grid), FadeIn(out_lbl))
        self.wait(0.2)

        # animate row copies
        for r in range(M):
            row_anims = []
            for c in range(N):
                cell = out_grid.cell(r, c)
                row_anims.append(
                    cell.animate.set_fill(color=PALETTE["fill_2"], opacity=0.88)
                )
            self.play(AnimationGroup(*row_anims, lag_ratio=0.06), run_time=0.35)
        self.wait(1)


# ── SoftmaxScene ─────────────────────────────────────────────────────────────

class SoftmaxScene(TensorScene):
    """
    Visualise row-wise softmax: raw logits → probabilities as a heat-map.

    source_data : 2-D np.ndarray of logits
    """

    source_data: np.ndarray = np.array([
        [2.0, 1.0, 0.1, 3.0],
        [0.5, 2.5, 1.0, 0.2],
        [1.0, 0.0, 4.0, 0.5],
    ])

    def construct(self):
        src = np.asarray(self.source_data)
        assert src.ndim == 2

        def _softmax(x):
            x = x - x.max(axis=1, keepdims=True)
            e = np.exp(x)
            return e / e.sum(axis=1, keepdims=True)

        probs = _softmax(src)

        title = self.title("softmax (row-wise)")
        self.play(Write(title))

        # normalise to [0,1] for colour interpolation
        def logit_color(idx, val):
            mn, mx = src.min(), src.max()
            t = (val - mn) / (mx - mn + 1e-9)
            # lerp blue → orange
            r = int(59 + t * (251 - 59))
            g = int(130 + t * (146 - 130))
            b = int(246 - t * (246 - 11))
            return f"#{r:02x}{g:02x}{b:02x}"

        def prob_color(idx, val):
            # green heat: 0 → dark, 1 → bright green
            t = float(val)
            r = int(16 + t * 20)
            g = int(100 + t * 155)
            b = int(50 + t * 30)
            return f"#{r:02x}{g:02x}{b:02x}"

        src_grid = TensorGrid(src, fill_color=logit_color, show_values=True)
        src_grid.center().shift(LEFT * 2.5)
        src_lbl = make_label("logits", 20)
        src_lbl.next_to(src_grid, DOWN, buff=SMALL_BUFF)

        self.play(FadeIn(src_grid), FadeIn(src_lbl))
        self.wait(0.5)

        out_grid = TensorGrid(probs, fill_color=prob_color, show_values=True,
                              fill_opacity=0.15, stroke_width=0.8, value_format=".2f")
        out_grid.center().shift(RIGHT * 2.5)
        out_lbl = make_label("softmax", 20)
        out_lbl.next_to(out_grid, DOWN, buff=SMALL_BUFF)

        arrow = make_arrow(src_grid, out_grid, label="softmax")
        self.play(Create(arrow), FadeIn(out_grid), FadeIn(out_lbl))
        self.wait(0.2)

        rows, cols = probs.shape
        for r in range(rows):
            anims = []
            for c in range(cols):
                cell = out_grid.cell(r, c)
                color = prob_color((r, c), probs[r, c])
                anims.append(cell.animate.set_fill(color=color, opacity=0.35))
            self.play(AnimationGroup(*anims, lag_ratio=0.08), run_time=0.5)
        self.wait(1)


# ── AttentionScene ───────────────────────────────────────────────────────────
class AttentionScene(TensorScene):
    """
    Animate scaled dot-product attention: softmax(Q Kᵀ / √d) V

    Q : (L, d)   — queries
    K : (S, d)   — keys  (displayed transposed as Kᵀ: d × S)
    V : (S, d)   — values
    Output: (L, d)

    Animation steps
    ───────────────
    1. Show Q (L×d) and Kᵀ (d×S) with a @ operator — matmul geometry visible
    2. Arrow → scores (L×S) heat-map with shape annotation
    3. Row-by-row softmax transition on the score grid → attn weights
    4. Show attn @ V with arrow → output (L×d)
    """

    Q: np.ndarray = np.random.randn(3, 4)
    K: np.ndarray = np.random.randn(3, 4)
    V: np.ndarray = np.random.randn(3, 4)

    def construct(self):
        Q = np.asarray(self.Q)
        K = np.asarray(self.K)
        V = np.asarray(self.V)
        L, d = Q.shape
        S    = K.shape[0]

        # ── compute all intermediate tensors up front ──────────────────────
        Kt     = K.T                           # (d, S)
        scores = Q @ Kt / np.sqrt(d)           # (L, S)
        attn   = self._softmax(scores)         # (L, S)
        out    = attn @ V                      # (L, d)

        # ── colour helpers ─────────────────────────────────────────────────
        def heat(mn, mx):
            """Blue (low) → Red (high) heat map with clamped RGB."""
            def _c(idx, val):
                t = float(np.clip((val - mn) / (mx - mn + 1e-9), 0.0, 1.0))
                r = int(np.clip(59  + t * (220 - 59),  0, 255))
                g = int(np.clip(130 + t * (30  - 130), 0, 255))
                b = int(np.clip(246 - t * 210,          0, 255))
                return f"#{r:02x}{g:02x}{b:02x}"
            return _c

        def attn_color():
            """Green heat map for attention weights (always in [0,1])."""
            def _c(idx, val):
                t = float(np.clip(val, 0.0, 1.0))
                r = int(np.clip(20  + t * 30,  0, 255))
                g = int(np.clip(80  + t * 170, 0, 255))
                b = int(np.clip(60  + t * 20,  0, 255))
                return f"#{r:02x}{g:02x}{b:02x}"
            return _c

        # ── title ─────────────────────────────────────────────────────────
        title = self.title("Scaled Dot-Product Attention")
        self.play(Write(title))
        self.wait(0.2)

        # ══════════════════════════════════════════════════════════════════
        # STEP 1 — show Q and Kᵀ with matmul geometry
        # ══════════════════════════════════════════════════════════════════
        sub1 = self.subtitle("Step 1: compute similarity scores  Q @ Kᵀ / √d")
        self.play(FadeIn(sub1))

        q_grid  = TensorGrid(Q,  fill_color=PALETTE["fill_0"], show_values=False)
        kt_grid = TensorGrid(Kt, fill_color=PALETTE["fill_1"], show_values=False)

        at_sym = Text("@", color=PALETTE["title"], font_size=32)

        left_group = VGroup(q_grid, at_sym, kt_grid)
        left_group.arrange(RIGHT, buff=0.35)
        left_group.move_to(LEFT * 3.2)

        q_shape  = shape_tag((L, d));    q_shape.next_to(q_grid,  DOWN, buff=SMALL_BUFF)
        kt_shape = shape_tag((d, S));    kt_shape.next_to(kt_grid, DOWN, buff=SMALL_BUFF)
        q_lbl    = make_label("Q",  16); q_lbl.next_to(q_grid,   UP,   buff=SMALL_BUFF)
        kt_lbl   = make_label("Kᵀ", 16); kt_lbl.next_to(kt_grid,  UP,   buff=SMALL_BUFF)

        self.play(
            FadeIn(q_grid), FadeIn(kt_grid), Write(at_sym),
            FadeIn(q_lbl),  FadeIn(kt_lbl),
            FadeIn(q_shape), FadeIn(kt_shape),
        )
        self.wait(0.5)

        # ghost scores grid on the right
        sc_grid_ghost = TensorGrid(
            scores, fill_color=PALETTE["cell"],
            fill_opacity=0.25, stroke_width=0.8, show_values=False,
        )
        sc_grid_ghost.move_to(RIGHT * 2.8)
        sc_shape = shape_tag((L, S)); sc_shape.next_to(sc_grid_ghost, DOWN, buff=SMALL_BUFF)
        sc_lbl   = make_label("scores  (Q @ Kᵀ / √d)", 14)
        sc_lbl.next_to(sc_grid_ghost, UP, buff=SMALL_BUFF)

        arr1 = make_arrow(kt_grid, sc_grid_ghost, label="/ √d")
        self.play(Create(arr1), FadeIn(sc_grid_ghost), FadeIn(sc_shape), FadeIn(sc_lbl))
        self.wait(0.3)

        # fill heat-map column by column
        heat_fn = heat(scores.min(), scores.max())
        col_anims = []
        for c in range(S):
            col = []
            for r in range(L):
                col.append(
                    sc_grid_ghost.cell(r, c).animate.set_fill(
                        color=heat_fn((r, c), scores[r, c]), opacity=0.9
                    )
                )
            col_anims.append(AnimationGroup(*col, lag_ratio=0.05))
        self.play(AnimationGroup(*col_anims, lag_ratio=0.25), run_time=1.2)
        self.wait(0.5)

        # ══════════════════════════════════════════════════════════════════
        # STEP 2 — row-wise softmax on scores → attention weights
        # ══════════════════════════════════════════════════════════════════
        self.play(FadeOut(sub1))
        sub2 = self.subtitle("Step 2: softmax row-wise → attention weights (each row sums to 1)")
        self.play(FadeIn(sub2))

        self.play(
            FadeOut(left_group),
            FadeOut(q_lbl), FadeOut(kt_lbl),
            FadeOut(q_shape), FadeOut(kt_shape),
            FadeOut(arr1),
            sc_grid_ghost.animate.move_to(LEFT * 3.2),
        )
        sc_shape.next_to(sc_grid_ghost, DOWN, buff=SMALL_BUFF)
        sc_lbl.next_to(sc_grid_ghost,   UP,   buff=SMALL_BUFF)
        self.wait(0.3)

        at_grid = TensorGrid(
            attn, fill_color=PALETTE["cell"],
            fill_opacity=0.25, stroke_width=0.8, show_values=False,
        )
        at_grid.move_to(RIGHT * 2.0)
        at_shape = shape_tag((L, S)); at_shape.next_to(at_grid, DOWN, buff=SMALL_BUFF)
        at_lbl   = make_label("attn weights", 14); at_lbl.next_to(at_grid, UP, buff=SMALL_BUFF)

        arr2 = make_arrow(sc_grid_ghost, at_grid, label="softmax")
        self.play(Create(arr2), FadeIn(at_grid), FadeIn(at_shape), FadeIn(at_lbl))
        self.wait(0.2)

        attn_fn = attn_color()
        for r in range(L):
            hi = [
                sc_grid_ghost.cell(r, c).animate.set_fill(
                    color=PALETTE["fill_2"], opacity=0.95)
                for c in range(S)
            ]
            self.play(AnimationGroup(*hi, lag_ratio=0.05), run_time=0.3)

            fill = [
                at_grid.cell(r, c).animate.set_fill(
                    color=attn_fn((r, c), attn[r, c]), opacity=0.9)
                for c in range(S)
            ]
            self.play(AnimationGroup(*fill, lag_ratio=0.08), run_time=0.4)

            restore = [
                sc_grid_ghost.cell(r, c).animate.set_fill(
                    color=heat_fn((r, c), scores[r, c]), opacity=0.9)
                for c in range(S)
            ]
            self.play(AnimationGroup(*restore, lag_ratio=0), run_time=0.15)

        self.wait(0.5)

        # ══════════════════════════════════════════════════════════════════
        # STEP 3 — attn @ V = output
        # ══════════════════════════════════════════════════════════════════
        self.play(FadeOut(sub2))
        sub3 = self.subtitle("Step 3: attn @ V — weighted sum of values → output")
        self.play(FadeIn(sub3))

        self.play(
            FadeOut(sc_grid_ghost), FadeOut(sc_shape), FadeOut(sc_lbl),
            FadeOut(arr2),
            at_grid.animate.move_to(LEFT * 3.8),
        )
        at_shape.next_to(at_grid, DOWN, buff=SMALL_BUFF)
        at_lbl.next_to(at_grid,   UP,   buff=SMALL_BUFF)

        at_sym2 = Text("@", color=PALETTE["title"], font_size=32)
        v_grid  = TensorGrid(V, fill_color=PALETTE["fill_4"], show_values=False)
        v_shape = shape_tag((S, d))
        v_lbl   = make_label("V", 16)

        right_group = VGroup(at_sym2, v_grid)
        right_group.arrange(RIGHT, buff=0.35)
        right_group.next_to(at_grid, RIGHT, buff=0.35)

        v_shape.next_to(v_grid, DOWN, buff=SMALL_BUFF)
        v_lbl.next_to(v_grid,   UP,   buff=SMALL_BUFF)

        self.play(
            Write(at_sym2), FadeIn(v_grid),
            FadeIn(v_shape), FadeIn(v_lbl),
        )
        self.wait(0.4)

        out_grid = TensorGrid(
            out, fill_color=PALETTE["cell"],
            fill_opacity=0.25, stroke_width=0.8, show_values=False,
        )
        out_grid.move_to(RIGHT * 4.0)
        out_shape = shape_tag((L, d)); out_shape.next_to(out_grid, DOWN, buff=SMALL_BUFF)
        out_lbl   = make_label("output", 16); out_lbl.next_to(out_grid, UP, buff=SMALL_BUFF)

        arr3 = make_arrow(v_grid, out_grid)
        self.play(Create(arr3), FadeIn(out_grid), FadeIn(out_shape), FadeIn(out_lbl))
        self.wait(0.2)

        out_color = PALETTE["fill_3"]
        for r in range(L):
            hi_attn = [
                at_grid.cell(r, c).animate.set_fill(color=PALETTE["fill_2"], opacity=0.95)
                for c in range(S)
            ]
            hi_v = [
                v_grid.cell(s, c).animate.set_fill(color=PALETTE["fill_2"], opacity=0.95)
                for s in range(S) for c in range(d)
            ]
            self.play(AnimationGroup(*(hi_attn + hi_v), lag_ratio=0.02), run_time=0.35)

            fill_out = [
                out_grid.cell(r, c).animate.set_fill(color=out_color, opacity=0.88)
                for c in range(d)
            ]
            self.play(AnimationGroup(*fill_out, lag_ratio=0.06), run_time=0.35)

            restore_attn = [
                at_grid.cell(r, c).animate.set_fill(
                    color=attn_fn((r, c), attn[r, c]), opacity=0.9)
                for c in range(S)
            ]
            restore_v = [
                v_grid.cell(s, c).animate.set_fill(
                    color=PALETTE["fill_4"], opacity=0.85)
                for s in range(S) for c in range(d)
            ]
            self.play(AnimationGroup(*(restore_attn + restore_v), lag_ratio=0),
                      run_time=0.15)

        self.wait(0.4)
        self.play(FadeOut(sub3))

        done = Text("✓  attention complete", font_size=22, color=PALETTE["fill_1"])
        done.next_to(out_grid, UP, buff=MED_SMALL_BUFF)
        self.play(FadeIn(done))
        self.wait(1.5)

    @staticmethod
    def _softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)