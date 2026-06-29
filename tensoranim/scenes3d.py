"""
tensoranim/scenes3d.py
──────────────────────
3-D Manim Scene subclasses for tensor operations that are inherently 3-D.

All scenes inherit from TensorScene3D (a ThreeDScene) so the camera gives a
perspective view and voxels are rendered as true 3-D boxes (Prisms).

Note: in Manim 0.20, 2-D overlays (labels, subtitles) that should not rotate
with the 3-D scene must be registered via self.add_fixed_in_frame_mobjects().

Available scenes
────────────────
  Tensor3DScene    – display a raw (D, R, C) tensor with an orbital camera
  DepthSliceScene  – extract one depth-slice at a time from a 3-D tensor
  Permute3DScene   – animate tensor.permute(axes) for 3-D tensors
  BMMScene         – batched matmul (B,M,K) @ (B,K,N) = (B,M,N)
  Conv2DScene      – 2-D convolution showing the sliding receptive field
  EinsumScene      – einsum with explicit subscript string, step-by-step
"""

from __future__ import annotations

import itertools
import numpy as np

from manim import (
    VGroup, Text, Arrow, FadeIn, FadeOut, Transform, AnimationGroup,
    Create, Write, ReplacementTransform,
    LEFT, RIGHT, UP, DOWN, ORIGIN, IN, OUT,
    SMALL_BUFF, MED_SMALL_BUFF, MED_LARGE_BUFF,
    WHITE, PI,
    UL, DR,
)

from core import (
    TensorGrid,
    ThreeDTensorGrid,
    TensorScene3D,
    make_label, make_arrow, shape_tag,
    PALETTE,
)


# ── tiny helpers ──────────────────────────────────────────────────────────────

def _resolve_fill(fill_color, idx, val):
    return fill_color(idx, val) if callable(fill_color) else fill_color


def _fix(*mobs, scene):
    """Register mobjects as fixed-in-frame (2-D overlay) in a ThreeDScene."""
    scene.add_fixed_in_frame_mobjects(*mobs)


def _unfix(*mobs, scene):
    scene.remove_fixed_in_frame_mobjects(*mobs)


# ── Tensor3DScene ─────────────────────────────────────────────────────────────

class Tensor3DScene(TensorScene3D):
    """
    Render a (D, R, C) tensor as a voxel grid and slowly orbit the camera so
    all three axes are visible.

    source_data : np.ndarray  shape (D, R, C)
    fill_color  : str or callable(idx, val) -> str
    rotate      : bool — whether to auto-rotate the camera
    """

    source_data: np.ndarray = np.arange(24).reshape(2, 3, 4)
    fill_color:  object     = PALETTE["fill_0"]
    rotate:      bool       = True

    def construct(self):
        data = np.asarray(self.source_data)
        assert data.ndim == 3, "Tensor3DScene requires a 3-D tensor."
        D, R, C = data.shape

        title = self.title(f"3-D Tensor  ({D}, {R}, {C})")
        _fix(title, scene=self)
        self.play(Write(title))

        grid = ThreeDTensorGrid(data, fill_color=self.fill_color)
        grid.center()
        self.play(FadeIn(grid))
        self.wait(0.3)

        d_lbl = make_label(f"D={D}  depth", 16, PALETTE["fill_0"])
        r_lbl = make_label(f"R={R}  rows",  16, PALETTE["fill_1"])
        c_lbl = make_label(f"C={C}  cols",  16, PALETTE["fill_2"])

        d_lbl.to_corner(DR, buff=0.5).shift(UP * 0.6)
        r_lbl.next_to(d_lbl, UP, buff=0.1)
        c_lbl.next_to(r_lbl, UP, buff=0.1)
        _fix(d_lbl, r_lbl, c_lbl, scene=self)
        self.play(FadeIn(d_lbl), FadeIn(r_lbl), FadeIn(c_lbl))

        if self.rotate:
            self.begin_ambient_camera_rotation(rate=0.25)
            self.wait(5)
            self.stop_ambient_camera_rotation()
        else:
            self.wait(1.5)


# ── DepthSliceScene ───────────────────────────────────────────────────────────

class DepthSliceScene(TensorScene3D):
    """
    Extract each depth-slice of a 3-D tensor one by one.

    For each slice d:
      1. Highlight that slab in the voxel grid (amber)
      2. Show it as a flat 2-D grid in a fixed overlay panel
      3. Restore and continue

    source_data : np.ndarray  shape (D, R, C)
    """

    source_data: np.ndarray = np.arange(24).reshape(2, 3, 4)
    fill_color:  object     = PALETTE["fill_0"]

    def construct(self):
        data = np.asarray(self.source_data)
        assert data.ndim == 3
        D, R, C = data.shape

        title = self.title(f"depth slice  ({D},{R},{C})[d, :, :]")
        _fix(title, scene=self)
        self.play(Write(title))

        grid = ThreeDTensorGrid(data, fill_color=self.fill_color)
        grid.center().shift(LEFT * 1.5)
        self.play(FadeIn(grid))
        self.wait(0.4)

        colors = [PALETTE["fill_0"], PALETTE["fill_1"],
                PALETTE["fill_2"], PALETTE["fill_3"], PALETTE["fill_4"]]

        for d in range(D):
            sub = self.subtitle(f"tensor[{d}, :, :]  →  ({R}, {C})")
            _fix(sub, scene=self)
            self.play(FadeIn(sub))

            # Dim all other slabs to near-invisible before highlighting the active
            # one. Manim has no z-buffering so back slabs always render through
            # front ones — opacity is the only reliable way to suppress this.
            other_slabs = [grid.get_slice_d(dd) for dd in range(D) if dd != d]
            dim_anims = [
                c.animate.set_fill(opacity=0.08)
                for slab_group in other_slabs
                for c in slab_group
            ]
            active_slab = grid.get_slice_d(d)
            hi_anims = [
                c.animate.set_fill(color=PALETTE["fill_2"], opacity=0.95)
                for c in active_slab
            ]
            self.play(*dim_anims, *hi_anims, run_time=0.4)
            self.wait(0.2)

            flat = TensorGrid(data[d], fill_color=colors[d % len(colors)],
                            show_values=True, cell_size=0.5)
            flat.to_edge(RIGHT, buff=0.5)
            flat_lbl = make_label(f"[{d}, :, :]  ({R}×{C})", 16)
            flat_lbl.next_to(flat, UP, buff=SMALL_BUFF)
            _fix(flat, flat_lbl, scene=self)
            self.play(FadeIn(flat), FadeIn(flat_lbl), run_time=0.4)
            self.wait(0.7)

            # Restore all slabs simultaneously with the panel fade-out
            restore_anims = [
                grid.cell(dd, r, c).animate.set_fill(
                    color=_resolve_fill(self.fill_color, (dd, r, c), float(data[dd, r, c])),
                    opacity=0.75)
                for dd in range(D)
                for r in range(R)
                for c in range(C)
            ]
            self.play(FadeOut(flat), FadeOut(flat_lbl), FadeOut(sub),
                    *restore_anims, run_time=0.4)
            _unfix(flat, flat_lbl, sub, scene=self)

        done = make_label("✓  slicing complete", 20, PALETTE["fill_1"])
        done.to_edge(DOWN, buff=0.4)
        _fix(done, scene=self)
        self.play(FadeIn(done))
        self.wait(1.2)


# ── Permute3DScene ────────────────────────────────────────────────────────────

class Permute3DScene(TensorScene3D):
    """
    Animate tensor.permute(order) / np.transpose(data, order) for 3-D tensors.

    source_data : np.ndarray  shape (D, R, C)
    order       : tuple of 3 ints, e.g. (2, 0, 1)
    """

    source_data: np.ndarray = np.arange(24).reshape(2, 3, 4)
    order:       tuple      = (2, 0, 1)
    fill_color:  object     = PALETTE["fill_4"]

    def construct(self):
        data  = np.asarray(self.source_data)
        order = self.order
        perm  = np.transpose(data, order)
        D, R, C   = data.shape
        pD, pR, pC = perm.shape

        title = self.title(f"permute  {data.shape}  →  {perm.shape}  axes={order}")
        _fix(title, scene=self)
        self.play(Write(title))

        src = ThreeDTensorGrid(data, fill_color=self.fill_color)
        src.center().shift(LEFT * 2.0)
        src_lbl = make_label(f"source  {data.shape}", 17)
        src_lbl.next_to(src, UP, buff=SMALL_BUFF)
        _fix(src_lbl, scene=self)
        self.play(FadeIn(src), FadeIn(src_lbl))
        self.wait(0.5)

        # explain each axis mapping
        axis_names = ["D", "R", "C"]
        lines = [
            f"axis {old} ({axis_names[old]}, size {data.shape[old]})  →  axis {new}"
            for new, old in enumerate(order)
        ]
        sub = self.subtitle("  |  ".join(lines))
        _fix(sub, scene=self)
        self.play(FadeIn(sub))
        self.wait(1.0)

        # show the permuted grid to the right
        dst = ThreeDTensorGrid(perm, fill_color=self.fill_color, cell_size=0.44)
        dst.center().shift(RIGHT * 2.5)
        dst_lbl = make_label(f"permuted  {perm.shape}", 17)
        dst_lbl.next_to(dst, UP, buff=SMALL_BUFF)
        _fix(dst_lbl, scene=self)
        self.play(FadeIn(dst), FadeIn(dst_lbl))
        self.wait(0.5)

        # highlight one axis at a time
        hi_colors = [PALETTE["fill_0"], PALETTE["fill_1"], PALETTE["fill_2"]]
        slice_getters = [src.get_slice_d, src.get_slice_r, src.get_slice_c]
        new_slices = [dst.get_slice_d, dst.get_slice_r, dst.get_slice_c]
        for new_pos, old_axis in enumerate(order):
            ax_sub = self.subtitle(
                f"old axis {old_axis} ({axis_names[old_axis]})  →  "
                f"new axis {new_pos}  (size {perm.shape[new_pos]})"
            )
            _fix(ax_sub, scene=self)
            self.play(ReplacementTransform(sub, ax_sub))
            sub = ax_sub

            slab = slice_getters[old_axis](0)
            perms = new_slices[new_pos](0)
            self.play(
                *[c.animate.set_fill(color=hi_colors[new_pos], opacity=0.95) for c in slab],
                *[c.animate.set_fill(color=hi_colors[new_pos], opacity=0.95) for c in perms],
                run_time=0.4,
            )
            
            self.wait(0.5)
            self.play(
                *[c.animate.set_fill(color=self.fill_color, opacity=0.75) for c in slab],
                *[c.animate.set_fill(color=self.fill_color, opacity=0.75) for c in perms],
                run_time=0.2,
            )

        self.play(FadeOut(sub))
        _unfix(sub, scene=self)

        done = make_label("✓  permute complete", 20, PALETTE["fill_1"])
        done.to_edge(DOWN, buff=0.4)
        _fix(done, scene=self)
        self.play(FadeIn(done))
        self.wait(1.5)


# ── BMMScene ─────────────────────────────────────────────────────────────────

class BMMScene(TensorScene3D):
    """
    Animate batched matrix multiply:  (B, M, K) @ (B, K, N) = (B, M, N)

    For each batch b: highlight A[b] and B[b], show A[b] @ B[b] = C[b]
    in a flat panel, then fill the corresponding slice of the output.

    A : np.ndarray  shape (B, M, K)
    B : np.ndarray  shape (B, K, N)
    """

    A: np.ndarray = np.arange(12).reshape(2, 2, 3).astype(float)
    B: np.ndarray = np.arange(12).reshape(2, 3, 2).astype(float)

    def construct(self):
        A = np.asarray(self.A);  B = np.asarray(self.B)
        assert A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1]
        batch, M, K = A.shape;  _, K2, N = B.shape
        C = np.einsum("bmk,bkn->bmn", A, B)

        title = self.title(
            f"bmm  ({batch},{M},{K}) @ ({batch},{K},{N}) = ({batch},{M},{N})"
        )
        _fix(title, scene=self)
        self.play(Write(title))

        a_grid = ThreeDTensorGrid(A, fill_color=PALETTE["fill_0"], cell_size=0.44)
        b_grid = ThreeDTensorGrid(B, fill_color=PALETTE["fill_1"], cell_size=0.44)
        c_grid = ThreeDTensorGrid(C, fill_color=PALETTE["cell"],
                                fill_opacity=0.18, stroke_width=0.8, cell_size=0.44)

        a_grid.center().shift(LEFT * 3.8)
        b_grid.center().shift(LEFT * 0.8)
        c_grid.center().shift(RIGHT * 3.0)

        for g, txt in [(a_grid, f"A {A.shape}"),
                    (b_grid, f"B {B.shape}"),
                    (c_grid, f"C {C.shape}")]:
            lbl = make_label(txt, 15)
            lbl.next_to(g, UP, buff=SMALL_BUFF)
            _fix(lbl, scene=self)
            self.add(lbl)

        self.play(FadeIn(a_grid), FadeIn(b_grid), FadeIn(c_grid))
        self.wait(0.4)

        for b in range(batch):
            sub = self.subtitle(f"batch {b}:  A[{b}] @ B[{b}]  →  C[{b}]")
            _fix(sub, scene=self)
            self.play(FadeIn(sub))

            # Dim all non-active batch slices across A, B, and C simultaneously
            # before highlighting — same z-buffer workaround as DepthSliceScene.
            other_batches = [bb for bb in range(batch) if bb != b]
            dim_anims = [
                grid.cell(bb, r, k).animate.set_fill(opacity=0.08)
                for grid, rows, cols in [
                    (a_grid, M, K), (b_grid, K, N), (c_grid, M, N)
                ]
                for bb in other_batches
                for r in range(rows)
                for k in range(cols)
            ]
            hi_anims = (
                [a_grid.cell(b, r, k).animate.set_fill(
                    color=PALETTE["fill_2"], opacity=0.95)
                for r in range(M) for k in range(K)] +
                [b_grid.cell(b, k, n).animate.set_fill(
                    color=PALETTE["fill_3"], opacity=0.95)
                for k in range(K) for n in range(N)]
            )
            self.play(*dim_anims, *hi_anims, run_time=0.4)
            self.wait(0.2)

            # flat panel: A[b] @ B[b] = C[b]
            fa = TensorGrid(A[b], fill_color=PALETTE["fill_2"],
                            show_values=True, cell_size=0.4)
            fb = TensorGrid(B[b], fill_color=PALETTE["fill_3"],
                            show_values=True, cell_size=0.4)
            fc = TensorGrid(C[b], fill_color=PALETTE["fill_4"],
                            show_values=True, cell_size=0.4)
            sym_at = Text("@", font_size=22, color=PALETTE["title"])
            sym_eq = Text("=", font_size=22, color=PALETTE["title"])
            panel = VGroup(fa, sym_at, fb, sym_eq, fc)
            panel.arrange(RIGHT, buff=0.2)
            panel.to_edge(DOWN, buff=1.1)
            _fix(panel, scene=self)
            self.play(FadeIn(panel))
            self.wait(0.5)

            # fill C[b] slice
            self.play(
                *[c_grid.cell(b, r, n).animate.set_fill(
                    color=PALETTE["fill_4"], opacity=0.88)
                for r in range(M) for n in range(N)],
                run_time=0.4,
            )
            self.wait(0.3)

            # restore all batch slices simultaneously with panel fade-out
            restore_anims = [
                grid.cell(bb, r, k).animate.set_fill(color=color, opacity=opacity)
                for grid, rows, cols, color, opacity in [
                    (a_grid, M, K, PALETTE["fill_0"], 0.75),
                    (b_grid, K, N, PALETTE["fill_1"], 0.75),
                    (c_grid, M, N, PALETTE["fill_4"], 0.88),
                ]
                for bb in range(batch)
                for r in range(rows)
                for k in range(cols)
            ]
            self.play(FadeOut(panel), FadeOut(sub), *restore_anims, run_time=0.3)
            _unfix(panel, sub, scene=self)

        done = make_label("✓  bmm complete", 20, PALETTE["fill_1"])
        done.to_edge(DOWN, buff=0.4)
        _fix(done, scene=self)
        self.play(FadeIn(done))
        self.wait(1.5)


# ── Conv2DScene ───────────────────────────────────────────────────────────────

class Conv2DScene(TensorScene3D):
    """
    Animate a 2-D convolution:
      input  (C_in, H, W) * kernel (C_in, kH, kW)  →  output (H_out, W_out)

    The input is shown as a 3-D voxel block (channel depth).
    The kernel slides across the input; the receptive field highlights
    and the corresponding output cell fills in.

    input_data : np.ndarray  shape (C_in, H, W)
    kernel     : np.ndarray  shape (C_in, kH, kW)   — one output channel
    stride     : int
    padding    : int
    """

    input_data: np.ndarray = np.arange(18).reshape(1, 3, 6).astype(float)
    kernel:     np.ndarray = np.ones((1, 2, 2))
    stride:     int        = 1
    padding:    int        = 0

    def construct(self):
        X  = np.asarray(self.input_data).astype(float)
        W  = np.asarray(self.kernel).astype(float)
        s, p = self.stride, self.padding
        C_in, H, Ww = X.shape
        _, kH, kW   = W.shape
        H_out = (H + 2*p - kH) // s + 1
        W_out = (Ww + 2*p - kW) // s + 1

        X_pad = np.pad(X, [(0,0),(p,p),(p,p)]) if p > 0 else X
        Y = np.zeros((H_out, W_out))
        for oh in range(H_out):
            for ow in range(W_out):
                Y[oh, ow] = float(np.sum(X_pad[:, oh*s:oh*s+kH, ow*s:ow*s+kW] * W))

        title = self.title(
            f"conv2d  ({C_in},{H},{Ww}) * kernel ({C_in},{kH},{kW})"
            f"  →  ({H_out},{W_out})"
        )
        _fix(title, scene=self)
        self.play(Write(title))

        in_grid = ThreeDTensorGrid(X, fill_color=PALETTE["fill_0"], cell_size=0.48)
        in_grid.center().shift(LEFT * 2.5)
        in_lbl = make_label(f"input ({C_in}×{H}×{Ww})", 15)
        in_lbl.to_corner(UL, buff=0.5).shift(DOWN * 0.7)
        _fix(in_lbl, scene=self)

        out_grid = TensorGrid(Y, fill_color=PALETTE["cell"],
                              fill_opacity=0.2, stroke_width=0.8,
                              show_values=False, cell_size=0.48)
        out_grid.to_edge(RIGHT, buff=0.5)
        out_lbl = make_label(f"output ({H_out}×{W_out})", 15)
        out_lbl.next_to(out_grid, UP, buff=SMALL_BUFF)
        _fix(out_grid, out_lbl, scene=self)

        k_grid = TensorGrid(W[0], fill_color=PALETTE["fill_2"],
                            show_values=True, cell_size=0.44)
        k_grid.to_edge(DOWN, buff=1.0)
        k_lbl = make_label(f"kernel[0] ({kH}×{kW})", 14)
        k_lbl.next_to(k_grid, UP, buff=SMALL_BUFF)
        _fix(k_grid, k_lbl, scene=self)

        self.play(FadeIn(in_grid), FadeIn(in_lbl),
                  FadeIn(out_grid), FadeIn(out_lbl),
                  FadeIn(k_grid), FadeIn(k_lbl))
        self.wait(0.4)

        for oh in range(H_out):
            for ow in range(W_out):
                sub = self.subtitle(
                    f"RF  [:, {oh*s}:{oh*s+kH}, {ow*s}:{ow*s+kW}]  →  Y[{oh},{ow}]"
                )
                _fix(sub, scene=self)
                self.play(FadeIn(sub), run_time=0.15)

                rf = []
                for ci in range(C_in):
                    for kh in range(kH):
                        for kw_i in range(kW):
                            ih, iw = oh*s+kh, ow*s+kw_i
                            if 0 <= ih < H and 0 <= iw < Ww:
                                rf.append(in_grid.cell(ci, ih, iw))

                self.play(
                    *[c.animate.set_fill(color=PALETTE["fill_2"], opacity=0.95)
                      for c in rf],
                    run_time=0.22,
                )
                self.play(
                    out_grid.cell(oh, ow).animate.set_fill(
                        color=PALETTE["fill_1"], opacity=0.9),
                    run_time=0.18,
                )
                self.play(
                    *[c.animate.set_fill(color=PALETTE["fill_0"], opacity=0.75)
                      for c in rf],
                    FadeOut(sub),
                    run_time=0.12,
                )
                _unfix(sub, scene=self)

        done = make_label("✓  conv2d complete", 20, PALETTE["fill_1"])
        done.to_edge(DOWN, buff=0.4)
        _fix(done, scene=self)
        self.play(FadeIn(done))
        self.wait(1.5)


# ── EinsumScene ───────────────────────────────────────────────────────────────

class EinsumScene(TensorScene3D):
    """
    Animate np.einsum(subscripts, A, B) — two-input, one-output.

    Steps:
      1. Show A and B (as 3-D voxel grids if rank-3, else flat 2-D grids)
      2. Annotate axis labels from the subscript string
      3. Slide through contracted indices, highlight matching cells
      4. Fill in each output position

    subscripts : str   e.g. "ij,jk->ik"  or  "bik,bkj->bij"
    A, B       : np.ndarray
    """

    subscripts: str        = "ij,jk->ik"
    A:          np.ndarray = np.arange(6).reshape(2, 3).astype(float)
    B:          np.ndarray = np.arange(6).reshape(3, 2).astype(float)

    def construct(self):
        subscripts = self.subscripts
        A = np.asarray(self.A);  B = np.asarray(self.B)
        C = np.einsum(subscripts, A, B)

        lhs, rhs = subscripts.split("->")
        a_sub, b_sub = lhs.split(",")
        c_sub = rhs.strip()
        contracted = sorted(set(a_sub) & set(b_sub) - set(c_sub))

        title = self.title(f"einsum  '{subscripts}'")
        _fix(title, scene=self)
        self.play(Write(title))

        def make_grid(arr, color):
            """Choose 2-D or 3-D grid based on rank."""
            if arr.ndim <= 2:
                a2 = arr if arr.ndim == 2 else arr[np.newaxis, :]
                g = TensorGrid(a2, fill_color=color, show_values=True, cell_size=0.46)
                return g, True   # is_2d=True
            else:
                g = ThreeDTensorGrid(arr, fill_color=color, cell_size=0.40)
                return g, False

        a_grid, a_2d = make_grid(A, PALETTE["fill_0"])
        b_grid, b_2d = make_grid(B, PALETTE["fill_1"])
        c_arr = C if C.ndim >= 2 else C[np.newaxis, :]
        c_grid = TensorGrid(c_arr, fill_color=PALETTE["cell"],
                            fill_opacity=0.2, stroke_width=0.8,
                            show_values=True, cell_size=0.46)

        a_grid.center().shift(LEFT * 4.2)
        b_grid.center().shift(LEFT * 1.2)
        c_grid.center().shift(RIGHT * 2.5)

        for g, is2d in [(a_grid, a_2d), (b_grid, b_2d), (c_grid, True)]:
            if is2d:
                _fix(g, scene=self)

        for g, sub_str, arr in [(a_grid, a_sub, A), (b_grid, b_sub, B),
                                 (c_grid, c_sub, C)]:
            lbl = make_label(f"'{sub_str}'  {arr.shape}", 14)
            lbl.next_to(g, UP, buff=SMALL_BUFF)
            _fix(lbl, scene=self)
            self.add(lbl)

        self.play(FadeIn(a_grid), FadeIn(b_grid), FadeIn(c_grid))
        self.wait(0.4)

        sub1 = self.subtitle(
            f"A[{a_sub}], B[{b_sub}] → C[{c_sub}]   "
            f"contracted axes: {contracted if contracted else 'none (outer product)'}"
        )
        _fix(sub1, scene=self)
        self.play(FadeIn(sub1))
        self.wait(1.2)
        self.play(FadeOut(sub1))
        _unfix(sub1, scene=self)

        # step through output indices (cap at 8 for legibility)
        out_indices = list(np.ndindex(*C.shape)) if C.shape else [()]
        shown = out_indices[:8]

        for out_idx in shown:
            idx_map = dict(zip(c_sub, out_idx))
            c_ranges = []
            for ax in contracted:
                sz = A.shape[a_sub.index(ax)] if ax in a_sub else B.shape[b_sub.index(ax)]
                c_ranges.append(range(sz))

            sub2 = self.subtitle(
                f"C{list(out_idx)}  ←  sum over {contracted}"
            )
            _fix(sub2, scene=self)
            self.play(FadeIn(sub2), run_time=0.15)

            for c_vals in itertools.product(*c_ranges):
                c_map  = dict(zip(contracted, c_vals))
                full   = {**idx_map, **c_map}
                a_idx  = tuple(full[ax] for ax in a_sub)
                b_idx  = tuple(full[ax] for ax in b_sub)
                a_2d_i = a_idx if A.ndim >= 2 else (0, a_idx[0])
                b_2d_i = b_idx if B.ndim >= 2 else (0, b_idx[0])

                try:
                    ac = a_grid.cell(*a_2d_i)
                    bc = b_grid.cell(*b_2d_i)
                    self.play(
                        ac.animate.set_fill(color=PALETTE["fill_2"], opacity=0.95),
                        bc.animate.set_fill(color=PALETTE["fill_3"], opacity=0.95),
                        run_time=0.12,
                    )
                    self.play(
                        ac.animate.set_fill(color=PALETTE["fill_0"], opacity=0.85),
                        bc.animate.set_fill(color=PALETTE["fill_1"], opacity=0.85),
                        run_time=0.08,
                    )
                except (KeyError, IndexError):
                    pass

            c_2d_i = out_idx if C.ndim >= 2 else (0, out_idx[0])
            try:
                self.play(
                    c_grid.cell(*c_2d_i).animate.set_fill(
                        color=PALETTE["fill_4"], opacity=0.9),
                    run_time=0.18,
                )
            except (KeyError, IndexError):
                pass

            self.play(FadeOut(sub2), run_time=0.08)
            _unfix(sub2, scene=self)

        if len(out_indices) > 8:
            note = make_label(f"… {len(out_indices)-8} more positions", 15,
                              PALETTE["dim_tag"])
            note.to_edge(DOWN, buff=0.5)
            _fix(note, scene=self)
            self.play(FadeIn(note))
            self.wait(0.8)
            self.play(FadeOut(note))

        done = make_label("✓  einsum complete", 20, PALETTE["fill_1"])
        done.to_edge(DOWN, buff=0.4)
        _fix(done, scene=self)
        self.play(FadeIn(done))
        self.wait(1.5)
