"""
tensoranim/core.py
──────────────────
Low-level Manim building blocks for rendering tensors as coloured grids.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Sequence, Optional, Tuple, Union

from manim import (
    VGroup, Rectangle, Text, MathTex, Arrow, FadeIn, FadeOut,
    AnimationGroup, Transform, Write, Create,
    WHITE, BLACK, GREY, BLUE_D, GREEN_D, ORANGE, RED_D, YELLOW_D,
    LEFT, RIGHT, UP, DOWN, ORIGIN,
    MED_SMALL_BUFF, SMALL_BUFF,
)
from manim import Scene, MovingCameraScene

# ── palette ──────────────────────────────────────────────────────────────────

PALETTE = {
    "cell":     "#2A2D3E",
    "border":   "#7B8CDE",
    "fill_0":   "#3B82F6",   # blue
    "fill_1":   "#10B981",   # green
    "fill_2":   "#F59E0B",   # amber
    "fill_3":   "#EF4444",   # red
    "fill_4":   "#8B5CF6",   # violet
    "label":    "#E2E8F0",
    "dim_tag":  "#94A3B8",
    "bg":       "#0F172A",
    "arrow":    "#CBD5E1",
    "title":    "#F1F5F9",
}

ColorSpec = Union[str, Callable[[Tuple, float], str], None]


def _resolve_color(color: ColorSpec, idx: tuple, val: float, default: str) -> str:
    if color is None:
        return default
    if callable(color):
        return color(idx, val)
    return color


# ── TensorGrid ───────────────────────────────────────────────────────────────

class TensorGrid(VGroup):
    """
    A Manim VGroup that renders an N-dimensional tensor as a coloured grid.

    Supports 1-D, 2-D, and 3-D tensors.  For 3-D tensors the depth slices are
    drawn with an isometric offset so the volume is readable.

    Parameters
    ----------
    data        : numpy array (1-D / 2-D / 3-D)
    cell_size   : side length of each cell in Manim units
    fill_color  : colour string, or callable(idx, value) → colour string
    show_values : whether to print the numeric value inside each cell
    label       : optional string label placed below the grid
    """

    def __init__(
        self,
        data: np.ndarray,
        cell_size: float = 0.55,
        fill_color: ColorSpec = PALETTE["fill_0"],
        show_values: bool = False,
        label: Optional[str] = None,
        stroke_width: float = 1.5,
        fill_opacity: float = 0.85,
        value_format: str = ".0f",
        value_color: str = PALETTE["label"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        data = np.asarray(data)
        self.data = data
        self.cell_size = cell_size
        self._cells: dict[tuple, Rectangle] = {}
        self._texts: dict[tuple, Text] = {}

        ndim = data.ndim
        if ndim == 1:
            data = data[np.newaxis, :]      # treat as (1, N)
        if ndim > 3:
            raise ValueError("TensorGrid supports up to 3-D tensors.")

        if data.ndim == 2:
            self._build_2d(data, cell_size, fill_color, show_values,
                           stroke_width, fill_opacity, value_format, value_color)
        elif data.ndim == 3:
            self._build_3d(data, cell_size, fill_color, show_values,
                           stroke_width, fill_opacity)

        if label:
            lbl = Text(label, font_size=18, color=PALETTE["dim_tag"])
            lbl.next_to(self, DOWN, buff=SMALL_BUFF)
            self.add(lbl)

    # ── 2-D ──────────────────────────────────────────────────────────────────

    def _build_2d(self, data, cs, fill_color, show_values, sw, fo, value_format=".0f", value_color=PALETTE["label"]):
        rows, cols = data.shape
        for r in range(rows):
            for c in range(cols):
                val = float(data[r, c])
                color = _resolve_color(fill_color, (r, c), val, PALETTE["fill_0"])
                rect = Rectangle(
                    width=cs, height=cs,
                    fill_color=color, fill_opacity=fo,
                    stroke_color=PALETTE["border"], stroke_width=sw,
                )
                rect.move_to([c * cs, -r * cs, 0])
                self._cells[(r, c)] = rect
                self.add(rect)
                if show_values:
                    txt = Text(f"{val:{value_format}}", font_size=int(cs * 22), color=value_color)
                    txt.move_to(rect.get_center())
                    self._texts[(r, c)] = txt
                    self.add(txt)

    # ── 3-D ──────────────────────────────────────────────────────────────────

    def _build_3d(self, data, cs, fill_color, show_values, sw, fo):
        depth, rows, cols = data.shape
        iso = cs * 0.35   # isometric x/y offset per depth-layer
        for d in range(depth):
            offset = np.array([d * iso, d * iso * 0.5, 0])
            for r in range(rows):
                for c in range(cols):
                    val = float(data[d, r, c])
                    color = _resolve_color(fill_color, (d, r, c), val, PALETTE["fill_0"])
                    rect = Rectangle(
                        width=cs, height=cs,
                        fill_color=color,
                        fill_opacity=fo - d * 0.08,   # slight fade for depth
                        stroke_color=PALETTE["border"], stroke_width=sw,
                    )
                    pos = np.array([c * cs, -r * cs, 0]) + offset
                    rect.move_to(pos)
                    self._cells[(d, r, c)] = rect
                    self.add(rect)

    # ── helpers ───────────────────────────────────────────────────────────────

    def cell(self, *idx) -> Rectangle:
        return self._cells[idx]

    def highlight(self, *idx, color: str = PALETTE["fill_2"]) -> None:
        """Permanently recolour a cell (call inside a scene)."""
        self._cells[idx].set_fill(color=color)

    def get_cells(self) -> list[Rectangle]:
        return list(self._cells.values())


# ── helpers for scenes ────────────────────────────────────────────────────────

def make_label(text: str, font_size: int = 22, color: str = PALETTE["title"]) -> Text:
    return Text(text, font_size=font_size, color=color)


def make_arrow(start_mob, end_mob, label: str = "", color: str = PALETTE["arrow"]):
    arr = Arrow(
        start_mob.get_right() + RIGHT * SMALL_BUFF,
        end_mob.get_left()  + LEFT  * SMALL_BUFF,
        buff=0.05,
        color=color,
        stroke_width=2,
        max_tip_length_to_length_ratio=0.15,
    )
    grp = VGroup(arr)
    if label:
        lbl = Text(label, font_size=16, color=PALETTE["dim_tag"])
        lbl.next_to(arr, UP, buff=0.05)
        grp.add(lbl)
    return grp


def shape_tag(shape: tuple, font_size: int = 16) -> Text:
    s = " × ".join(str(d) for d in shape)
    return Text(f"({s})", font_size=font_size, color=PALETTE["dim_tag"])


# ── 3-D primitives ────────────────────────────────────────────────────────────

from manim import ThreeDScene, Prism, ThreeDAxes, PI, TAU, IN, OUT

class ThreeDTensorGrid(VGroup):
    """
    Renders a 3-D tensor as a grid of Prisms in true 3-D space.
    For use inside TensorScene3D (a ThreeDScene subclass).

    Axis convention: X = columns (+right), Y = -rows (+down), Z = -depth (+into screen)
    matching numpy's (D, R, C) layout.
    """

    def __init__(
        self,
        data: np.ndarray,
        cell_size: float = 0.5,
        fill_color: ColorSpec = PALETTE["fill_0"],
        fill_opacity: float = 0.75,
        stroke_width: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        data = np.asarray(data)
        if data.ndim != 3:
            raise ValueError("ThreeDTensorGrid requires a 3-D tensor.")
        self.data = data
        self.cell_size = cell_size
        self._cells: dict = {}
        D, R, C = data.shape
        cs = cell_size
        for d in range(D):
            for r in range(R):
                for c in range(C):
                    val = float(data[d, r, c])
                    color = _resolve_color(fill_color, (d, r, c), val, PALETTE["fill_0"])
                    box = Prism(
                        dimensions=[cs * 0.92, cs * 0.92, cs * 0.92],
                        fill_color=color,
                        fill_opacity=fill_opacity,
                        stroke_color=PALETTE["border"],
                        stroke_width=stroke_width,
                    )
                    box.move_to(np.array([c * cs, -r * cs, -d * cs]))
                    self._cells[(d, r, c)] = box
                    self.add(box)

    def cell(self, d: int, r: int, c: int):
        return self._cells[(d, r, c)]

    def get_slice_d(self, d: int) -> VGroup:
        return VGroup(*[v for (dd, r, c), v in self._cells.items() if dd == d])

    def get_slice_r(self, r: int) -> VGroup:
        return VGroup(*[v for (d, rr, c), v in self._cells.items() if rr == r])

    def get_slice_c(self, c: int) -> VGroup:
        return VGroup(*[v for (d, r, cc), v in self._cells.items() if cc == c])

    def get_row(self, d: int, r: int) -> VGroup:
        return VGroup(*[v for (dd, rr, c), v in self._cells.items()
                        if dd == d and rr == r])

    def get_col(self, d: int, c: int) -> VGroup:
        return VGroup(*[v for (dd, r, cc), v in self._cells.items()
                        if dd == d and cc == c])


class TensorScene3D(ThreeDScene):
    """
    Base class for 3-D tensor scenes.
    Sets camera to isometric angle; provides title/subtitle as fixed-frame overlays.
    """

    def setup(self):
        self.camera.background_color = PALETTE["bg"]
        self.set_camera_orientation(phi=65 * PI / 180, theta=-45 * PI / 180)

    def title(self, text: str) -> Text:
        t = Text(text, font_size=24, color=PALETTE["title"])
        t.to_corner(UP + LEFT, buff=0.3)
        self.add_fixed_in_frame_mobjects(t)
        return t

    def subtitle(self, text: str) -> Text:
        t = Text(text, font_size=16, color=PALETTE["dim_tag"])
        t.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(t)
        return t
