"""
tensoranim/model_scene.py
─────────────────────────
ModelScene — a single Manim scene that plays through every op in a TensorGraph,
maintaining a scrolling left-to-right layout so the full forward pass is visible
as one coherent animation.

Layout strategy
───────────────
Tensors are arranged left-to-right in a strip.  When the strip would exceed the
screen width, the camera slides right to keep the current op centred (moving
camera via MovingCameraScene).  Each grid is labelled with its name and shape.

The scene is never instantiated directly — TensorGraph.render() configures the
class attributes and then calls render_scene(ModelScene, ...).
"""

from __future__ import annotations

import numpy as np
from manim import (
    MovingCameraScene, VGroup, Text, FadeIn, FadeOut, Write, Create,
    AnimationGroup, Arrow,
    LEFT, RIGHT, UP, DOWN, ORIGIN,
    SMALL_BUFF, MED_SMALL_BUFF, MED_LARGE_BUFF,
    WHITE,
)

from .core import (
    TensorGrid, make_label, make_arrow, shape_tag, PALETTE, _resolve_color
)
from .ops import TensorGraph, TensorState, GraphNode, OpBlock, _auto_cell_size


# ── constants ─────────────────────────────────────────────────────────────────

GRID_GAP   = 1.4    # horizontal gap between grids (Manim units)
LABEL_BUFF = 0.18   # gap between grid and its name label
TAG_BUFF   = 0.12   # gap between name label and shape tag

# ── ModelScene ────────────────────────────────────────────────────────────────

class ModelScene(MovingCameraScene):
    """
    Renders a full TensorGraph as a single left-to-right animation.

    Class attributes (set by TensorGraph.render before instantiation)
    ─────────────────────────────────────────────────────────────────
    graph       : TensorGraph   (already run — all states populated)
    cell_size   : float or None (None → auto per grid)
    show_values : bool
    """

    graph:       "TensorGraph" = None
    cell_size:   float         = None
    show_values: bool          = False

    # ── setup ────────────────────────────────────────────────────────────────

    def setup(self):
        super().setup()
        self.camera.background_color = PALETTE["bg"]

    # ── main ─────────────────────────────────────────────────────────────────

    def construct(self):
        g = self.graph
        assert g is not None, "ModelScene.graph must be set before rendering."

        # ── title ─────────────────────────────────────────────────────────
        title = Text("Model Forward Pass", font_size=28, color=PALETTE["title"])
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # ── collect ordered states: inputs first, then each op's output ───
        # We render every state in topological order (inputs → nodes).
        state_names: list[str] = []
        for name in g._states:
            if name not in [n.output for n in g._nodes]:
                state_names.append(name)   # pure inputs
        for node in g._nodes:
            state_names.append(node.output)

        # ── build and position all grids ──────────────────────────────────
        grids:  dict[str, TensorGrid] = {}
        labels: dict[str, VGroup]     = {}
        cursor_x = 0.0

        for name in state_names:
            state = g._states[name]
            cs    = self.cell_size or _auto_cell_size(state.shape)
            grid  = state.make_grid(cell_size=cs, show_values=self.show_values)

            # centre grid at cursor_x
            grid.move_to([cursor_x + grid.get_width() / 2, 0, 0])

            # name label above, shape tag below
            n_lbl = make_label(state.name, 18)
            s_tag = shape_tag(state.shape, font_size=14)
            n_lbl.next_to(grid, UP,   buff=LABEL_BUFF)
            s_tag.next_to(grid, DOWN, buff=TAG_BUFF)

            lbl_group = VGroup(n_lbl, s_tag)
            grids[name]  = grid
            labels[name] = lbl_group

            cursor_x += grid.get_width() + GRID_GAP

        # total width of the strip
        total_w = cursor_x - GRID_GAP

        # ── animate inputs ────────────────────────────────────────────────
        input_names = [n for n in state_names
                       if n not in [nd.output for nd in g._nodes]]

        if input_names:
            self._camera_to(grids[input_names[-1]])
            for name in input_names:
                self.play(
                    FadeIn(grids[name]),
                    FadeIn(labels[name]),
                    run_time=0.4,
                )
            self.wait(0.3)

        # ── animate each op ───────────────────────────────────────────────
        for node in g._nodes:
            in_states  = [g._states[n] for n in node.inputs]
            out_state  = g._states[node.output]
            in_grids   = [grids[n] for n in node.inputs if n in grids]
            out_grid   = grids[node.output]
            out_labels = labels[node.output]

            # slide camera to keep current op centred
            self._camera_to(out_grid)

            # op name subtitle
            sub = Text(
                f"{node.op.__class__.__name__}  ·  {node.inputs} → {node.output}",
                font_size=16, color=PALETTE["dim_tag"],
            )
            sub.to_edge(DOWN, buff=0.35)
            self.play(FadeIn(sub), run_time=0.2)

            # delegate animation to the op
            node.op.animate(
                scene=self,
                inputs=in_states,
                output=out_state,
                in_grids=in_grids,
                out_grid=out_grid,
            )

            # show output label
            self.play(FadeIn(out_labels), run_time=0.3)
            self.wait(0.3)
            self.play(FadeOut(sub), run_time=0.2)

        self.wait(0.5)

        # ── final overview: zoom out to show the full strip ───────────────
        if total_w > 0:
            overview_sub = Text(
                "Full forward pass complete", font_size=20, color=PALETTE["fill_1"]
            )
            overview_sub.to_edge(DOWN, buff=0.35)
            self.play(
                self.camera.frame.animate.set_width(min(total_w + 2.0, 30.0))
                                         .move_to([total_w / 2, 0, 0]),
                FadeIn(overview_sub),
                run_time=1.2,
            )
            self.wait(1.5)

    # ── camera helper ─────────────────────────────────────────────────────────

    def _camera_to(self, mob, padding: float = 2.5):
        """Smoothly slide the camera to centre on mob."""
        cx = mob.get_center()[0]
        self.play(
            self.camera.frame.animate.move_to([cx, 0, 0]),
            run_time=0.5,
        )
