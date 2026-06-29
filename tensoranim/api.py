"""
tensoranim/api.py
─────────────────
Convenience wrappers that let you animate tensor ops in a single function call.
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union

from manim import config, tempconfig

from .scenes import (
    ReshapeScene, TransposeScene, SliceScene, MatMulScene,
    ConcatScene, BroadcastScene, SoftmaxScene, AttentionScene,
)


# ── internal helper ──────────────────────────────────────────────────────────

def render_scene(
    SceneClass,
    output_path: str = "output.mp4",
    quality: str = "medium_quality",
    fps: int = 15,
    **scene_attrs,
):
    """
    Render any TensorScene subclass to a video file.

    Parameters
    ----------
    SceneClass   : class (subclass of TensorScene / Scene)
    output_path  : destination file path (mp4 / gif)
    quality      : "low_quality" | "medium_quality" | "high_quality" | "production_quality"
    fps          : frames-per-second (default 15 is fine for debugging)
    **scene_attrs: attributes to set on SceneClass before rendering
    """
    # patch class attributes
    for k, v in scene_attrs.items():
        setattr(SceneClass, k, v)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempconfig({
        "quality":        quality,
        "output_file":    str(out.stem),
        "media_dir":      str(out.parent / "_media"),
        "save_last_frame": False,
        "frame_rate":     fps,
    }):
        scene = SceneClass()
        scene.render()

    # manim puts output inside _media/videos/…; move it to the requested path
    media_dir = out.parent / "_media" / "videos" / out.stem
    candidates = list(media_dir.rglob("*.mp4")) + list(media_dir.rglob("*.gif"))
    if candidates:
        import shutil
        shutil.move(str(candidates[-1]), str(out))
        # clean up media dir
        shutil.rmtree(str(out.parent / "_media"), ignore_errors=True)

    return str(out)


# ── public convenience functions ─────────────────────────────────────────────

def animate_reshape(
    source_data:  np.ndarray,
    target_shape: tuple,
    output_path:  str = "reshape.mp4",
    quality:      str = "medium_quality",
    show_values:  bool = True,
    fill_color:   str = "#3B82F6",
    fps:          int = 15,
) -> str:
    """Animate tensor.reshape(target_shape) and save to output_path."""
    return render_scene(
        ReshapeScene, output_path=output_path, quality=quality, fps=fps,
        source_data=source_data, target_shape=target_shape,
        show_values=show_values, fill_color=fill_color,
    )


def animate_transpose(
    source_data: np.ndarray,
    output_path: str = "transpose.mp4",
    quality:     str = "medium_quality",
    show_values: bool = True,
    fps:         int = 15,
) -> str:
    """Animate matrix.T and save to output_path."""
    return render_scene(
        TransposeScene, output_path=output_path, quality=quality, fps=fps,
        source_data=source_data, show_values=show_values,
    )


def animate_slice(
    source_data: np.ndarray,
    row_slice:   object = None,
    col_slice:   object = None,
    output_path: str = "slice.mp4",
    quality:     str = "medium_quality",
    show_values: bool = True,
    fps:         int = 15,
) -> str:
    """Animate a 2-D tensor slice and save to output_path."""
    return render_scene(
        SliceScene, output_path=output_path, quality=quality, fps=fps,
        source_data=source_data, row_slice=row_slice, col_slice=col_slice,
        show_values=show_values,
    )


def animate_matmul(
    A:           np.ndarray,
    B:           np.ndarray,
    output_path: str = "matmul.mp4",
    quality:     str = "medium_quality",
    show_values: bool = True,
    fps:         int = 15,
) -> str:
    """Animate A @ B = C and save to output_path."""
    return render_scene(
        MatMulScene, output_path=output_path, quality=quality, fps=fps,
        A=A, B=B, show_values=show_values,
    )


def animate_concat(
    tensors:     List[np.ndarray],
    axis:        int = 0,
    output_path: str = "concat.mp4",
    quality:     str = "medium_quality",
    fps:         int = 15,
) -> str:
    """Animate torch.cat(tensors, dim=axis) and save to output_path."""
    return render_scene(
        ConcatScene, output_path=output_path, quality=quality, fps=fps,
        tensors=tensors, axis=axis,
    )


def animate_broadcast(
    vector_data: np.ndarray,
    n_rows:      int = 3,
    output_path: str = "broadcast.mp4",
    quality:     str = "medium_quality",
    fps:         int = 15,
) -> str:
    """Animate broadcasting a (1, N) vector to (n_rows, N)."""
    return render_scene(
        BroadcastScene, output_path=output_path, quality=quality, fps=fps,
        vector_data=vector_data, n_rows=n_rows,
    )


def animate_softmax(
    source_data: np.ndarray,
    output_path: str = "softmax.mp4",
    quality:     str = "medium_quality",
    fps:         int = 15,
) -> str:
    """Animate row-wise softmax as a heat-map transition."""
    return render_scene(
        SoftmaxScene, output_path=output_path, quality=quality, fps=fps,
        source_data=source_data,
    )


def animate_attention(
    Q:           np.ndarray,
    K:           np.ndarray,
    V:           np.ndarray,
    output_path: str = "attention.mp4",
    quality:     str = "medium_quality",
    fps:         int = 15,
) -> str:
    """Animate scaled dot-product attention: softmax(QKᵀ/√d) V."""
    return render_scene(
        AttentionScene, output_path=output_path, quality=quality, fps=fps,
        Q=Q, K=K, V=V,
    )
