# tensoranim

**Animated tensor transformation visualizations** — a Manim-powered companion to [TensorDiagram](https://github.com/hardik-vala/tensordiagram).

trying to animate some of tensor diagrams visualizations

---

## install

```bash
pip install manim          # animation backend
pip install numpy          # tensor data
# clone this repo and add to PYTHONPATH, or pip install -e .
```

---

## quick start

```python
import numpy as np
import tensoranim as ta

# reshape (3,4) → (4,3)
ta.animate_reshape(
    source_data  = np.arange(12).reshape(3, 4),
    target_shape = (4, 3),
    output_path  = "reshape.mp4",
)

# matrix multiply
ta.animate_matmul(
    A = np.array([[1,2,3],[4,5,6]], dtype=float),
    B = np.array([[1,4],[2,5],[3,6]], dtype=float),
    output_path = "matmul.mp4",
)

# row-wise softmax with heat-map transition
ta.animate_softmax(
    source_data = np.random.randn(3, 5),
    output_path = "softmax.mp4",
)

# scaled dot-product attention
ta.animate_attention(
    Q = np.random.randn(4, 8),
    K = np.random.randn(4, 8),
    V = np.random.randn(4, 8),
    output_path = "attention.mp4",
)
```

---

## all scenes

| Function | Scene class | What it shows |
|---|---|---|
| `animate_reshape` | `ReshapeScene` | cells fly from source → target positions |
| `animate_transpose` | `TransposeScene` | rows become columns |
| `animate_slice` | `SliceScene` | highlights a sub-tensor and extracts it |
| `animate_matmul` | `MatMulScene` | row × column pairs light up as C fills in |
| `animate_concat` | `ConcatScene` | two tensors merge along an axis |
| `animate_broadcast` | `BroadcastScene` | vector copies across rows |
| `animate_softmax` | `SoftmaxScene` | logits → probability heat-map |
| `animate_attention` | `AttentionScene` | Q K V → scores → attention → output |

---

## quality settings

```python
ta.animate_reshape(..., quality="low_quality")        # fast preview (480p)
ta.animate_reshape(..., quality="medium_quality")     # balanced (720p)
ta.animate_reshape(..., quality="high_quality")       # 1080p
ta.animate_reshape(..., quality="production_quality") # 4K
```

---

## use scene classes directly

```python
from manim import config
from tensoranim import MatMulScene
import numpy as np

MatMulScene.A = np.random.randn(4, 6)
MatMulScene.B = np.random.randn(6, 3)
MatMulScene.show_values = False
```

Then render with Manim's CLI:
```bash
manim -pql examples.py MatMulScene
```

---

## custom TensorGrid

Build your own scenes using the `TensorGrid` primitive:

```python
from manim import Scene
from tensoranim import TensorGrid, PALETTE
import numpy as np

class MyScene(Scene):
    def construct(self):
        data = np.random.randn(3, 4)
        grid = TensorGrid(
            data,
            cell_size   = 0.6,
            fill_color  = lambda idx, val: "#3B82F6" if val > 0 else "#EF4444",
            show_values = False,
        )
        self.add(grid)
        self.wait(1)
```

---

## design decisions

- **Manim** over matplotlib animations: Manim gives precise control over per-object timing, easing, and transforms which are essential for tensor pedagogy.
- **Cell-level granularity**: every grid cell is a separate Manim `Rectangle`, so you can animate individual elements flying to new positions.
- **Dark-mode palette** matching TensorDiagram's aesthetic.
- **No PyTorch/JAX required**: all scenes take plain `numpy` arrays; convert upstream with `.numpy()` or `np.array(tensor)`.

---

## license
MIT
