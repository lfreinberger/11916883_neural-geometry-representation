"""Shape metrics on a dense grid: Chamfer distance, IoU and topology errors."""

import numpy as np
import torch
from scipy.spatial import cKDTree
from shapely.geometry.base import BaseGeometry
from skimage import measure

from neural_geometry.model import LatentSDF
from neural_geometry.shapes import signed_distance

# no zero level at all, about the size of a shape
EMPTY_CHAMFER = 1.0


def grid(resolution: int) -> np.ndarray:
    """Regular grid on [-1, 1]^2, shape (resolution, resolution, 2)."""
    axis = np.linspace(-1, 1, resolution, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    return np.stack([x, y], axis=-1)


def sdf_grid(shapes: list[BaseGeometry], resolution: int) -> np.ndarray:
    points = grid(resolution).reshape(-1, 2)
    return np.stack([signed_distance(s, points).reshape(resolution, resolution) for s in shapes])


@torch.no_grad()
def predict_grid(
    model: LatentSDF, codes: torch.Tensor, resolution: int, batch: int = 8
) -> np.ndarray:
    points = torch.from_numpy(grid(resolution).reshape(-1, 2)).to(codes.device)
    fields = [model(c, points.expand(len(c), -1, -1)).cpu() for c in codes.split(batch)]
    return torch.cat(fields).reshape(-1, resolution, resolution).numpy()


def zero_level(field: np.ndarray) -> np.ndarray:
    """Points (N, 2) on the zero level of a field from sdf_grid or predict_grid."""
    contours = measure.find_contours(field, 0.0)
    if not contours:
        return np.zeros((0, 2))
    # (row, col) = (y, x) in pixels
    return np.concatenate(contours)[:, ::-1] * 2 / (len(field) - 1) - 1


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return EMPTY_CHAMFER
    return (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return (a & b).sum() / max((a | b).sum(), 1)


def topology_error(pred: np.ndarray, true: np.ndarray) -> int:
    """Wrong number of parts plus wrong number of holes, from two inside masks."""
    # outside components = holes + the one around the shape
    parts = abs(measure.label(pred).max() - measure.label(true).max())
    holes = abs(measure.label(~pred).max() - measure.label(~true).max())
    return parts + holes


def evaluate(pred: np.ndarray, true: np.ndarray) -> dict[str, np.ndarray]:
    """Per-shape metrics for predicted and true sdf grids (S, R, R)."""
    metrics = {"chamfer": [], "iou": [], "topology": []}
    for p, t in zip(pred, true, strict=True):
        metrics["chamfer"].append(chamfer(zero_level(p), zero_level(t)))
        metrics["iou"].append(iou(p < 0, t < 0))
        metrics["topology"].append(topology_error(p < 0, t < 0))
    return {name: np.array(values) for name, values in metrics.items()}


def summarize(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    # the mean alone hides single failed shapes
    chamfer, iou = metrics["chamfer"], metrics["iou"]
    summary = {
        "chamfer": chamfer.mean(),
        "chamfer_median": np.median(chamfer),
        "chamfer_max": chamfer.max(),
        "iou": iou.mean(),
        "iou_min": iou.min(),
        "topology": metrics["topology"].mean(),
        "empty": (chamfer == EMPTY_CHAMFER).sum(),
    }
    return {name: float(value) for name, value in summary.items()}
