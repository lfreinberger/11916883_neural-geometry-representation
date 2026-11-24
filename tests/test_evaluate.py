import numpy as np
import pytest
import torch
from shapely import affinity
from shapely.geometry import box

from neural_geometry.evaluate import (
    EMPTY_CHAMFER,
    evaluate,
    grid,
    predict_grid,
    sdf_grid,
    summarize,
    zero_level,
)
from neural_geometry.model import LatentSDF
from neural_geometry.shapes import ellipse


def test_zero_level_has_x_and_y_the_right_way_round():
    wide = box(-0.6, -0.2, 0.6, 0.2)
    points = zero_level(sdf_grid([wide], 128)[0])
    np.testing.assert_allclose(points.min(axis=0), [-0.6, -0.2], atol=0.02)
    np.testing.assert_allclose(points.max(axis=0), [0.6, 0.2], atol=0.02)


def test_chamfer_of_shifted_circle():
    circle = ellipse(0.5, 0.5)
    shift = 0.05
    true, pred = sdf_grid([circle, affinity.translate(circle, shift)], 256)
    chamfer = evaluate(pred[None], true[None])["chamfer"][0]
    # mean of |shift * cos(angle)| over the circle
    assert chamfer == pytest.approx(2 * shift / np.pi, rel=0.05)


def test_topology_counts_parts_and_holes():
    square = box(-0.5, -0.5, 0.5, 0.5)
    island = square.union(box(0.6, 0.6, 0.7, 0.7))
    hole = square.difference(box(-0.1, -0.1, 0.1, 0.1))
    true = sdf_grid([square, square, hole], 64)
    pred = sdf_grid([island, hole, square], 64)
    np.testing.assert_array_equal(evaluate(pred, true)["topology"], [1, 1, 1])


def test_empty_prediction_is_counted():
    true = sdf_grid([ellipse(0.5, 0.5)] * 2, 64)
    pred = np.stack([true[0], np.ones((64, 64))])
    summary = summarize(evaluate(pred, true))
    assert summary["chamfer_max"] == EMPTY_CHAMFER
    assert summary["iou_min"] == 0
    assert summary["empty"] == 1


def test_predict_grid_matches_model():
    torch.manual_seed(0)
    model = LatentSDF(n_ctrl=5, degree=2, code_length=1)
    codes = torch.randn(3, 5, 5, 1)
    points = torch.from_numpy(grid(32).reshape(1, -1, 2)).expand(3, -1, -1)
    expected = model(codes, points).reshape(3, 32, 32).numpy()
    np.testing.assert_allclose(predict_grid(model, codes, 32, batch=2), expected, atol=1e-6)