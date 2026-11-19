import numpy as np
import pytest
import shapely
from shapely.geometry import box

from neural_geometry.shapes import (
    HALF_WIDTH,
    ellipse,
    primitive_scenes,
    sample_dataset,
    signed_distance,
)


def test_signed_distance_of_square():
    square = box(-0.5, -0.5, 0.5, 0.5)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.2]])
    expected = [-0.5, 0.5, np.sqrt(0.5), 0.0]
    np.testing.assert_allclose(signed_distance(square, points), expected, atol=1e-12)


def test_polygonal_circle_matches_analytic_sdf():
    # ellipse() is a 128-gon, not an exact circle
    points = np.random.default_rng(0).uniform(-1, 1, (1000, 2))
    analytic = np.linalg.norm(points, axis=1) - 0.5
    np.testing.assert_allclose(signed_distance(ellipse(0.5, 0.5), points), analytic, atol=2e-4)


def test_scenes_are_reproducible():
    first = primitive_scenes(5, seed=0)
    again = primitive_scenes(5, seed=0)
    other = primitive_scenes(5, seed=1)
    assert all(a.equals(b) for a, b in zip(first, again, strict=True))
    assert not any(a.equals(b) for a, b in zip(first, other, strict=True))


@pytest.mark.parametrize("seed", range(3))
def test_scenes_are_valid_and_fit_into_domain(seed):
    for scene in primitive_scenes(50, seed):
        assert scene.is_valid and scene.area > 0
        xmin, ymin, xmax, ymax = scene.bounds
        assert -HALF_WIDTH - 1e-9 <= min(xmin, ymin) and max(xmax, ymax) <= HALF_WIDTH + 1e-9


def test_some_scenes_have_holes():
    scenes = primitive_scenes(100, seed=0)
    n_with_holes = sum(
        any(len(part.interiors) > 0 for part in shapely.get_parts(scene)) for scene in scenes
    )
    assert 20 <= n_with_holes <= 40


def test_samples_concentrate_near_the_boundary():
    scenes = primitive_scenes(4, seed=0)
    points, sdf = sample_dataset(scenes, n_points=1000, seed=0)
    assert points.shape == (4, 1000, 2) and sdf.shape == (4, 1000)
    assert np.abs(points).max() <= 1
    assert (np.abs(sdf) < 0.02).mean() > 0.5
    np.testing.assert_allclose(sdf[2], signed_distance(scenes[2], points[2]), atol=1e-6)
