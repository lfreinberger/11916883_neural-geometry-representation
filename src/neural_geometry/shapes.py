"""Primitive shapes, signed distances and point sampling in 2D."""

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry

# all shapes are scaled to [-0.8, 0.8]^2, so there is some margin to the border of
# the latent grid at +-1
HALF_WIDTH = 0.8


def signed_distance(shape: BaseGeometry, points: np.ndarray) -> np.ndarray:
    """Signed distance of points (N, 2) to the boundary of shape, negative inside."""
    distance = shapely.distance(shape.boundary, shapely.points(points))
    inside = shapely.contains_xy(shape, points[:, 0], points[:, 1])
    return np.where(inside, -distance, distance)


def normalize(shape: BaseGeometry, half_width: float = HALF_WIDTH) -> BaseGeometry:
    """Center the shape at the origin and scale its longer side to 2 * half_width."""
    xmin, ymin, xmax, ymax = shape.bounds
    shape = affinity.translate(shape, -(xmin + xmax) / 2, -(ymin + ymax) / 2)
    factor = 2 * half_width / max(xmax - xmin, ymax - ymin)
    return affinity.scale(shape, factor, factor, origin=(0, 0))


def rounded_rectangle(width: float, height: float, radius: float) -> BaseGeometry:
    core = box(-width / 2 + radius, -height / 2 + radius, width / 2 - radius, height / 2 - radius)
    return core.buffer(radius, quad_segs=16) if radius > 0 else core


def ellipse(semi_x: float, semi_y: float) -> BaseGeometry:
    # 128-gon, so the sdf is exact for the polygon but not for the true ellipse
    return affinity.scale(Point(0, 0).buffer(1.0, quad_segs=32), semi_x, semi_y)


def random_primitive(rng: np.random.Generator, max_shift: float = 0.5) -> BaseGeometry:
    """Random rectangle or ellipse, rotated and shifted."""
    width, height = rng.uniform(0.2, 0.8, size=2)
    if rng.random() < 0.7:
        # sharp, fully rounded or partly rounded corners, the MPEG-7 shapes have all of them
        radius = rng.choice([0.0, 1.0, rng.uniform(0.0, 1.0)]) * min(width, height) / 2
        primitive = rounded_rectangle(width, height, radius)
    else:
        primitive = ellipse(width / 2, height / 2)
    primitive = affinity.rotate(primitive, rng.uniform(0, 180), origin=(0, 0))
    return affinity.translate(primitive, *rng.uniform(-max_shift, max_shift, size=2))


def cut_hole(shape: BaseGeometry, rng: np.random.Generator) -> BaseGeometry:
    # Put the hole at the deepest of some random points and keep it smaller than the
    # distance to the boundary there. With a random position most holes broke through
    # the boundary and only made a notch.
    xmin, ymin, xmax, ymax = shape.bounds
    candidates = rng.uniform((xmin, ymin), (xmax, ymax), size=(256, 2))
    depths = -signed_distance(shape, candidates)
    hole = random_primitive(rng, max_shift=0.0)
    hole_radius = np.linalg.norm(shapely.get_coordinates(hole), axis=1).max()
    factor = rng.uniform(0.4, 0.8) * depths.max() / hole_radius
    if factor <= 0:
        return shape
    hole = affinity.scale(hole, factor, factor, origin=(0, 0))
    return shape.difference(affinity.translate(hole, *candidates[depths.argmax()]))


def random_scene(
    rng: np.random.Generator, max_primitives: int = 4, hole_probability: float = 0.3
) -> BaseGeometry:
    """Union of 1 to max_primitives random primitives, sometimes with a hole."""
    scene = shapely.union_all(
        [random_primitive(rng) for _ in range(rng.integers(1, max_primitives + 1))]
    )
    if rng.random() < hole_probability:
        scene = cut_hole(scene, rng)
    return normalize(scene)


def sample_points(
    shape: BaseGeometry,
    n_points: int,
    rng: np.random.Generator,
    near_fraction: float = 0.7,
    near_std: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points in [-1, 1]^2 and their signed distance.

    near_fraction of the points are boundary points plus Gaussian noise (std near_std),
    the rest is uniform. The near points are what matters for the zero level set, the
    uniform ones keep the far field reasonable.
    """
    n_near = int(near_fraction * n_points)
    boundary = shape.boundary
    on_boundary = shapely.get_coordinates(
        shapely.line_interpolate_point(boundary, rng.uniform(0, boundary.length, n_near))
    )
    near = np.clip(on_boundary + rng.normal(0, near_std, (n_near, 2)), -1, 1)
    uniform = rng.uniform(-1, 1, (n_points - n_near, 2))
    points = np.concatenate([near, uniform]).astype(np.float32)
    return points, signed_distance(shape, points).astype(np.float32)


def sample_dataset(
    shapes: list[BaseGeometry], n_points: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns points (S, N, 2) and sdf (S, N) for S shapes."""
    rng = np.random.default_rng(seed)
    points, sdf = zip(*(sample_points(shape, n_points, rng) for shape in shapes), strict=True)
    return np.stack(points), np.stack(sdf)


def primitive_scenes(n_scenes: int, seed: int) -> list[BaseGeometry]:
    """Random training scenes. Use a different seed for the held-out scenes."""
    rng = np.random.default_rng(seed)
    return [random_scene(rng) for _ in range(n_scenes)]
