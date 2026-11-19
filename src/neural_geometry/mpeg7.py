"""MPEG-7 CE-Shape-1 silhouettes (70 classes x 20 GIFs) and the val/test split.

The archive has to be downloaded by hand (see README) and unpacked to
data/mpeg7/original/.
"""

import re
from pathlib import Path

import numpy as np
import shapely
from PIL import Image
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from skimage import filters, measure

from neural_geometry.shapes import normalize

# "<class>-<i>.gif"; also skips confusions.gif and shapedata.gif, which are in the archive
_FILE_NAME = re.compile(r"^(?P<label>.+)-(?P<index>\d+)\.gif$")

# the fur texture is stored as background pixels inside the body, which gives large fake
# holes (checked visually, see the dog class)
DEFECTIVE = frozenset(f"dog-{index}.gif" for index in (3, 6, 7, 14, 15, 16, 17, 19))


def shape_names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir() if _FILE_NAME.match(path.name))


def split(names: list[str], seed: int = 0) -> dict[str, list[str]]:
    """Per class, one silhouette for validation and a different one for testing.

    Defective silhouettes are never picked.
    """
    by_label: dict[str, list[str]] = {}
    for name in sorted(set(names) - DEFECTIVE):
        by_label.setdefault(_FILE_NAME.match(name)["label"], []).append(name)
    rng = np.random.default_rng(seed)
    picks = [rng.choice(members, size=2, replace=False) for _, members in sorted(by_label.items())]
    return {"val": [str(pick[0]) for pick in picks], "test": [str(pick[1]) for pick in picks]}


def silhouette_to_polygon(
    mask: np.ndarray,
    blur_sigma: float = 1.0,
    min_feature_fraction: float = 0.01,
    simplify_tolerance: float = 0.25,
) -> BaseGeometry:
    """Polygon of the foreground of a binary mask, normalized like the training scenes."""
    # without the blur the contour follows the pixel staircase
    image = filters.gaussian(np.pad(mask.astype(float), 2), sigma=blur_sigma)
    shape = Polygon()
    for contour in measure.find_contours(image, 0.5):
        # rows go down, y goes up
        ring = Polygon(np.column_stack([contour[:, 1], -contour[:, 0]]))
        # hole contours lie inside their outer contour, so the xor of all rings gives the
        # right holes without sorting the contours
        shape = shape.symmetric_difference(ring)
    # small parts and holes are scan noise and would dominate the Chamfer distance
    min_area = min_feature_fraction * shape.area
    parts = [
        Polygon(part.exterior, [hole for hole in part.interiors if Polygon(hole).area >= min_area])
        for part in shapely.get_parts(shape)
        if part.area >= min_area
    ]
    # a quarter pixel is invisible but removes most vertices -> much faster distance queries
    shape = shapely.union_all(parts).simplify(simplify_tolerance)
    return normalize(shape)


def load_shape(path: Path) -> BaseGeometry:
    return silhouette_to_polygon(np.asarray(Image.open(path)) > 0)
