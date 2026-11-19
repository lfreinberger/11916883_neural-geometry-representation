import numpy as np
import shapely

from neural_geometry.mpeg7 import DEFECTIVE, shape_names, silhouette_to_polygon, split
from neural_geometry.shapes import HALF_WIDTH, signed_distance


def disk_mask(size, radius):
    rows, cols = np.mgrid[:size, :size]
    return np.hypot(rows - size / 2, cols - size / 2) < radius


def test_ring_becomes_polygon_with_hole():
    ring = disk_mask(100, 40) & ~disk_mask(100, 20)
    shape = silhouette_to_polygon(ring)
    assert len(shapely.get_parts(shape)) == 1 and len(shape.interiors) == 1
    xmin, ymin, xmax, ymax = shape.bounds
    np.testing.assert_allclose([xmin + xmax, ymin + ymax], [0, 0], atol=1e-9)
    np.testing.assert_allclose(max(xmax - xmin, ymax - ymin), 2 * HALF_WIDTH)
    # center of the hole is outside, the ring itself inside
    sdf = signed_distance(shape, np.array([[0.0, 0.0], [0.6, 0.0]]))
    assert sdf[0] > 0 > sdf[1]


def test_image_rows_are_flipped_to_y_up():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:90] = True  # wide bar at the top of the image
    mask[30:90, 40:60] = True  # narrow stem below it
    shape = silhouette_to_polygon(mask)
    sdf = signed_distance(shape, np.array([[-0.7, 0.7], [-0.7, -0.7]]))
    assert sdf[0] < 0 < sdf[1]


def test_scan_noise_is_removed():
    mask = disk_mask(100, 40)
    mask[2:4, 2:4] = True  # speck outside
    mask[50:52, 30:32] = False  # pinhole inside
    shape = silhouette_to_polygon(mask)
    assert len(shapely.get_parts(shape)) == 1 and len(shape.interiors) == 0


def test_shape_names_skip_non_dataset_files(tmp_path):
    for name in ["apple-1.gif", "apple-2.gif", "shapedata.gif", "confusions.eps"]:
        (tmp_path / name).touch()
    assert shape_names(tmp_path) == ["apple-1.gif", "apple-2.gif"]


def test_split_picks_one_validation_and_one_test_shape_per_class():
    labels = ["apple", "device0", "cellular_phone"]
    names = [f"{label}-{index}.gif" for label in labels for index in range(1, 21)]
    result = split(names)
    assert split(list(reversed(names))) == result
    for subset in ("val", "test"):
        assert sorted(name.rsplit("-", 1)[0] for name in result[subset]) == sorted(labels)
    assert not set(result["val"]) & set(result["test"])


def test_split_skips_defective_silhouettes():
    result = split(sorted(DEFECTIVE) + ["dog-1.gif", "dog-2.gif"])
    assert sorted(result["val"] + result["test"]) == ["dog-1.gif", "dog-2.gif"]
