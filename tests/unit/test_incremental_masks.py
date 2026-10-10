"""Sensor-mask generation tests.

The mask families are drawn from seeds and are not shipped with the code, so these
tests pin the sampling rules rather than the sampled coordinates: the cylinder
section is not a candidate, and one draw of 50 points fixes every mask of a family.
"""

import numpy as np

from features.sensors import incremental_masks as im

NC_GRID = (80, 160)
ALL_GRID_POINTS = np.arange(NC_GRID[0] * NC_GRID[1], dtype=np.int64)


def _distance_to_axis(flat: np.ndarray) -> np.ndarray:
    rows, cols = np.divmod(np.asarray(flat, dtype=np.int64), NC_GRID[1])
    centre_row, centre_col = im.CYLINDER_CENTER_RC
    return np.hypot(rows - centre_row, cols - centre_col)


def test_candidates_exclude_the_cylinder_section():
    """No candidate lies inside the cylinder section and every other grid point stays."""
    candidates = im._exclude_body_candidates(ALL_GRID_POINTS, W=NC_GRID[1])
    assert np.all(_distance_to_axis(candidates) > im.CYLINDER_RADIUS)
    assert candidates.size == int((_distance_to_axis(ALL_GRID_POINTS) > im.CYLINDER_RADIUS).sum())
    assert np.all(np.isin(candidates, ALL_GRID_POINTS))


def test_one_draw_fixes_a_nested_family():
    """The masks of the five counts are the prefixes of a single draw without replacement."""
    draw = im._sample_incremental_random_coords(
        H=NC_GRID[0], W=NC_GRID[1], candidates_flat=ALL_GRID_POINTS, seed=20260522
    )
    assert draw.shape == (im.TOTAL_POINTS, 2)
    assert len({tuple(point) for point in draw.tolist()}) == im.TOTAL_POINTS
    for smaller, larger in ((10, 15), (15, 20), (20, 30), (30, 50)):
        assert np.array_equal(draw[:smaller], draw[:larger][:smaller])


def test_seed_determines_the_family():
    """A family is reproducible from its seed, and different seeds give different draws."""
    arguments = {"H": NC_GRID[0], "W": NC_GRID[1], "candidates_flat": ALL_GRID_POINTS}
    first = im._sample_incremental_random_coords(seed=20260806, **arguments)
    second = im._sample_incremental_random_coords(seed=20260806, **arguments)
    other = im._sample_incremental_random_coords(seed=20260807, **arguments)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
