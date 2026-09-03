"""Supplementary mask registry tests (P1-0)."""

import numpy as np
import pytest

from features.sensors import mask_registry as mr


def test_five_families():
    assert mr.list_families() == [f"family_{i:02d}" for i in range(1, 6)]


def test_family_seeds():
    assert mr.family_seed("family_01") == 20260522
    assert mr.family_seed("family_05") == 20260951


@pytest.mark.parametrize("fam", [f"family_{i:02d}" for i in range(1, 6)])
def test_family_nested_and_unique(fam):
    checks = mr.verify_family(fam)
    assert checks["strict_nested"]
    assert checks["all_unique"]


def test_families_pairwise_distinct():
    sets = {}
    for fam in mr.iter_families():
        c = mr.load_nc_mask(fam, 50)
        sets[fam] = set(map(tuple, c.tolist()))
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert sets[keys[i]] != sets[keys[j]], f"{keys[i]} == {keys[j]}"


def test_load_mask_shape():
    for M in mr.MASK_COUNTS:
        c = mr.load_nc_mask("family_01", M)
        assert c.shape == (M, 2)
        b = mr.load_nc_mask_bool("family_01", M)
        assert b.shape == (80, 160)
        assert int(b.sum()) == M
