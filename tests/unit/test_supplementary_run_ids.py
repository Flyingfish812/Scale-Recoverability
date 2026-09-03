"""Supplementary run-id helpers tests (P1-0)."""

from applications.paper_supplementary.run_ids import make_run_id, parse_run_id


def test_make_parse_roundtrip():
    rid = make_run_id("nc", "mlp", "family_03", 20, 0.001, 101)
    assert rid == "supp_nc_mlp_family_03_M20_sigma0p001_seed101"
    p = parse_run_id(rid)
    assert p == {"dataset": "nc", "model": "mlp", "family": "family_03",
                 "M": 20, "sigma": 0.001, "seed": 101}


def test_deterministic_seed_tag():
    rid = make_run_id("nc", "ridge", "family_01", 30, 0.0, "deterministic")
    assert rid.endswith("deterministic")


def test_sigma_zero_tag():
    assert make_run_id("nc", "gappy", "family_02", 10, 0.0, 0).endswith("sigma0_seed000")
