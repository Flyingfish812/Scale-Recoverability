"""The sigma label means the same physical perturbation for every estimator.

Each observation pipeline adds Gaussian noise whose standard deviation in field
units equals the requested sigma, with the same standard deviation for both
velocity components: the MLP observation dataset, the VCNN observation builder
and the closed-form Ridge / Gappy test-noise helper. Forward reproductions of
the stored runs show these are the conventions of the recorded experiments (noise
audit, 2026-10-10); this test pins them so the pipelines cannot drift apart again.
"""

from __future__ import annotations

import numpy as np
import pytest

from features.sensors.observations import build_nearest_seed_index, build_observation_feature
from features.training.pod_sweep import PODObservationDataset, add_physical_noise

SIGMA = 0.01
_TOLERANCE = 0.25  # relative tolerance of the measured standard deviation


def _fields(n: int, h: int = 6, w: int = 6, seed: int = 3) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (1.0 + 0.2 * rng.randn(n, h, w, 2)).astype(np.float32)


def _mask() -> np.ndarray:
    mask = np.zeros((6, 6), dtype=bool)
    for row, col in [(1, 1), (2, 3), (4, 4), (0, 5), (5, 1)]:
        mask[row, col] = True
    return mask


def _channel_stds(delta: np.ndarray) -> list[float]:
    """Per-component standard deviation of a flat [u0, v0, u1, v1, ...] delta."""
    assert delta.shape[0] % 2 == 0
    return [float(delta[0::2].std()), float(delta[1::2].std())]


def test_closed_form_helper_adds_sigma_in_field_units():
    fields = _fields(200, 4, 4)
    noisy = add_physical_noise(fields, SIGMA)
    assert noisy.shape == fields.shape
    delta = (noisy - fields).reshape(-1)
    for std in _channel_stds(delta):
        assert abs(std - SIGMA) < _TOLERANCE * SIGMA


def test_helper_with_zero_sigma_is_exact():
    fields = _fields(5)
    assert np.array_equal(add_physical_noise(fields, 0.0), fields.astype(np.float64))


def test_mlp_dataset_noise_has_sigma_in_field_units():
    fields = _fields(60)
    mask = _mask()
    points = np.argwhere(mask)
    coeff = np.zeros((fields.shape[0], 4), dtype=np.float32)
    dataset = PODObservationDataset(fields, mask, coeff, noise_sigma=SIGMA, normalize=True, seed=0)

    channel_index = np.tile(np.arange(2), len(points))
    deltas = []
    for i in range(fields.shape[0]):
        observation, _ = dataset[i]
        observation = observation.numpy()
        noisy = observation * dataset.chan_std[channel_index] + dataset.chan_mean[channel_index]
        clean = fields[i][points[:, 0], points[:, 1], :].reshape(-1)
        deltas.append(noisy - clean)
    for std in _channel_stds(np.concatenate(deltas)):
        assert abs(std - SIGMA) < _TOLERANCE * SIGMA


def test_vcnn_feature_noise_has_sigma_in_field_units():
    fields = _fields(60)
    mask = _mask()
    points = np.argwhere(mask)
    nearest = build_nearest_seed_index(mask)
    mean_c = np.zeros(2, dtype=np.float32)
    std_c = np.ones(2, dtype=np.float32)

    deltas = []
    for i in range(fields.shape[0]):
        feature, _, _ = build_observation_feature(
            fields[i], mask_hw=mask, nearest_index_hw=nearest, noise_sigma=SIGMA,
            representation="voronoi", include_mask_channel=True,
            norm_mean_c=mean_c, norm_std_c=std_c, noise_seed=i,
        )
        flat = np.stack([feature[0][mask], feature[1][mask]], axis=1).reshape(-1)
        clean = fields[i][points[:, 0], points[:, 1], :].reshape(-1)
        deltas.append(flat - clean)
    for std in _channel_stds(np.concatenate(deltas)):
        assert abs(std - SIGMA) < _TOLERANCE * SIGMA


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
