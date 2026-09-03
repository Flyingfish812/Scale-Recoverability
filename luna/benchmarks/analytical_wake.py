"""
NC-inspired analytical multiscale wake benchmark (P0-1).

This module provides a fully analytical, ground-truth-controlled multiscale
field that mimics the cylinder-wake structure of the NC dataset (80x160
velocity field, downstream von-Karman vortex street, coarse-to-fine scale
content). It is used to validate the Scale-Recoverability diagnostics
(S_full, S_coh, per-band errors) in a setting where the scale content is
strictly known.

Construction
------------
A streamfunction is built analytically and velocities follow
u = d(psi)/dy, v = -d(psi)/dx (exactly divergence-free):

    psi(x, y) = psi0(x, y) + E(x, y) * sum_{j=2..5} A_j g_j(x, y)
                            + A_1 g_1(x, y)

  * psi0        : global (A4-level) base flow -- uniform stream plus a smooth
                  Gaussian wake deficit that turns on downstream of the
                  cylinder.  This is the physical content of the coarsest
                  wavelet band A4 (like the real NC mean field).
  * E(x, y)     : smooth wake envelope (ramp-on at the cylinder, exponential
                  downstream decay, Gaussian cross-stream profile with
                  downstream widening).  Localizes detail scales to the wake.
  * g_j(x, y)   : L2-normalized superposition of Fourier modes whose
                  wavenumbers lie in the octave of wavelet band j, calibrated
                  empirically against db2 / level-4 / periodization on the
                  80x160 grid (see WAVENUM).

Controlled cases (each "reconstruction" removes a known scale)
--------------------------------------------------------------
  A : full field (reference, error 0)
  B : finest scale removed (W1 component)            -> expect S_full = 4
  C : two finest scales removed (W1 + W2)            -> expect S_full = 3
  D : intermediate scale destroyed (W3)              -> expect S_full = 2
  E : matched-GER pair with different failed bands   -> expect same GER,
                                                       different S_full

Amplitudes A_j are set so that the band energy hierarchy matches the real NC
data (A4 ~95%, W4 ~2.5%, W3 ~1.5%, W2 ~0.3%, W1 ~0.1%), which is exactly the
regime in which the metric is designed to add information beyond GER.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import erf

from luna.core.constants import BANDS_CF, DEFAULT_LEVEL, DEFAULT_MODE, DEFAULT_WAVELET
from luna.wavelet.metrics import band_errors_all, compute_S_full, compute_S_coh, rel_l2
from luna.wavelet.transform import decompose_field_2d


# ══════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════
@dataclass
class WakeParams:
    """Parameters of the analytical wake field.

    All length units are grid points.  The grid is H (y) x W (x) with the
    cylinder on the left and flow to the right (matching the NC layout).
    """
    H: int = 80
    W: int = 160
    x0: float = 28.0          # cylinder x position
    y0: float = 40.0          # wake centerline y position

    # base flow
    u0: float = 0.60          # uniform stream speed
    delta_u: float = 0.30     # wake deficit strength
    sigma_d: float = 9.0      # wake deficit Gaussian width
    delta_ramp: float = 22.0  # deficit ramp smoothness

    # envelope
    delta_e: float = 40.0     # wake-start ramp smoothness
    l_dec: float = 200.0      # downstream decay length
    sigma_w0: float = 12.0    # initial wake width
    s_w: float = 0.04         # wake widening rate

    # amplitudes (streamfunction units), calibrated to NC band hierarchy
    amplitudes: dict[int, float] = field(default_factory=lambda: {
        1: 0.6, 2: 110.0, 3: 12.0, 4: 4.0, 5: 5.0,
    })


# Wavenumber octaves per band, empirically calibrated so that each mode's
# dominant wavelet band is its target band and the *coarse-direction* leakage
# (into the next coarser band) is minimized.  Bands: 1=A4 (coarsest) ... 5=W1.
WAVENUM: dict[int, list[tuple[int, int]]] = {
    1: [(1, 1), (2, 1), (3, 2)],   # A4  (global, no envelope)
    2: [(6, 4), (7, 4)],           # W4
    3: [(14, 7)],                  # W3  (single clean mode; (12,6) leaks to W4)
    4: [(28, 14), (24, 12)],       # W2
    5: [(48, 32), (56, 28)],       # W1
}


# ══════════════════════════════════════════════════════════════════════
# Primitive functions
# ══════════════════════════════════════════════════════════════════════
def _ramp(x: np.ndarray, x0: float, delta: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((x - x0) / delta))


def _grid(params: WakeParams) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(params.W, dtype=np.float64)
    y = np.arange(params.H, dtype=np.float64)
    return x, y


def base_streamfunction(x: np.ndarray, y: np.ndarray, p: WakeParams) -> np.ndarray:
    """psi0: uniform stream + smooth Gaussian wake deficit (A4 content)."""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = _ramp(xx, p.x0, p.delta_ramp)
    deficit = (
        p.delta_u * r * (np.sqrt(2 * np.pi) * p.sigma_d / 2.0)
        * erf((yy - p.y0) / (np.sqrt(2.0) * p.sigma_d))
    )
    return p.u0 * yy - deficit


def wake_envelope(x: np.ndarray, y: np.ndarray, p: WakeParams) -> np.ndarray:
    """E(x,y): smooth wake-localization envelope for the detail bands."""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = np.maximum(xx - p.x0, 0.0)
    R = _ramp(xx, p.x0, p.delta_e)
    D = np.exp(-r / p.l_dec)
    sig = p.sigma_w0 + p.s_w * r
    Y = np.exp(-(yy - p.y0) ** 2 / (2.0 * sig ** 2))
    return R * D * Y


def scale_component(
    j: int,
    x: np.ndarray,
    y: np.ndarray,
    params: WakeParams | None = None,
    seed: int = 0,
) -> np.ndarray:
    """g_j: L2-normalized Fourier-mode superposition in band-j octave."""
    p = params or WakeParams()
    rng = np.random.RandomState(seed * 100 + j)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = np.zeros((p.W, p.H))
    for kx, ky in WAVENUM[j]:
        phi = rng.uniform(0, 2 * np.pi)
        chi = rng.uniform(0, 2 * np.pi)
        field += np.sin(2 * np.pi * kx * (xx - p.x0) / p.W + phi) * np.sin(
            2 * np.pi * ky * (yy - p.y0) / p.H + chi
        )
    return field / (np.linalg.norm(field) + 1e-12)


# ══════════════════════════════════════════════════════════════════════
# Field construction
# ══════════════════════════════════════════════════════════════════════
def streamfunction(
    x: np.ndarray,
    y: np.ndarray,
    params: WakeParams | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Analytical streamfunction and its scale components.

    Returns:
        psi: (W, H) streamfunction field.
        psi_j: {j: (W, H)} streamfunction contribution of scale j
               (psi_j = E * A_j g_j for j>=2, A_1 g_1 for j=1; psi0 is not
               attributed to a component).
    """
    p = params or WakeParams()
    psi = base_streamfunction(x, y, p)
    E = wake_envelope(x, y, p)
    psi_j: dict[int, np.ndarray] = {}
    for j in range(1, 6):
        if j == 1:
            psi_j[j] = p.amplitudes[1] * scale_component(1, x, y, p, seed)
        else:
            psi_j[j] = p.amplitudes[j] * E * scale_component(j, x, y, p, seed)
        psi = psi + psi_j[j]
    return psi, psi_j


def velocity(
    x: np.ndarray,
    y: np.ndarray,
    params: WakeParams | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Divergence-free velocity pair (u, v) from the analytical streamfunction."""
    psi, _ = streamfunction(x, y, params, seed)
    u = np.gradient(psi, axis=1)    # d/dy
    v = -np.gradient(psi, axis=0)   # -d/dx
    return u, v


def scale_u_components(
    x: np.ndarray,
    y: np.ndarray,
    params: WakeParams | None = None,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """u^{(j)} = d(psi_j)/dy : u-field content of each scale j."""
    p = params or WakeParams()
    _, psi_j = streamfunction(x, y, p, seed)
    return {j: np.gradient(psi_j[j], axis=1) for j in range(1, 6)}


def snapshot(
    x: np.ndarray,
    y: np.ndarray,
    params: WakeParams | None = None,
    seed: int = 0,
) -> np.ndarray:
    """u-field of a single analytical snapshot."""
    p = params or WakeParams()
    u, _ = velocity(x, y, p, seed)
    return u


def generate_ensemble(
    n: int,
    params: WakeParams | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Generate n analytical snapshots (u fields) with random phases.

    Returns:
        fields: (n, H, W) u-fields.
        (x, y): grid arrays.
    """
    p = params or WakeParams()
    x, y = _grid(p)
    fields = np.stack([snapshot(x, y, p, seed_offset + i) for i in range(n)])
    return fields, (x, y)


# ══════════════════════════════════════════════════════════════════════
# Controlled cases
# ══════════════════════════════════════════════════════════════════════
def controlled_reconstructions(
    u: np.ndarray,
    u_j: dict[int, np.ndarray],
    alpha_e: float | None = None,
) -> dict[str, np.ndarray]:
    """Build the controlled-case reconstructions.

    Args:
        u: target u-field.
        u_j: {j: u^{(j)}} scale components of the target.
        alpha_e: partial-deletion factor for Case E (auto-solved if None).

    Returns:
        {case_name: reconstructed u-field}
    """
    cases = {
        "A_full": u.copy(),
        "B_del_W1": u - u_j[5],
        "C_del_W1W2": u - u_j[5] - u_j[4],
        "D_del_W3": u - u_j[3],
    }
    if alpha_e is None:
        alpha_e = _solve_alpha_e(u, u_j)
    cases["E1_del_W1_only"] = u - u_j[5]
    cases["E2_partial_W3"] = u - alpha_e * u_j[3]
    return cases


def _solve_alpha_e(
    u: np.ndarray,
    u_j: dict[int, np.ndarray],
    tol: float = 1e-6,
) -> float:
    """Find alpha so GER(u, u - alpha*u3) == GER(u, u - u5)."""
    ger_e1 = rel_l2(u - u_j[5], u)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        ger = rel_l2(u - mid * u_j[3], u)
        if ger < ger_e1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ══════════════════════════════════════════════════════════════════════
# Metric computation for a single snapshot
# ══════════════════════════════════════════════════════════════════════
def case_metrics(
    target: np.ndarray,
    u_j: dict[int, np.ndarray],
    band_pod: dict[str, dict[str, np.ndarray]] | None = None,
    tau: float = 0.05,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    mode: str = DEFAULT_MODE,
) -> dict[str, dict]:
    """Compute GER / E_direct / S_full / S_coh for every controlled case.

    Returns:
        {case_name: {'GER': float, 'E_direct': {band: float}, 'S_full': int,
                     'S_coh': int | None}}
    """
    cases = controlled_reconstructions(target, u_j)
    out: dict[str, dict] = {}
    for name, uhat in cases.items():
        rec = {
            "GER": rel_l2(uhat, target),
            "E_direct": band_errors_all(target, uhat, wavelet, level, mode),
            "S_full": compute_S_full(target, uhat, tau, wavelet, level, mode),
            "S_coh": None,
        }
        if band_pod is not None:
            rec["S_coh"] = compute_S_coh(
                target, uhat, band_pod, tau, wavelet, level, mode
            )
        out[name] = rec
    return out
