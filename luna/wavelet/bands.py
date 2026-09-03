"""
Band name/index conversion utilities.

Provides consistent mapping between band names (A4, W4, ...) and
integer indices, in both coarse-to-fine and fine-to-coarse orderings.
"""

from __future__ import annotations

from luna.core.constants import BANDS_CF, BANDS_FC


def band_name_to_index(band: str, order: str = "coarse_to_fine") -> int:
    """Convert a band name to its 0-based index.

    Args:
        band: Band name (e.g. 'A4', 'W3').
        order: 'coarse_to_fine' → A4=0, W1=4
               'fine_to_coarse' → W1=0, A4=4

    Returns:
        0-based index.
    """
    band_list = BANDS_CF if order == "coarse_to_fine" else BANDS_FC
    try:
        return band_list.index(band)
    except ValueError:
        raise ValueError(f"Unknown band '{band}'. Valid: {band_list}")


def band_index_to_name(idx: int, order: str = "coarse_to_fine") -> str:
    """Convert a 0-based index to its band name."""
    band_list = BANDS_CF if order == "coarse_to_fine" else BANDS_FC
    if not (0 <= idx < len(band_list)):
        raise ValueError(f"Band index {idx} out of range [0, {len(band_list)})")
    return band_list[idx]


def get_band_order(fine_to_coarse: bool = False) -> list[str]:
    """Return the standard band order list."""
    return list(BANDS_FC if fine_to_coarse else BANDS_CF)
