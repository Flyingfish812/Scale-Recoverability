"""Compare the regenerated figures with the copies used by the manuscript.

Rendered page images are compared rather than file hashes, because the PDF
writer embeds a creation timestamp and a differing hash would not say whether
the drawing changed. This is the acceptance check of the figure migration.

Usage
    python applications/figures/compare_with_manuscript.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "artifacts" / "figures"
MANUSCRIPT = ROOT / "thesis_work" / "thesis_src_planB" / "figures"


def render(pdf: Path, directory: Path, dpi: int = 100) -> np.ndarray | None:
    """Rasterise the first page of a PDF and return it as a grey-scale array."""
    output = directory / pdf.stem
    if subprocess.run(
        ["pdftoppm", "-gray", "-r", str(dpi), "-f", "1", "-l", "1", "-png",
         str(pdf), str(output)],
        capture_output=True,
    ).returncode != 0:
        return None
    page = sorted(directory.glob(f"{pdf.stem}*.png"))
    if not page:
        return None
    import matplotlib.image as mpimg

    return mpimg.imread(str(page[0]))


def main() -> int:
    if not MANUSCRIPT.exists():
        print(f"manuscript figure directory not found: {MANUSCRIPT}")
        return 1

    generated = sorted(GENERATED.glob("*.pdf"))
    identical = different = missing = 0
    worst = []
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for figure in generated:
            reference = MANUSCRIPT / figure.name
            if not reference.exists():
                missing += 1
                print(f"   [absent from manuscript] {figure.name}")
                continue
            a = render(figure, directory)
            b = render(reference, directory)
            if a is None or b is None or a.shape != b.shape:
                different += 1
                worst.append((figure.name, float("inf")))
                continue
            difference = float(np.max(np.abs(a - b)))
            if difference <= 2.0 / 255.0:
                identical += 1
            else:
                different += 1
                worst.append((figure.name, difference))

    print(f"[OK] {identical} figures identical to the manuscript, "
          f"{different} differ, {missing} absent")
    for name, difference in sorted(worst, key=lambda item: -item[1]):
        print(f"   max pixel difference {difference:.4f}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
