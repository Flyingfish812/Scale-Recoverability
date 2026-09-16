"""Copy the generated figures into the manuscript tree.

The figure scripts write into ``artifacts/figures`` and the repository contains
no generated output. The manuscript keeps its own copy of the PDFs next to the
LaTeX sources, because a submission has to be self-contained; this module is the
explicit step that updates that copy.

The destination defaults to the manuscript tree used for the submission and can
be pointed elsewhere with ``--destination``.

Usage
    python applications/figures/publish_figures.py
    python applications/figures/publish_figures.py --check
    python applications/figures/publish_figures.py --destination /path/to/figures
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts" / "figures"
DEFAULT_DESTINATION = ROOT / "thesis_work" / "manuscript_src" / "figures"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default=str(DEFAULT_DESTINATION))
    parser.add_argument("--check", action="store_true",
                        help="report the differences without copying")
    args = parser.parse_args()

    destination = Path(args.destination)
    if not SOURCE.exists():
        raise SystemExit(f"no figures generated yet: {SOURCE}")
    if not destination.exists():
        raise SystemExit(f"manuscript figure directory not found: {destination}")

    published = unchanged = 0
    for figure in sorted(SOURCE.glob("*.pdf")):
        target = destination / figure.name
        if target.exists() and digest(target) == digest(figure):
            unchanged += 1
            continue
        state = "update" if target.exists() else "new"
        print(f"   [{state}] {figure.name}")
        if not args.check:
            shutil.copy2(figure, target)
        published += 1

    verb = "would update" if args.check else "updated"
    print(f"[OK] {verb} {published} figures, {unchanged} already current "
          f"-> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
