"""Check that the raw arrays the paper is computed from are in place.

Raw fields are not shipped with the repository: they have to be fetched from the public sources listed in ``scripts/download_data.sh`` and placed under ``data/``. This step is the gate in front of the pipeline. It verifies that each array is present and has the shape the paper's experiments assume, and it prints the fetch instructions of the ones that are missing or unusable, so that a failure says what to download instead of surfacing later as a shape error inside the POD decomposition or the statistics layer.

Prerequisites
    none (this is the first step of the pipeline)

Verified arrays (sources, layouts and download steps: scripts/download_data.sh)
    data/cylinder2d_q1.npy   (1501, 80, 160, 2)   NC, primary dataset
    data/rdb_h5.npy          (5050, 128, 128, 1)  RDB, rank-adequacy check
    data/sst_weekly.npy      (1914, 180, 360, 1)  SST, rank-adequacy check

Paths come from the dataset registry (``luna.data.registry``), which is the single source of truth for where a dataset lives. The expected shapes are stated here, because they describe the experiments of the paper rather than the filesystem: they are the shapes of the arrays the published artifacts were produced from.

Outputs
    none (the step reads data/ and reports)

Pipeline
    previous  scripts/download_data.sh (fetch the raw sources, manual step)
    this step verifies the raw arrays under data/
    next      applications/pipelines/02_build_pod_bases.py

Usage
    python applications/pipelines/01_prepare_data.py
    python applications/pipelines/01_prepare_data.py --datasets nc
    python applications/pipelines/01_prepare_data.py --check

``--check`` runs the same verification and only drops the surrounding notes, which is convenient when the status of the arrays is read from a script.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luna.data.io import load_npy  # noqa: E402
from luna.data.registry import get_dataset  # noqa: E402

# : Datasets of the paper, in the order of scripts/download_data.sh.
DEFAULT_DATASETS = ("nc", "rdb_h5", "sst_weekly")

# : Shape of the raw array of each dataset; see scripts/download_data.sh for the : public source and the cropping or packing steps that produce it.
EXPECTED_SHAPES = {
    "nc": (1501, 80, 160, 2),
    "rdb_h5": (5050, 128, 128, 1),
    "sst_weekly": (1914, 180, 360, 1),
}

# : Tag of each dataset in the section headings of scripts/download_data.sh.
SOURCE_TAG = {"nc": "NC", "rdb_h5": "RDB", "sst_weekly": "SST"}

# : The fetch instructions that this step prints when an array is unusable.
DOWNLOAD_SCRIPT = ROOT / "scripts" / "download_data.sh"


def relative(path: Path) -> str:
    """Path as it is reported to the user: relative to the repository when it is
    inside it, absolute otherwise."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def download_instructions(dataset: str) -> str:
    """The section of scripts/download_data.sh that explains how to obtain one
    dataset; the whole script when no section matches."""
    if not DOWNLOAD_SCRIPT.exists():
        return f"{DOWNLOAD_SCRIPT.name} is not part of this checkout"
    text = DOWNLOAD_SCRIPT.read_text(encoding="utf-8")
    tag = SOURCE_TAG.get(dataset)
    if tag is None:
        return text
    block: list[str] = []
    inside = False
    for line in text.splitlines():
        if re.match(r"^\d+\)\s", line):
            inside = re.search(rf"\b{tag}\b", line) is not None
        if inside:
            block.append(line)
    return "\n".join(block).strip() or text


def describe(dataset: str) -> tuple[bool, str]:
    """Status of one raw array: whether it is present with the expected shape,
    and the line that reports it."""
    expected = EXPECTED_SHAPES[dataset]
    path = get_dataset(dataset).data_array
    if not path.exists():
        return False, f"{relative(path)} is missing (expected {expected})"
    array = load_npy(path, mmap=True)
    shape = tuple(int(v) for v in array.shape)
    if shape != expected:
        return False, f"{relative(path)} has shape {shape}, expected {expected}"
    return True, f"{relative(path)}  {shape}  {array.dtype}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"datasets to verify (default: {' '.join(DEFAULT_DATASETS)})")
    parser.add_argument("--check", action="store_true",
                        help="report the status of the arrays only")
    args = parser.parse_args()

    datasets = args.datasets or list(DEFAULT_DATASETS)
    unknown = [name for name in datasets if name not in EXPECTED_SHAPES]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; "
                         f"known: {', '.join(DEFAULT_DATASETS)}")

    if not args.check:
        print("== checking the raw arrays under data/ "
              "(sources and layouts: scripts/download_data.sh)")

    failed: list[str] = []
    for dataset in datasets:
        ok, message = describe(dataset)
        print(f"   [{'ok' if ok else '!!'}] {dataset:11s} {message}")
        if not ok:
            failed.append(dataset)

    if failed:
        for dataset in failed:
            print(f"\n--- {dataset}: how to obtain it "
                  f"(from {relative(DOWNLOAD_SCRIPT)}) " + "-" * 20)
            print(download_instructions(dataset))
        print(f"\n[failed] {len(failed)}/{len(datasets)} arrays are not usable; "
              "place them under data/ and run this step again")
        return 1

    if not args.check:
        print(f"\n[ok] {len(datasets)}/{len(datasets)} arrays present with the expected shapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
