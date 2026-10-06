#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figS01_oracle.py — supplementary figure S3: oracle truncation audit (RDB / SST)

Purpose
    Mean per-band truncation error of the oracle band-POD reference as a function of the POD rank, one panel per dataset, with the mean criterion tau drawn as a reference line.
Data source
    artifacts/statistics/truncation_reference_audit.json
Output
    artifacts/figures/figS01_oracle_rdb.pdf
    artifacts/figures/figS01_oracle_sst.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ps

OUT_DIR = _ROOT / "artifacts" / "figures"
SUPP = _ROOT / "artifacts" / "statistics"

BANDS = ps.BANDS
TAU = ps.TAU


def main() -> int:
    ps.apply()

    d = json.loads((SUPP / "truncation_reference_audit.json").read_text())
    cfg = {
        "rdb_h5": ("RDB (radial dam-break)", [16, 32, 64, 128], "figS01_oracle_rdb"),
        "sst_weekly": ("SST (sea surface temperature)", [32, 64, 128, 256, 512, 1024],
                       "figS01_oracle_sst"),
    }
    summaries = {s["dataset"]: s for s in d["summaries"]}
    for ds, (label, ranks, stem) in cfg.items():
        sm = summaries.get(ds)
        if sm is None:
            continue
        table = sm.get("table", {})
        fig, ax = ps.figure(6.6, 3.4, 1, 1)
        for b in BANDS:
            xs, ys = [], []
            for r in ranks:
                e = table.get(str(r), {}).get("bands", {}).get(b, {}).get("mean")
                if e is not None:
                    xs.append(r)
                    ys.append(float(e))
            if xs:
                ax.plot(xs, ys, marker=ps.BAND_MARKERS[b], color=ps.BAND_COLORS[b],
                        label=b, lw=1.3, ms=4.5)
        ax.axhline(TAU, color="0.45", ls="--", lw=0.9,
                   label=f"$\\tau$={TAU} (mean criterion)")
        ax.set_xlabel("POD rank $r$")
        ax.set_ylabel("Mean band truncation error")
        ax.set_yscale("log")
        ax.legend(ncol=3, fontsize=7)
        ps.save(fig, OUT_DIR, stem)
        print(f"  [S1] {stem} done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
