#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig01_method_framework.py — supplementary figure S1: workflow schematic.

Layout:
  - canvas 6.0 x 3.6 in, main chain and reference branch split 62% / 30%, symmetric left and right margins (4%);
  - the POD truncation reference branch sits in the same horizontal band as the reconstruction box of the main chain;
  - all boxes square, line width 0.9 pt, uniform arrows; white background, black and grey plus one accent colour;
  - main chain: Sparse measurements -> Reconstruction models -> Reconstructed field -> Wavelet scale evaluation;
  - inside Reconstruction: POD-based (Ridge / Gappy POD / MLP) and End-to-end (VCNN) stacked in two groups.

Workflow: build SVG -> (optionally tweak in Inkscape) -> export vector PDF with cairosvg.
Output: fig01_method_framework.{svg,pdf} under artifacts/figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 576, 346          # 6.0 x 3.6 in @96dpi
INK = "#1A1A1A"
GRAY = "#5A5A5A"
LGRAY = "#8F8F8F"
BOX_EC = "#333333"
BOX_FC = "#FFFFFF"
POD_FC = "#F1F3F5"
E2E_FC = "#EEF2F7"
LW = 0.9
FONT = "Liberation Serif, Nimbus Roman, DejaVu Serif, serif"


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, text="", fontsize=9, fill=BOX_FC, ec=BOX_EC, lw=LW,
        dashed=False, tc=INK, weight="normal", sub=None, sub_fs=8,
        anchor="middle"):
    dash = ' stroke-dasharray="4,3"' if dashed else ""
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
         f'fill="{fill}" stroke="{ec}" stroke-width="{lw}"{dash}/>']
    if text:
        cx = x + w / 2 if anchor == "middle" else x
        ha = "middle" if anchor == "middle" else "start"
        cy = y + h / 2
        if sub is None:
            s.append(
                f'<text x="{cx}" y="{cy}" text-anchor="{ha}" '
                f'dominant-baseline="central" font-family="{FONT}" '
                f'font-size="{fontsize}" fill="{tc}" '
                f'font-weight="{weight}">{esc(text)}</text>')
        else:
            s.append(
                f'<text x="{cx}" y="{cy - fontsize * 0.5}" text-anchor="{ha}" '
                f'dominant-baseline="central" font-family="{FONT}" '
                f'font-size="{fontsize}" fill="{tc}" '
                f'font-weight="{weight}">{esc(text)}</text>')
            for i, line in enumerate(sub.split("\n")):
                s.append(
                    f'<text x="{cx}" y="{cy + fontsize * 0.5 + i * (sub_fs + 2)}" '
                    f'text-anchor="{ha}" dominant-baseline="central" '
                    f'font-family="{FONT}" font-size="{sub_fs}" '
                    f'fill="{GRAY}">{esc(line)}</text>')
    return "\n".join(s)


_marker_n = [0]


def arrow(x1, y1, x2, y2, color=INK, lw=LW, dashed=False):
    _marker_n[0] += 1
    mid = f"ar{_marker_n[0]}"
    dash = ' stroke-dasharray="4,3"' if dashed else ""
    return (
        f'<defs><marker id="{mid}" markerWidth="7" markerHeight="7" '
        f'refX="3.5" refY="3.5" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L7,3.5 L0,7 z" fill="{color}"/></marker></defs>'
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{lw}"{dash} marker-end="url(#{mid})"/>'
    )


def htext(x, y, text, fontsize=9, color=INK, weight="normal", anchor="middle"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'dominant-baseline="middle" font-family="{FONT}" '
            f'font-size="{fontsize}" fill="{color}" '
            f'font-weight="{weight}">{esc(text)}</text>')


def build_svg() -> str:
    p = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}">')
    p.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

    # ══ main chain (62% of width) ═══════════════════════════════════
    CX, BW = 178, 250               # main chain centre x, box width
    X0 = CX - BW / 2                # box left edge = 53
    # (1) input: y 8..52
    p.append(box(X0, 8, BW, 44, "Sparse measurements", fontsize=11.4,
                 weight="bold", sub="M sensor locations, noise level σ", sub_fs=9.6))
    p.append(arrow(CX, 52, CX, 64))

    # (2) reconstruction: y 64..216
    RX, RY, RW, RH = X0, 64, BW, 152
    p.append(box(RX, RY, RW, RH, "", fill=BOX_FC, ec=BOX_EC, lw=LW))
    p.append(htext(RX + 12, RY + 12, "Reconstruction models",
                   fontsize=11.4, weight="bold", anchor="start"))

    # POD-based subgroup: y 84..140
    PX, PY, PW, PH = X0 + 12, 84, BW - 24, 56
    p.append(box(PX, PY, PW, PH, "", fill=POD_FC, ec=BOX_EC, lw=0.8))
    p.append(htext(PX + 10, PY + 10, "POD-based", fontsize=9.5,
                   weight="bold", anchor="start"))
    est_w, est_h, est_y = 64, 24, PY + 16
    ests = ["Ridge", "Gappy POD", "MLP"]
    for i, name in enumerate(ests):
        ex = PX + 10 + i * (est_w + 8)
        p.append(box(ex, est_y, est_w, est_h, name, fontsize=10.2,
                     fill="#FFFFFF", ec=BOX_EC, lw=0.7))
    # output row
    p.append(box(PX + 10, PY + 42, PW - 20, 12, "", fill="#FFFFFF",
                 ec=LGRAY, lw=0.6))
    p.append(htext(PX + 10 + (PW - 20) / 2, PY + 48,
                   "predict rank-r POD coefficients  â",
                   fontsize=9.0, color=GRAY))

    # End-to-end subgroup: y 146..204
    EX, EY, EW, EH = X0 + 12, 146, BW - 24, 58
    p.append(box(EX, EY, EW, EH, "", fill=E2E_FC, ec=BOX_EC, lw=0.8))
    p.append(htext(EX + 10, EY + 10, "End-to-end", fontsize=10.2,
                   weight="bold", anchor="start"))
    p.append(box(EX + 10, EY + 16, 62, 22, "VCNN", fontsize=10.2,
                 fill="#FFFFFF", ec=BOX_EC, lw=0.7))
    p.append(htext(EX + 10 + 62 + 8, EY + 27,
                   "learned field model", fontsize=9.0, color=GRAY,
                   anchor="start"))
    p.append(box(EX + 10, EY + 42, EW - 20, 12, "", fill="#FFFFFF",
                 ec=LGRAY, lw=0.6))
    p.append(htext(EX + 10 + (EW - 20) / 2, EY + 48,
                   "direct field  û", fontsize=8.6, color=GRAY))

    p.append(arrow(CX, 216, CX, 228))

    # (3) field: y 228..268
    p.append(box(X0, 228, BW, 40, "Reconstructed field", fontsize=11.4,
                 weight="bold", sub="û(x, t)", sub_fs=9.6))
    p.append(arrow(CX, 268, CX, 280))

    # (4) evaluation: y 280..324
    p.append(box(X0, 280, BW, 44, "Wavelet scale evaluation", fontsize=11.4,
                 weight="bold", sub="E_direct(b)   ·   S_full   ·   S_coh",
                 sub_fs=9.6))
    p.append(htext(CX, 338, "per-band errors + scale-recoverability indices",
                   fontsize=9.2, color=GRAY))

    # ══ POD truncation reference branch (right side, level with the chain) ══
    OX = X0 + BW + 18               # next to the main chain
    OW = W - OX - 18                # right-aligned to the 18 px canvas margin
    OY, OH = 104, 76                # branch box y and height
    p.append(box(OX, OY, OW, OH, "POD truncation reference", fontsize=11.2,
                 weight="bold", sub="exact rank-r coefficients", sub_fs=9.4,
                 fill=POD_FC, ec=LGRAY, lw=0.8, dashed=True, tc=GRAY))
    # reference -> evaluation: L-shaped dashed link with a short label
    lx = OX + OW / 2
    p.append(f'<line x1="{lx}" y1="{OY + OH}" x2="{lx}" y2="300" '
             f'stroke="{LGRAY}" stroke-width="0.8" stroke-dasharray="4,3"/>')
    p.append(arrow(lx, 300, X0 + BW + 3, 300, color=LGRAY, lw=0.8, dashed=True))
    p.append(htext(X0 + BW + 4, 290, "band-wise error baseline",
                   fontsize=9.4, color=GRAY, anchor="start"))
    # reference -> reconstruction (horizontal dashed link)
    p.append(arrow(OX, OY + 22, X0 + BW + 2, OY + 22, color=LGRAY, lw=0.8,
                   dashed=True))

    p.append("</svg>")
    return "\n".join(p)


def main() -> int:
    svg = build_svg()
    svg_path = OUT_DIR / "fig01_method_framework.svg"
    svg_path.write_text(svg, encoding="utf-8")
    pdf_path = OUT_DIR / "fig01_method_framework.pdf"
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(pdf_path))
    print(f"[ok] {svg_path}")
    print(f"[ok] {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
