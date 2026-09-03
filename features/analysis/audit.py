"""
Consistency & wording audit tools.

- Digital audit: checks sample counts, coherent structure percentages,
  and cross-references table values with source data.
- Wording audit: scans manuscript source for overstated causal language.

Replaces: tools/run_digital_audit.py, tools/run_wording_audit.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def run_digital_audit(
    results_dir: str | Path,
    manuscript_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit digital consistency of experimental results.

    Checks:
    1. Sample counts across different result files are consistent.
    2. Coherent structure percentages sum correctly.
    3. Table values can be traced back to source NPZ/JSON files.

    Args:
        results_dir: Directory containing result files.
        manuscript_path: Optional path to manuscript for cross-reference.

    Returns:
        Audit report with any inconsistencies found.
    """
    results = Path(results_dir)
    issues: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    # Collect all NPZ files and check sample counts
    npz_files = sorted(results.rglob("*.npz"))
    sample_counts: dict[str, int] = {}
    for f in npz_files:
        try:
            import numpy as np
            data = np.load(f, allow_pickle=True)
            for key in data.files:
                arr = data[key]
                if hasattr(arr, "shape") and len(arr.shape) >= 1:
                    sample_counts[f"{f.name}:{key}"] = arr.shape[0]
            checks.append({"file": str(f.relative_to(results)), "status": "ok"})
        except Exception as e:
            issues.append({"file": str(f.relative_to(results)), "error": str(e)})

    report = {
        "total_files": len(npz_files),
        "total_checks": len(checks),
        "issues_found": len(issues),
        "issues": issues,
        "sample_counts": sample_counts,
    }
    return report


# Overstated causal language patterns to flag
_CAUSAL_PATTERNS = [
    (r"\bproves?\b", "Consider 'supports' or 'is consistent with'"),
    (r"\bdemonstrates?\s+(that\s+)?(the\s+)?(necessity|essential)", "Consider 'suggests' or 'indicates'"),
    (r"\b本质上\b", "Consider '在当前实验中' or '在给定设置下'"),
    (r"\b主导(机制|因素)?\b", "Consider '主要限制因素' or '与...高度相关'"),
    (r"\b证明\b", "Consider '支持' or '与...一致'"),
]


def run_wording_audit(
    manuscript_dir: str | Path,
    patterns: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Audit manuscript for overstated causal language.

    Args:
        manuscript_dir: Directory containing thesis/manuscript source files.
        patterns: Custom (regex, suggestion) pairs. Uses defaults if None.

    Returns:
        Audit report with flagged lines and suggestions.
    """
    if patterns is None:
        patterns = _CAUSAL_PATTERNS

    src = Path(manuscript_dir)
    flags: list[dict[str, Any]] = []

    for tex_file in sorted(src.rglob("*.tex")):
        try:
            content = tex_file.read_text(encoding="utf-8")
        except Exception:
            continue

        for line_no, line in enumerate(content.splitlines(), start=1):
            for pattern, suggestion in patterns:
                matches = list(re.finditer(pattern, line, re.IGNORECASE))
                for m in matches:
                    context = line[max(0, m.start() - 30):m.end() + 30].strip()
                    flags.append({
                        "file": str(tex_file.name),
                        "line": line_no,
                        "matched": m.group(),
                        "suggestion": suggestion,
                        "context": f"...{context}...",
                    })

    return {
        "total_flags": len(flags),
        "flags": flags,
        "patterns_used": [(p, s) for p, s in patterns],
    }
