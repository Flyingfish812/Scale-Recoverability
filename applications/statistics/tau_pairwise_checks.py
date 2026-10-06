"""Does the recovery threshold change the ranking of the estimators?

The scale count is quoted at τ=0.05. This module re-ranks the three estimators by their probability of recovering at least three scales at τ=0.03, 0.05 and 0.08, for every sensor count, and records where the ordering differs from the reference threshold. A ranking that survives a 1.7× change of the threshold is evidence that the comparison does not hinge on the choice of τ.

Output
------
artifacts/statistics/tau_pairwise_checks.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.config import get_config  # noqa: E402

MIN_RECOVERED = 3
REFERENCE_TAU = round(float(get_config().tau), 3)


def main() -> int:
    print("== ranking consistency across the recovery threshold")
    source = ROOT / "artifacts" / "statistics" / "threshold_sensitivity.json"
    entries = json.loads(source.read_text(encoding="utf-8"))["results"]

    taus = sorted({round(float(e["tau"]), 3) for e in entries})
    sensors = sorted({e["mask_num"] for e in entries})
    pass_probability = {(e["model"], e["mask_num"], round(float(e["tau"]), 3)): e["P3"]
                        for e in entries}

    ridge_below_mlp = 0
    ridge_below_vcnn = 0
    consistent = 0
    ties = []
    violations = []
    for tau in taus:
        for mask in sensors:
            ridge = pass_probability[("ridge", mask, tau)]
            mlp = pass_probability[("mlp", mask, tau)]
            vcnn = pass_probability[("vcnn", mask, tau)]
            # The closed-form estimator is never better than a learned one; at the coarse thresholds both pass probabilities are zero, hence "not above".
            ridge_below_mlp += int(ridge <= mlp)
            ridge_below_vcnn += int(ridge <= vcnn)

            reference = (pass_probability[("mlp", mask, REFERENCE_TAU)]
                         - pass_probability[("vcnn", mask, REFERENCE_TAU)])
            current = mlp - vcnn
            if current == 0.0:
                ties.append({
                    "sensor_count": mask,
                    "tau": tau,
                    "mlp_minus_vcnn_reference": round(reference, 4),
                    "effect_size": round(abs(reference), 4),
                })
            elif (reference > 0) == (current > 0):
                consistent += 1
            else:
                violations.append({
                    "sensor_count": mask,
                    "tau": tau,
                    "mlp_minus_vcnn_reference": round(reference, 4),
                    "mlp_minus_vcnn_here": round(current, 4),
                    "effect_size": round(abs(current), 4),
                })

    n_checks = len(taus) * len(sensors)
    result = {
        "description": (f"Ranking of the estimators by the probability of recovering "
                        f"at least {MIN_RECOVERED} scales, at tau = {taus}; "
                        f"reference threshold tau = {REFERENCE_TAU}"),
        "taus": taus,
        "sensor_counts": sensors,
        "min_recovered_scales": MIN_RECOVERED,
        "n_checks_per_ranking": n_checks,
        "ridge_below_mlp_pass": ridge_below_mlp,
        "ridge_below_vcnn_pass": ridge_below_vcnn,
        "mlp_vcnn_consistent": consistent,
        "ties": ties,
        "violations": violations,
        "pass_probability": {f"{model}_M{mask}_tau{tau}": value
                             for (model, mask, tau), value in sorted(pass_probability.items())},
    }

    target = ROOT / "artifacts" / "statistics" / "tau_pairwise_checks.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"   ridge not above MLP {ridge_below_mlp}/{n_checks}, "
          f"ridge not above VCNN {ridge_below_vcnn}/{n_checks}")
    print(f"   MLP-VCNN ordering unchanged in {consistent}/{n_checks} checks; "
          f"ties {result['ties']}; violations {result['violations']}")
    print(f"[OK] {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
