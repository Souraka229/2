"""Baseline reproductible — Phase 0 jour 1 du PLAN_100J."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import config
from cv_group import run_group_cv
from cv_rolling import run_rolling_cv


def _file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rolling = run_rolling_cv()
    group = run_group_cv()

    baseline = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.RANDOM_STATE,
        "train_hash": _file_hash(config.TRAIN_PATH),
        "test_hash": _file_hash(config.TEST_PATH),
        "rolling_cv": rolling,
        "group_cv": group,
    }
    out = config.REPORTS_DIR / "baseline_score.json"
    out.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"\nBaseline enregistrée : {out}")
    print(f"Rolling AP : {rolling['mean_ap']:.6f} ± {rolling['std_ap']:.6f}")
    print(f"Group AP   : {group['mean_ap']:.6f} ± {group['std_ap']:.6f}")


if __name__ == "__main__":
    main()
