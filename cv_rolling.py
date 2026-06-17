"""CV temporelle rolling — 5 folds, cutoffs 75, 81, 87, 93, 99."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score

import config
from features import get_feature_columns
from run_pipeline import (
    CAT_COLS,
    TE_COLS,
    _prepare_lgbm_df,
    load_data,
    prepare_datasets,
    temporal_split,
)

warnings.filterwarnings("ignore")

ROLLING_CUTOFFS = config.ROLLING_CUTOFFS


def train_lgbm_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    feature_cols: list[str],
) -> tuple[np.ndarray, float]:
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    model = LGBMClassifier(
        objective="binary",
        metric="average_precision",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=neg / max(pos, 1),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    X_tr_fit = _prepare_lgbm_df(X_tr, feature_cols)
    X_va_fit = _prepare_lgbm_df(X_va, feature_cols, ref=X_tr)
    model.fit(
        X_tr_fit,
        y_tr,
        eval_set=[(X_va_fit, y_va)],
        eval_metric="average_precision",
        callbacks=[early_stopping(80, verbose=False)],
        categorical_feature=CAT_COLS,
    )
    pred = model.predict_proba(X_va_fit)[:, 1]
    return pred, float(average_precision_score(y_va, pred))


def run_rolling_cv(train: pd.DataFrame | None = None) -> dict:
    if train is None:
        train, _, _ = load_data()

    scores: list[float] = []
    fold_details: list[dict] = []

    for cutoff in ROLLING_CUTOFFS:
        tr, va = temporal_split(train, cutoff)
        if len(va) < 1000 or tr[config.TARGET_COL].sum() < 10:
            continue
        combined = pd.concat([tr, va], axis=0)
        tr_feat, va_feat, feature_cols = prepare_datasets(
            tr,
            va,
            te_source=tr,
            ref_unsupervised=pd.concat([tr, va], axis=0),
            combined_for_expanding=combined,
            use_oof_te=True,
        )
        _, ap = train_lgbm_fold(
            tr_feat,
            tr[config.TARGET_COL],
            va_feat,
            va[config.TARGET_COL],
            feature_cols,
        )
        scores.append(ap)
        fold_details.append({"cutoff": cutoff, "train_rows": len(tr), "val_rows": len(va), "ap": ap})
        print(f"Fold period<{cutoff}: AP={ap:.6f}  (train={len(tr):,}, val={len(va):,})")

    report = {
        "type": "rolling_temporal",
        "cutoffs": ROLLING_CUTOFFS,
        "folds": fold_details,
        "mean_ap": float(np.mean(scores)) if scores else None,
        "std_ap": float(np.std(scores)) if scores else None,
        "n_folds": len(scores),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "cv_rolling_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nRolling CV: {report['mean_ap']:.6f} ± {report['std_ap']:.6f}  -> {out}")
    return report


if __name__ == "__main__":
    run_rolling_cv()
