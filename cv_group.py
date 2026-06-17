"""CV group-aware K=5 sur origin_account."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

import config
from run_pipeline import (
    CAT_COLS,
    _prepare_lgbm_df,
    load_data,
    prepare_datasets,
)

warnings.filterwarnings("ignore")

N_SPLITS = 5


def run_group_cv(train: pd.DataFrame | None = None, n_splits: int = N_SPLITS) -> dict:
    if train is None:
        train, _, _ = load_data()

    groups = train["origin_account"]
    y = train[config.TARGET_COL]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    scores: list[float] = []
    fold_details: list[dict] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(train, y, groups=groups)):
        tr = train.iloc[tr_idx]
        va = train.iloc[va_idx]
        combined = pd.concat([tr, va], axis=0)
        tr_feat, va_feat, feature_cols = prepare_datasets(
            tr,
            va,
            te_source=tr,
            ref_unsupervised=combined,
            combined_for_expanding=combined,
            use_oof_te=True,
        )
        pos = tr[config.TARGET_COL].sum()
        neg = len(tr) - pos
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
        X_tr = _prepare_lgbm_df(tr_feat, feature_cols)
        X_va = _prepare_lgbm_df(va_feat, feature_cols, ref=tr_feat)
        model.fit(
            X_tr,
            tr[config.TARGET_COL],
            eval_set=[(X_va, va[config.TARGET_COL])],
            eval_metric="average_precision",
            callbacks=[early_stopping(80, verbose=False)],
            categorical_feature=CAT_COLS,
        )
        pred = model.predict_proba(X_va)[:, 1]
        ap = float(average_precision_score(va[config.TARGET_COL], pred))
        scores.append(ap)
        fold_details.append({"fold": fold_idx, "train_rows": len(tr), "val_rows": len(va), "ap": ap})
        print(f"Group fold {fold_idx}: AP={ap:.6f}")

    report = {
        "type": "group_aware",
        "n_splits": n_splits,
        "folds": fold_details,
        "mean_ap": float(np.mean(scores)),
        "std_ap": float(np.std(scores)),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "cv_group_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nGroup CV: {report['mean_ap']:.6f} ± {report['std_ap']:.6f}  -> {out}")
    return report


if __name__ == "__main__":
    run_group_cv()
