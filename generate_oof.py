"""Genere les predictions OOF des base models pour le stacking.

Stratification : StratifiedGroupKFold sur origin_account → preserve la coherence
des comptes (pas le meme compte en train et val d'un meme fold) et la balance
fraude/non-fraude.

Sorties :
  - reports/oof_predictions.csv : id, target, lgbm, catboost, xgboost
  - outputs/test_base_preds.csv : id, lgbm, catboost, xgboost (entraine sur tout le train)

Le meta-modele (stacking.py) peut ensuite etre entraine sur oof_predictions.csv.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

import config
from run_pipeline import (
    CAT_COLS,
    TE_COLS,
    predict_lightgbm,
    predict_xgboost,
    prepare_datasets,
    train_catboost,
    train_lightgbm,
    train_xgboost,
)


N_SPLITS = 5


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(config.TRAIN_PATH).set_index(config.ID_COL)
    test = pd.read_csv(config.TEST_PATH).set_index(config.ID_COL)
    print(f"[load] train={len(train):,} test={len(test):,}  ({time.time()-t0:.0f}s)")

    oof = pd.DataFrame(
        index=train.index,
        columns=["lgbm", "catboost", "xgboost"],
        dtype=np.float32,
    )
    oof[:] = np.nan

    y = train[config.TARGET_COL]
    groups = train["origin_account"]
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    fold_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(train, y, groups=groups)):
        t1 = time.time()
        tr_part = train.iloc[tr_idx].copy()
        va_part = train.iloc[va_idx].copy()
        combined = pd.concat([tr_part, va_part, test], axis=0)

        train_feat, val_feat, feature_cols = prepare_datasets(
            tr_part, va_part,
            te_source=tr_part,
            ref_unsupervised=combined,
            combined_for_expanding=combined,
        )
        y_tr = tr_part[config.TARGET_COL]
        y_va = va_part[config.TARGET_COL]

        lgbm, lgbm_s = train_lightgbm(train_feat, y_tr, val_feat, y_va, feature_cols)
        cat, cat_s = train_catboost(train_feat, y_tr, val_feat, y_va, feature_cols)
        xgb, xgb_s = train_xgboost(train_feat, y_tr, val_feat, y_va, feature_cols)

        oof.loc[va_part.index, "lgbm"] = predict_lightgbm(lgbm, val_feat, feature_cols, train_feat)
        oof.loc[va_part.index, "catboost"] = cat.predict_proba(val_feat[feature_cols])[:, 1]
        oof.loc[va_part.index, "xgboost"] = predict_xgboost(xgb, val_feat, feature_cols, train_feat)

        dt = time.time() - t1
        print(
            f"[fold {fold_idx}] lgbm={lgbm_s:.4f} cat={cat_s:.4f} xgb={xgb_s:.4f}  ({dt:.0f}s)"
        )
        fold_scores.append({"fold": fold_idx, "lgbm": float(lgbm_s),
                            "catboost": float(cat_s), "xgboost": float(xgb_s)})

    oof["target"] = y.values
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = config.REPORTS_DIR / "oof_predictions.csv"
    oof.reset_index().to_csv(oof_path, index=False)
    print(f"[save] OOF -> {oof_path}")

    oof_scores = {
        "lgbm": float(average_precision_score(y, oof["lgbm"])),
        "catboost": float(average_precision_score(y, oof["catboost"])),
        "xgboost": float(average_precision_score(y, oof["xgboost"])),
    }
    print("[oof] AP global par modele:", oof_scores)

    # Retrain on full train, predict test (base preds for stacking)
    print("[final] retrain on full train + predict test...")
    combined_full = pd.concat([train, test], axis=0)
    train_feat, test_feat, feature_cols = prepare_datasets(
        train, test,
        te_source=train,
        ref_unsupervised=combined_full,
        combined_for_expanding=combined_full,
    )
    # Hold-out interne pour early stopping
    tr_mask = train["period"] < config.VAL_PERIOD_CUTOFF
    tr_feat = train_feat.loc[tr_mask]
    va_feat = train_feat.loc[~tr_mask]
    y_tr = y.loc[tr_mask]
    y_va = y.loc[~tr_mask]

    lgbm, _ = train_lightgbm(tr_feat, y_tr, va_feat, y_va, feature_cols)
    cat, _ = train_catboost(tr_feat, y_tr, va_feat, y_va, feature_cols)
    xgb, _ = train_xgboost(tr_feat, y_tr, va_feat, y_va, feature_cols)

    test_preds = pd.DataFrame({
        config.ID_COL: test.index,
        "lgbm": predict_lightgbm(lgbm, test_feat, feature_cols, train_feat),
        "catboost": cat.predict_proba(test_feat[feature_cols])[:, 1],
        "xgboost": predict_xgboost(xgb, test_feat, feature_cols, train_feat),
    })
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_path = config.OUTPUT_DIR / "test_base_preds.csv"
    test_preds.to_csv(test_path, index=False)
    print(f"[save] test base preds -> {test_path}")

    report = {
        "n_splits": N_SPLITS,
        "fold_scores": fold_scores,
        "oof_scores": oof_scores,
        "total_seconds": time.time() - t0,
    }
    (config.REPORTS_DIR / "oof_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(f"[done] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
