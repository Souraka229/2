"""Validation rolling temporelle + group-aware pour DataTour 2026.

Usage :
    python validation.py             # CV rolling temporelle 5 folds
    python validation.py --group     # CV group-aware 5 folds en plus
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

import config
from features import (
    add_expanding_temporal_features,
    build_features,
    compute_oof_target_encoding,
    get_feature_columns,
)
from run_pipeline import (
    CAT_COLS,
    TE_COLS,
    train_lightgbm,
    train_catboost,
    train_xgboost,
    predict_lightgbm,
    predict_xgboost,
    blend_predictions,
    ensemble_weights,
)


ROLLING_CUTOFFS = [75, 81, 87, 93, 99]


def _prepare_fold(
    tr_part: pd.DataFrame,
    va_part: pd.DataFrame,
    use_oof_te: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    combined = pd.concat([tr_part, va_part], axis=0)
    expanding = add_expanding_temporal_features(combined)

    if use_oof_te:
        oof_te_df, te_maps = compute_oof_target_encoding(
            tr_part,
            tr_part[config.TARGET_COL],
            TE_COLS,
            config.TE_SMOOTHING,
            n_splits=5,
            groups=tr_part["origin_account"],
            random_state=config.RANDOM_STATE,
        )
    else:
        from features import fit_target_encoding
        te_maps = fit_target_encoding(
            tr_part, tr_part[config.TARGET_COL], TE_COLS, config.TE_SMOOTHING
        )
        oof_te_df = None

    train_feat = build_features(
        tr_part,
        ref_unsupervised=tr_part,
        te_maps=te_maps,
        te_cols=TE_COLS,
        expanding_df=expanding.loc[tr_part.index],
    )
    if oof_te_df is not None:
        for c in TE_COLS:
            train_feat[f"{c}_fraud_rate"] = oof_te_df[f"{c}_fraud_rate"].values
    train_feat.index = tr_part.index

    val_feat = build_features(
        va_part,
        ref_unsupervised=tr_part,
        te_maps=te_maps,
        te_cols=TE_COLS,
        expanding_df=expanding.loc[va_part.index],
    )
    val_feat.index = va_part.index

    feature_cols = get_feature_columns(train_feat)
    return train_feat, val_feat, feature_cols


def rolling_temporal_cv(train: pd.DataFrame, cutoffs: list[int]) -> dict:
    """Pour chaque cutoff: train sur period<cutoff, valide sur [cutoff, cutoff+6)."""
    results = []
    for cutoff in cutoffs:
        t0 = time.time()
        tr_part = train[train["period"] < cutoff].copy()
        va_part = train[(train["period"] >= cutoff) & (train["period"] < cutoff + 6)].copy()
        if va_part.empty or tr_part.empty:
            continue

        train_feat, val_feat, feature_cols = _prepare_fold(tr_part, va_part, use_oof_te=True)
        y_tr = tr_part[config.TARGET_COL]
        y_va = va_part[config.TARGET_COL]

        lgbm, lgbm_s = train_lightgbm(train_feat, y_tr, val_feat, y_va, feature_cols)
        cat, cat_s = train_catboost(train_feat, y_tr, val_feat, y_va, feature_cols)
        xgb, xgb_s = train_xgboost(train_feat, y_tr, val_feat, y_va, feature_cols)

        scores = {"lgbm": lgbm_s, "catboost": cat_s, "xgboost": xgb_s}
        w = ensemble_weights(scores)
        preds = {
            "lgbm": predict_lightgbm(lgbm, val_feat, feature_cols, train_feat),
            "catboost": cat.predict_proba(val_feat[feature_cols])[:, 1],
            "xgboost": predict_xgboost(xgb, val_feat, feature_cols, train_feat),
        }
        ens = blend_predictions(preds, w)
        ens_s = float(average_precision_score(y_va, ens))

        dt = time.time() - t0
        print(
            f"[cutoff={cutoff:3d}] tr={len(tr_part):>7,}  va={len(va_part):>6,}  "
            f"lgbm={lgbm_s:.4f}  cat={cat_s:.4f}  xgb={xgb_s:.4f}  ens={ens_s:.4f}  ({dt:.0f}s)"
        )
        results.append(
            {
                "cutoff": cutoff,
                "n_train": int(len(tr_part)),
                "n_val": int(len(va_part)),
                "scores": {k: float(v) for k, v in scores.items()},
                "ensemble": ens_s,
                "weights": w,
                "seconds": dt,
            }
        )

    ens_scores = [r["ensemble"] for r in results]
    summary = {
        "type": "rolling_temporal",
        "folds": results,
        "mean_ensemble": float(np.mean(ens_scores)),
        "std_ensemble": float(np.std(ens_scores)),
        "per_model_mean": {
            m: float(np.mean([r["scores"][m] for r in results]))
            for m in ["lgbm", "catboost", "xgboost"]
        },
    }
    return summary


def group_kfold_cv(train: pd.DataFrame, n_splits: int = 5) -> dict:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    y = train[config.TARGET_COL]
    groups = train["origin_account"]
    results = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(train, y, groups=groups)):
        t0 = time.time()
        tr_part = train.iloc[tr_idx].copy()
        va_part = train.iloc[va_idx].copy()
        train_feat, val_feat, feature_cols = _prepare_fold(tr_part, va_part, use_oof_te=True)
        y_tr = tr_part[config.TARGET_COL]
        y_va = va_part[config.TARGET_COL]

        lgbm, lgbm_s = train_lightgbm(train_feat, y_tr, val_feat, y_va, feature_cols)
        cat, cat_s = train_catboost(train_feat, y_tr, val_feat, y_va, feature_cols)
        xgb, xgb_s = train_xgboost(train_feat, y_tr, val_feat, y_va, feature_cols)

        scores = {"lgbm": lgbm_s, "catboost": cat_s, "xgboost": xgb_s}
        w = ensemble_weights(scores)
        preds = {
            "lgbm": predict_lightgbm(lgbm, val_feat, feature_cols, train_feat),
            "catboost": cat.predict_proba(val_feat[feature_cols])[:, 1],
            "xgboost": predict_xgboost(xgb, val_feat, feature_cols, train_feat),
        }
        ens = blend_predictions(preds, w)
        ens_s = float(average_precision_score(y_va, ens))
        dt = time.time() - t0
        print(
            f"[group fold {fold_idx}] tr={len(tr_part):>7,}  va={len(va_part):>6,}  "
            f"ens={ens_s:.4f}  ({dt:.0f}s)"
        )
        results.append({"fold": fold_idx, "ensemble": ens_s, "scores": scores, "weights": w})

    ens_scores = [r["ensemble"] for r in results]
    return {
        "type": "group_kfold",
        "folds": results,
        "mean_ensemble": float(np.mean(ens_scores)),
        "std_ensemble": float(np.std(ens_scores)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="store_true", help="exec aussi CV group-aware")
    ap.add_argument("--no-temporal", action="store_true")
    args = ap.parse_args()

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(config.TRAIN_PATH).set_index(config.ID_COL)

    out: dict = {}
    if not args.no_temporal:
        print(f"\n=== CV rolling temporelle, cutoffs={ROLLING_CUTOFFS} ===")
        out["rolling_temporal"] = rolling_temporal_cv(train, ROLLING_CUTOFFS)
        print(
            f"\nRolling mean AP = {out['rolling_temporal']['mean_ensemble']:.4f} "
            f"+- {out['rolling_temporal']['std_ensemble']:.4f}"
        )

    if args.group:
        print(f"\n=== CV group-aware 5 folds ===")
        out["group_kfold"] = group_kfold_cv(train, n_splits=5)
        print(
            f"\nGroup mean AP = {out['group_kfold']['mean_ensemble']:.4f} "
            f"+- {out['group_kfold']['std_ensemble']:.4f}"
        )

    target = config.REPORTS_DIR / "validation_rolling.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport ecrit: {target}")


if __name__ == "__main__":
    main()
