"""Validation rapide : sous-echantillonne par comptes pour mesurer le gain.

On garde une fraction (par defaut 25%) des comptes origin choisis aleatoirement
puis on filtre train+test sur ces comptes. La structure graphe et temporelle
locale est preservee, ce qui donne un signal de gain comparable au baseline en
~10-15 min au lieu de ~3h.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import config
from run_pipeline import (
    CAT_COLS,
    TE_COLS,
    blend_predictions,
    ensemble_weights,
    predict_lightgbm,
    predict_xgboost,
    prepare_datasets,
    train_catboost,
    train_lightgbm,
    train_xgboost,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.25, help="fraction de comptes a garder")
    ap.add_argument("--cutoff", type=int, default=96, help="cutoff period train|val")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] train={config.TRAIN_PATH}")
    train_full = pd.read_csv(config.TRAIN_PATH).set_index(config.ID_COL)
    print(f"[load] test ={config.TEST_PATH}")
    test_full = pd.read_csv(config.TEST_PATH).set_index(config.ID_COL)
    print(f"[load] done in {time.time()-t0:.0f}s. train={len(train_full):,} test={len(test_full):,}")

    rng = np.random.default_rng(args.seed)
    all_accounts = pd.Index(
        pd.unique(pd.concat([train_full["origin_account"], test_full["origin_account"]]))
    )
    n_keep = max(int(len(all_accounts) * args.frac), 1)
    keep = set(rng.choice(all_accounts, size=n_keep, replace=False))
    print(f"[sample] kept {n_keep}/{len(all_accounts)} origin accounts ({args.frac*100:.0f}%)")

    train_sub = train_full[train_full["origin_account"].isin(keep)].copy()
    test_sub = test_full[test_full["origin_account"].isin(keep)].copy()
    print(f"[sample] train_sub={len(train_sub):,} test_sub={len(test_sub):,}")

    tr_part = train_sub[train_sub["period"] < args.cutoff].copy()
    va_part = train_sub[train_sub["period"] >= args.cutoff].copy()
    print(f"[split] cutoff={args.cutoff} tr={len(tr_part):,} va={len(va_part):,}")
    print(f"[split] fraud_tr={tr_part[config.TARGET_COL].mean():.4f}  "
          f"fraud_va={va_part[config.TARGET_COL].mean():.4f}")

    combined = pd.concat([tr_part, va_part, test_sub], axis=0)
    print(f"[prep] preparing features on {len(combined):,} rows...")
    t1 = time.time()
    train_feat, val_feat, feature_cols = prepare_datasets(
        tr_part,
        va_part,
        te_source=tr_part,
        ref_unsupervised=combined,
        combined_for_expanding=combined,
    )
    print(f"[prep] done in {time.time()-t1:.0f}s. n_features={len(feature_cols)}")

    y_tr = tr_part[config.TARGET_COL]
    y_va = va_part[config.TARGET_COL]

    print("[train] LightGBM...")
    t1 = time.time()
    lgbm, lgbm_s = train_lightgbm(train_feat, y_tr, val_feat, y_va, feature_cols)
    print(f"[train] LGBM AP = {lgbm_s:.4f}  ({time.time()-t1:.0f}s)")

    print("[train] CatBoost...")
    t1 = time.time()
    cat, cat_s = train_catboost(train_feat, y_tr, val_feat, y_va, feature_cols)
    print(f"[train] CatBoost AP = {cat_s:.4f}  ({time.time()-t1:.0f}s)")

    print("[train] XGBoost...")
    t1 = time.time()
    xgb, xgb_s = train_xgboost(train_feat, y_tr, val_feat, y_va, feature_cols)
    print(f"[train] XGBoost AP = {xgb_s:.4f}  ({time.time()-t1:.0f}s)")

    scores = {"lgbm": float(lgbm_s), "catboost": float(cat_s), "xgboost": float(xgb_s)}
    w = ensemble_weights(scores)
    preds = {
        "lgbm": predict_lightgbm(lgbm, val_feat, feature_cols, train_feat),
        "catboost": cat.predict_proba(val_feat[feature_cols])[:, 1],
        "xgboost": predict_xgboost(xgb, val_feat, feature_cols, train_feat),
    }
    ens = blend_predictions(preds, w)
    ens_s = float(average_precision_score(y_va, ens))

    print()
    print("=" * 60)
    print(f"BASELINE (avant Phase 0+1) : AP ~ 0.360 (memorise)")
    print(f"LightGBM solo              : {lgbm_s:.4f}")
    print(f"CatBoost solo              : {cat_s:.4f}")
    print(f"XGBoost solo               : {xgb_s:.4f}")
    print(f"Ensemble pondere par AP    : {ens_s:.4f}")
    print(f"Gain vs baseline           : {ens_s-0.360:+.4f}")
    print(f"Total runtime              : {time.time()-t0:.0f}s")
    print("=" * 60)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.REPORTS_DIR / "quick_validation.json"
    out_path.write_text(
        json.dumps(
            {
                "frac": args.frac,
                "cutoff": args.cutoff,
                "n_train": int(len(tr_part)),
                "n_val": int(len(va_part)),
                "n_features": len(feature_cols),
                "scores": scores,
                "ensemble": ens_s,
                "weights": w,
                "baseline_ref": 0.360,
                "gain": ens_s - 0.360,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
