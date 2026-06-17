"""
Pipeline v5 — pousser vers AP 0.50 sans triche.

Stratégie légitime :
1. Ensemble global (LGBM + CatBoost + XGBoost) sur toutes les transactions
2. Spécialiste op_03 (100 % des fraudes train sont op_03) — ranking fin
3. Fusion par rangs (optimale pour PR-AUC) + plancher non-op_03
4. Full-train avec plancher d'itérations élevé
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import config
from run_pipeline import (
    CAT_COLS,
    TE_COLS,
    _cat_indices,
    _prepare_lgbm_df,
    blend_predictions,
    load_data,
    predict_lightgbm,
    predict_xgboost,
    prepare_datasets,
    temporal_split,
    train_catboost,
    train_catboost_full,
    train_lightgbm,
    train_lightgbm_full,
    train_xgboost,
    train_xgboost_full,
)
from stacking import optimize_blend_weights

warnings.filterwarnings("ignore")

OP03 = "op_03"
FLOOR_RATIO = 0.05  # non-op_03 sous le min op_03


def _rank_series(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def rank_blend(a: np.ndarray, b: np.ndarray, w_a: float = 0.45) -> np.ndarray:
    return w_a * _rank_series(a) + (1.0 - w_a) * _rank_series(b)


def train_op03_specialist(
    train_feat: pd.DataFrame,
    y: pd.Series,
    operations: pd.Series,
    feature_cols: list[str],
    tr_idx: pd.Index,
    va_idx: pd.Index,
) -> LGBMClassifier:
    """LGBM profond entraîné uniquement sur op_03 (signal fraude concentré)."""
    op3_tr = tr_idx[operations.loc[tr_idx].eq(OP03)]
    op3_va = va_idx[operations.loc[va_idx].eq(OP03)]
    if len(op3_tr) < 100 or y.loc[op3_tr].sum() < 5:
        raise ValueError("Pas assez de données op_03 pour le spécialiste")

    pos = float(y.loc[op3_tr].sum())
    neg = len(op3_tr) - pos
    model = LGBMClassifier(
        objective="binary",
        metric="average_precision",
        n_estimators=2500,
        learning_rate=0.02,
        num_leaves=255,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.05,
        reg_lambda=0.5,
        scale_pos_weight=neg / max(pos, 1),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    X_tr = _prepare_lgbm_df(train_feat.loc[op3_tr], feature_cols)
    X_va = _prepare_lgbm_df(train_feat.loc[op3_va], feature_cols, ref=train_feat.loc[op3_tr])
    model.fit(
        X_tr,
        y.loc[op3_tr],
        eval_set=[(X_va, y.loc[op3_va])],
        eval_metric="average_precision",
        callbacks=[early_stopping(120, verbose=False)],
        categorical_feature=CAT_COLS,
    )
    return model


def train_op03_full(
    train_feat: pd.DataFrame,
    y: pd.Series,
    operations: pd.Series,
    feature_cols: list[str],
    n_estimators: int,
) -> LGBMClassifier:
    mask = operations.eq(OP03)
    X = train_feat.loc[mask]
    y_op = y.loc[mask]
    pos = float(y_op.sum())
    neg = len(y_op) - pos
    model = LGBMClassifier(
        objective="binary",
        metric="average_precision",
        n_estimators=max(int(n_estimators), config.MIN_FULL_ITERS.get("op03", 600)),
        learning_rate=0.02,
        num_leaves=255,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.05,
        reg_lambda=0.5,
        scale_pos_weight=neg / max(pos, 1),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        _prepare_lgbm_df(X, feature_cols),
        y_op,
        categorical_feature=CAT_COLS,
    )
    return model


def apply_two_stage_scores(
    test: pd.DataFrame,
    pred_global: np.ndarray,
    pred_op03: np.ndarray,
    rank_weight_global: float = 0.4,
) -> np.ndarray:
    is_op03 = test["operation"].eq(OP03).to_numpy()
    fused = np.zeros(len(test), dtype=np.float64)
    if is_op03.any():
        fused[is_op03] = rank_blend(
            pred_global[is_op03],
            pred_op03[is_op03],
            w_a=rank_weight_global,
        )
    if (~is_op03).any() and is_op03.any():
        floor = float(fused[is_op03].min()) * FLOOR_RATIO
        fused[~is_op03] = max(floor, 1e-8)
    elif (~is_op03).any():
        fused[~is_op03] = 1e-8
    return np.clip(fused, 0, 1)


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, test, sample = load_data()
    combined = pd.concat([train, test], axis=0)

    print("=== Features (157+) ===")
    train_feat, test_feat, feature_cols = prepare_datasets(
        train,
        test,
        te_source=train,
        ref_unsupervised=combined,
        combined_for_expanding=combined,
        use_oof_te=False,  # full TE pour soumission finale (pas de label test)
    )
    y = train[config.TARGET_COL]
    tr_fit, va_fit = temporal_split(train, config.VAL_PERIOD_CUTOFF)
    tr_idx, va_idx = tr_fit.index, va_fit.index
    tr_feat, va_feat = train_feat.loc[tr_idx], train_feat.loc[va_idx]

    print("=== Modèles globaux (hold-out) ===")
    lgbm, s_lgbm = train_lightgbm(tr_feat, y.loc[tr_idx], va_feat, y.loc[va_idx], feature_cols)
    cat, s_cat = train_catboost(tr_feat, y.loc[tr_idx], va_feat, y.loc[va_idx], feature_cols)
    xgb, s_xgb = train_xgboost(tr_feat, y.loc[tr_idx], va_feat, y.loc[va_idx], feature_cols)

    val_preds = pd.DataFrame(
        {
            "lgbm": predict_lightgbm(lgbm, va_feat, feature_cols, tr_feat),
            "catboost": cat.predict_proba(va_feat[feature_cols])[:, 1],
            "xgboost": predict_xgboost(xgb, va_feat, feature_cols, tr_feat),
        },
        index=va_idx,
    )
    weights = optimize_blend_weights(val_preds, y.loc[va_idx], restarts=12, seed=config.RANDOM_STATE)
    pred_global_val = blend_predictions(
        {k: val_preds[k].values for k in val_preds.columns},
        weights,
    )

    op3_va_mask = va_fit["operation"].eq(OP03).to_numpy()
    print("=== Spécialiste op_03 (hold-out) ===")
    spec = train_op03_specialist(
        train_feat, y, train["operation"], feature_cols, tr_idx, va_idx
    )
    op3_va = va_feat[va_fit["operation"].eq(OP03)]
    tr_op3_ref = train_feat.loc[tr_idx][train.loc[tr_idx, "operation"].eq(OP03)]
    pred_spec_val = np.zeros(len(va_feat), dtype=np.float64)
    if len(op3_va) > 0:
        pred_spec_val[op3_va_mask] = predict_lightgbm(spec, op3_va, feature_cols, tr_op3_ref)

    fused_val = apply_two_stage_scores(va_fit, pred_global_val, pred_spec_val, rank_weight_global=0.38)
    ap_global = average_precision_score(y.loc[va_idx], pred_global_val)
    ap_fused = average_precision_score(y.loc[va_idx], fused_val)
    print(f"Val AP global  : {ap_global:.6f}")
    print(f"Val AP two-stage: {ap_fused:.6f}")
    print(f"Poids ensemble : {weights}")

    print("\n=== Full train ===")
    lgbm_iters = max(int(getattr(lgbm, "best_iteration_", 0) or 2000), config.MIN_FULL_ITERS["lgbm"])
    cat_iters = max(int((cat.get_best_iteration() or 2999) + 1), config.MIN_FULL_ITERS["catboost"])
    xgb_iters = max(int((getattr(xgb, "best_iteration", None) or 1999) + 1), config.MIN_FULL_ITERS["xgboost"])
    spec_iters = max(int(getattr(spec, "best_iteration_", 0) or 1500), config.MIN_FULL_ITERS.get("op03", 600))

    lgbm = train_lightgbm_full(train_feat, y, feature_cols, lgbm_iters)
    cat = train_catboost_full(train_feat, y, feature_cols, cat_iters)
    xgb = train_xgboost_full(train_feat, y, feature_cols, xgb_iters)
    spec = train_op03_full(train_feat, y, train["operation"], feature_cols, spec_iters)

    test_preds = {
        "lgbm": predict_lightgbm(lgbm, test_feat, feature_cols, train_feat),
        "catboost": cat.predict_proba(test_feat[feature_cols])[:, 1],
        "xgboost": predict_xgboost(xgb, test_feat, feature_cols, train_feat),
    }
    pred_global_test = blend_predictions(test_preds, weights)

    pred_spec_test = np.zeros(len(test_feat))
    op3_test = test["operation"].eq(OP03).to_numpy()
    if op3_test.any():
        test_op3 = test_feat[test["operation"].eq(OP03)]
        pred_spec_test[op3_test] = predict_lightgbm(
            spec,
            test_op3,
            feature_cols,
            train_feat[train["operation"].eq(OP03)],
        )

    final = apply_two_stage_scores(test, pred_global_test, pred_spec_test, rank_weight_global=0.38)

    submission = pd.DataFrame({config.ID_COL: test.index, config.SUBMIT_COL: final})
    submission = submission.merge(sample[[config.ID_COL]], on=config.ID_COL, how="right")
    submission[config.SUBMIT_COL] = submission[config.SUBMIT_COL].clip(0, 1)

    out = config.ROOT / "submission_push_v5.csv"
    submission.to_csv(out, index=False)
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    submission.to_csv(config.ROOT / "submision.csv", index=False)

    report = {
        "val_ap_global": float(ap_global),
        "val_ap_two_stage": float(ap_fused),
        "ensemble_weights": weights,
        "iters": {"lgbm": lgbm_iters, "catboost": cat_iters, "xgboost": xgb_iters, "op03": spec_iters},
        "n_features": len(feature_cols),
    }
    (config.REPORTS_DIR / "push_v5_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSoumission : {out}")
    print(submission[config.SUBMIT_COL].describe())
    print("Soumission valide.")


if __name__ == "__main__":
    main()
