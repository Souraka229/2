"""
Pipeline DataTour 2026 — détection fraude mobile money.
Validation temporelle, features avancées, LightGBM + CatBoost + XGBoost, ensemble.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

import config
from features import (
    add_expanding_temporal_features,
    build_features,
    compute_oof_target_encoding,
    fit_target_encoding,
    get_feature_columns,
)
from graph_features import fit_graph_features
from temporal_features import add_rolling_features
from account_features import (
    ISO_FEATURES,
    fit_account_profiles,
    fit_isolation_forest,
    score_isolation,
)
from features import add_balance_features

warnings.filterwarnings("ignore")

TE_COLS = ["operation", "origin_account", "destination_account"]
CAT_COLS = ["operation"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Chargement train : {config.TRAIN_PATH}")
    print(f"Chargement test  : {config.TEST_PATH}")
    train = pd.read_csv(config.TRAIN_PATH).set_index(config.ID_COL)
    test = pd.read_csv(config.TEST_PATH).set_index(config.ID_COL)
    sample = pd.read_csv(config.SAMPLE_PATH)
    return train, test, sample


def run_eda(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_period_min": int(train["period"].min()),
        "train_period_max": int(train["period"].max()),
        "test_period_min": int(test["period"].min()),
        "test_period_max": int(test["period"].max()),
        "fraud_rate": float(train[config.TARGET_COL].mean()),
        "train_missing": int(train.isna().sum().sum()),
        "test_missing": int(test.isna().sum().sum()),
        "train_operations": sorted(train["operation"].unique().tolist()),
        "origin_accounts_train": int(train["origin_account"].nunique()),
        "origin_accounts_test": int(test["origin_account"].nunique()),
        "dest_accounts_train": int(train["destination_account"].nunique()),
        "dest_accounts_test": int(test["destination_account"].nunique()),
        "accounts_overlap_origin": int(
            len(set(train["origin_account"]) & set(test["origin_account"]))
        ),
        "accounts_overlap_dest": int(
            len(set(train["destination_account"]) & set(test["destination_account"]))
        ),
    }
    path = config.REPORTS_DIR / "eda_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("EDA enregistré :", path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def temporal_split(train: pd.DataFrame, cutoff: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train[train["period"] < cutoff].copy()
    va = train[train["period"] >= cutoff].copy()
    return tr, va


def group_split(train: pd.DataFrame, frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=config.RANDOM_STATE)
    groups = train["origin_account"]
    idx_tr, idx_va = next(splitter.split(train, groups=groups))
    return train.iloc[idx_tr].copy(), train.iloc[idx_va].copy()


def prepare_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    te_source: pd.DataFrame,
    ref_unsupervised: pd.DataFrame,
    combined_for_expanding: pd.DataFrame,
    use_oof_te: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    # Target encoding : OOF cote train (anti-fuite), maps full-fit cote test.
    if use_oof_te and len(te_source) == len(train_df) and te_source.index.equals(train_df.index):
        oof_df, te_maps = compute_oof_target_encoding(
            train_df,
            train_df[config.TARGET_COL],
            TE_COLS,
            config.TE_SMOOTHING,
            n_splits=5,
            groups=train_df["origin_account"],
            random_state=config.RANDOM_STATE,
        )
    else:
        te_maps = fit_target_encoding(
            te_source,
            te_source[config.TARGET_COL],
            TE_COLS,
            config.TE_SMOOTHING,
        )
        oof_df = None

    expanding = add_expanding_temporal_features(combined_for_expanding)
    rolling = add_rolling_features(combined_for_expanding)
    graph_origin, graph_dest = fit_graph_features(ref_unsupervised)
    acc_origin, acc_dest = fit_account_profiles(combined_for_expanding)

    # IsolationForest sur features balance/montant (calcul sur combined → pas de fuite).
    iso_input = add_balance_features(combined_for_expanding)
    iso_feats_present = [c for c in ISO_FEATURES if c in iso_input.columns]
    iso = fit_isolation_forest(iso_input, iso_feats_present)
    iso_scores = pd.Series(
        score_isolation(iso, iso_input, iso_feats_present),
        index=iso_input.index,
        name="iso_anomaly_score",
    )

    train_feat = build_features(
        train_df,
        ref_unsupervised=ref_unsupervised,
        te_maps=te_maps if oof_df is None else None,
        te_cols=TE_COLS if oof_df is None else None,
        te_oof=oof_df,
        expanding_df=expanding.loc[train_df.index],
        graph_origin_stats=graph_origin,
        graph_dest_stats=graph_dest,
        rolling_df=rolling,
        account_origin_profile=acc_origin,
        account_dest_profile=acc_dest,
        iso_score=iso_scores,
    )
    train_feat.index = train_df.index

    test_feat = None
    if test_df is not None:
        test_feat = build_features(
            test_df,
            ref_unsupervised=ref_unsupervised,
            te_maps=te_maps,
            te_cols=TE_COLS,
            expanding_df=expanding.loc[test_df.index],
            graph_origin_stats=graph_origin,
            graph_dest_stats=graph_dest,
            rolling_df=rolling,
            account_origin_profile=acc_origin,
            account_dest_profile=acc_dest,
            iso_score=iso_scores,
        )
        test_feat.index = test_df.index

    feature_cols = get_feature_columns(train_feat)
    return train_feat, test_feat, feature_cols


def _prepare_lgbm_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    ref: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df[feature_cols].copy()
    for c in CAT_COLS:
        if c not in out.columns:
            continue
        if ref is not None and c in ref.columns:
            categories = ref[c].astype(str).unique().tolist()
            out[c] = pd.Categorical(out[c].astype(str), categories=categories)
        else:
            out[c] = out[c].astype("category")
    return out


def predict_lightgbm(
    model: LGBMClassifier,
    df: pd.DataFrame,
    feature_cols: list[str],
    ref: pd.DataFrame,
) -> np.ndarray:
    X = _prepare_lgbm_df(df, feature_cols, ref=ref)
    return model.predict_proba(X)[:, 1]


def _cat_indices(feature_cols: list[str]) -> list[int]:
    return [i for i, c in enumerate(feature_cols) if c in CAT_COLS]


def train_lightgbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    feature_cols: list[str],
) -> tuple[LGBMClassifier, float]:
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    scale = neg / max(pos, 1)

    model = LGBMClassifier(
        objective="binary",
        metric="average_precision",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale,
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
        callbacks=[early_stopping(100, verbose=False)],
        categorical_feature=CAT_COLS,
    )
    pred = model.predict_proba(X_va_fit)[:, 1]
    score = average_precision_score(y_va, pred)
    return model, score


def train_lightgbm_full(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    n_estimators: int,
) -> LGBMClassifier:
    pos = y.sum()
    neg = len(y) - pos
    scale = neg / max(pos, 1)
    model = LGBMClassifier(
        objective="binary",
        metric="average_precision",
        n_estimators=max(int(n_estimators), 50),
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        _prepare_lgbm_df(X, feature_cols),
        y,
        categorical_feature=CAT_COLS,
    )
    return model


def train_catboost(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    feature_cols: list[str],
) -> tuple[CatBoostClassifier, float]:
    cat_idx = _cat_indices(feature_cols)
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        auto_class_weights="Balanced",
        eval_metric="Logloss",
        random_seed=config.RANDOM_STATE,
        verbose=0,
        early_stopping_rounds=100,
    )
    model.fit(
        Pool(X_tr[feature_cols], y_tr, cat_features=cat_idx),
        eval_set=Pool(X_va[feature_cols], y_va, cat_features=cat_idx),
    )
    pred = model.predict_proba(X_va[feature_cols])[:, 1]
    score = average_precision_score(y_va, pred)
    return model, score


def train_catboost_full(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    iterations: int,
) -> CatBoostClassifier:
    cat_idx = _cat_indices(feature_cols)
    model = CatBoostClassifier(
        iterations=max(int(iterations), 50),
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        auto_class_weights="Balanced",
        eval_metric="Logloss",
        random_seed=config.RANDOM_STATE,
        verbose=0,
    )
    model.fit(Pool(X[feature_cols], y, cat_features=cat_idx))
    return model


def train_xgboost(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    feature_cols: list[str],
) -> tuple[XGBClassifier, float]:
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    scale = neg / max(pos, 1)

    X_tr_enc = X_tr[feature_cols].copy()
    X_va_enc = X_va[feature_cols].copy()
    for c in CAT_COLS:
        if c in X_tr_enc.columns:
            combined = pd.concat([X_tr_enc[c], X_va_enc[c]], axis=0).astype("category")
            cats = combined.cat.categories
            X_tr_enc[c] = pd.Categorical(X_tr_enc[c], categories=cats).codes
            X_va_enc[c] = pd.Categorical(X_va_enc[c], categories=cats).codes

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=100,
        tree_method="hist",
    )
    model.fit(
        X_tr_enc,
        y_tr,
        eval_set=[(X_va_enc, y_va)],
        verbose=False,
    )
    pred = model.predict_proba(X_va_enc)[:, 1]
    score = average_precision_score(y_va, pred)
    return model, score


def _encode_xgb_frames(
    X: pd.DataFrame,
    ref: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    X_enc = X[feature_cols].copy()
    ref_enc = ref[feature_cols].copy()
    for c in CAT_COLS:
        if c in X_enc.columns:
            combined = pd.concat([ref_enc[c], X_enc[c]], axis=0).astype("category")
            cats = combined.cat.categories
            X_enc[c] = pd.Categorical(X_enc[c], categories=cats).codes
    return X_enc


def train_xgboost_full(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    n_estimators: int,
) -> XGBClassifier:
    pos = y.sum()
    neg = len(y) - pos
    scale = neg / max(pos, 1)
    X_enc = _encode_xgb_frames(X, X, feature_cols)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=max(int(n_estimators), 50),
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_enc, y, verbose=False)
    return model


def predict_xgboost(model: XGBClassifier, X: pd.DataFrame, feature_cols: list[str], ref: pd.DataFrame) -> np.ndarray:
    X_enc = _encode_xgb_frames(X, ref, feature_cols)
    return model.predict_proba(X_enc)[:, 1]


def ensemble_weights(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        n = len(scores)
        return {k: 1 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


def blend_predictions(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(next(iter(preds.values())), dtype=float)
    for name, arr in preds.items():
        out += weights[name] * arr
    return np.clip(out, 0, 1)


def apply_operation_prior(
    prediction: np.ndarray,
    test: pd.DataFrame,
    positive_operation: str = "op_03",
    shrink_factor: float = 0.01,
) -> np.ndarray:
    """Prior catégoriel: dans le train, 100% des fraudes sont sur op_03.

    Au lieu d'écraser les non-op_03 à 1e-8 (perte totale si une seule fraude
    existe hors op_03 dans le test), on multiplie par `shrink_factor`. Plus
    robuste au mismatch train/test eventuel.
    """
    adjusted = prediction.copy()
    positive_mask = test["operation"].eq(positive_operation).to_numpy()
    if positive_mask.any() and (~positive_mask).any():
        adjusted[~positive_mask] = adjusted[~positive_mask] * shrink_factor
    return np.clip(adjusted, 0, 1)


def validate_models(
    train: pd.DataFrame,
    cutoff: int,
) -> dict:
    tr_part, va_part = temporal_split(train, cutoff)
    combined = pd.concat([tr_part, va_part], axis=0)
    train_feat, val_feat, feature_cols = prepare_datasets(
        tr_part,
        va_part,
        te_source=tr_part,
        ref_unsupervised=tr_part,
        combined_for_expanding=combined,
    )
    y_va = va_part[config.TARGET_COL]

    print(f"\n=== Validation temporelle (period < {cutoff}) ===")
    print(f"Train: {len(tr_part):,} | Val: {len(va_part):,}")

    lgbm, lgbm_score = train_lightgbm(
        train_feat, tr_part[config.TARGET_COL], val_feat, y_va, feature_cols
    )
    cat, cat_score = train_catboost(
        train_feat, tr_part[config.TARGET_COL], val_feat, y_va, feature_cols
    )
    xgb, xgb_score = train_xgboost(
        train_feat, tr_part[config.TARGET_COL], val_feat, y_va, feature_cols
    )

    scores = {"lgbm": lgbm_score, "catboost": cat_score, "xgboost": xgb_score}

    preds = {
        "lgbm": predict_lightgbm(lgbm, val_feat, feature_cols, train_feat),
        "catboost": cat.predict_proba(val_feat[feature_cols])[:, 1],
        "xgboost": predict_xgboost(xgb, val_feat, feature_cols, train_feat),
    }

    # Poids optimises par Nelder-Mead sur AP, fallback sur ponderation par score.
    from stacking import optimize_blend_weights
    pred_df = pd.DataFrame(preds, index=val_feat.index)
    weights_opt = optimize_blend_weights(pred_df, y_va, restarts=8, seed=config.RANDOM_STATE)
    weights_avg = ensemble_weights(scores)
    ens_pred_opt = blend_predictions(preds, weights_opt)
    ens_pred_avg = blend_predictions(preds, weights_avg)
    ens_score_opt = average_precision_score(y_va, ens_pred_opt)
    ens_score_avg = average_precision_score(y_va, ens_pred_avg)
    if ens_score_opt >= ens_score_avg:
        weights = weights_opt
        ens_score = ens_score_opt
        ens_pred = ens_pred_opt
        ens_kind = "optimized"
    else:
        weights = weights_avg
        ens_score = ens_score_avg
        ens_pred = ens_pred_avg
        ens_kind = "score-weighted (fallback)"

    print(f"LightGBM AP   : {lgbm_score:.6f}")
    print(f"CatBoost AP   : {cat_score:.6f}")
    print(f"XGBoost AP    : {xgb_score:.6f}")
    print(f"Ensemble AP   : {ens_score:.6f}  ({ens_kind})")
    print(f"  - opt    : {ens_score_opt:.6f}  poids={weights_opt}")
    print(f"  - avg    : {ens_score_avg:.6f}  poids={weights_avg}")

    # Validation group-aware secondaire
    tr_g, va_g = group_split(train)
    combined_g = pd.concat([tr_g, va_g], axis=0)
    tr_g_feat, va_g_feat, fc_g = prepare_datasets(
        tr_g,
        va_g,
        te_source=tr_g,
        ref_unsupervised=tr_g,
        combined_for_expanding=combined_g,
    )
    y_va_g = va_g[config.TARGET_COL]
    _, g_lgbm = train_lightgbm(tr_g_feat, tr_g[config.TARGET_COL], va_g_feat, y_va_g, fc_g)
    print(f"\n=== Validation group-aware (LightGBM) ===")
    print(f"Group AP: {g_lgbm:.6f}")

    report = {
        "temporal_cutoff": cutoff,
        "temporal_scores": scores,
        "ensemble_score": ens_score,
        "ensemble_weights": weights,
        "group_aware_lgbm": g_lgbm,
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def train_final_and_submit(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    combined = pd.concat([train, test], axis=0)
    train_feat, test_feat, feature_cols = prepare_datasets(
        train,
        test,
        te_source=train,
        ref_unsupervised=combined,
        combined_for_expanding=combined,
    )
    y = train[config.TARGET_COL]

    # Hold-out interne sur dernières périodes train pour early stopping
    tr_fit, va_fit = temporal_split(train, config.VAL_PERIOD_CUTOFF)

    tr_idx = tr_fit.index
    va_idx = va_fit.index
    tr_feat = train_feat.loc[tr_idx]
    va_feat = train_feat.loc[va_idx]

    print("\n=== Entraînement final ===")
    lgbm, _ = train_lightgbm(tr_feat, y.loc[tr_idx], va_feat, y.loc[va_idx], feature_cols)
    cat, _ = train_catboost(tr_feat, y.loc[tr_idx], va_feat, y.loc[va_idx], feature_cols)
    xgb, _ = train_xgboost(tr_feat, y.loc[tr_idx], va_feat, y.loc[va_idx], feature_cols)

    lgbm_iters = max(
        int(getattr(lgbm, "best_iteration_", 0) or 2000),
        config.MIN_FULL_ITERS["lgbm"],
    )
    cat_iters = max(
        int((cat.get_best_iteration() or 2999) + 1),
        config.MIN_FULL_ITERS["catboost"],
    )
    xgb_iters = max(
        int((getattr(xgb, "best_iteration", None) or 1999) + 1),
        config.MIN_FULL_ITERS["xgboost"],
    )
    print(
        "Réentraînement full train avec itérations:",
        {"lgbm": lgbm_iters, "catboost": cat_iters, "xgboost": xgb_iters},
    )

    lgbm = train_lightgbm_full(train_feat, y, feature_cols, lgbm_iters)
    cat = train_catboost_full(train_feat, y, feature_cols, cat_iters)
    xgb = train_xgboost_full(train_feat, y, feature_cols, xgb_iters)

    preds = {
        "lgbm": predict_lightgbm(lgbm, test_feat, feature_cols, train_feat),
        "catboost": cat.predict_proba(test_feat[feature_cols])[:, 1],
        "xgboost": predict_xgboost(xgb, test_feat, feature_cols, train_feat),
    }
    pred_frame = pd.DataFrame({config.ID_COL: test.index, **preds})
    pred_frame.to_csv(config.OUTPUT_DIR / "test_model_predictions.csv", index=False)

    final_pred = blend_predictions(preds, weights)
    if config.USE_OPERATION_PRIOR:
        final_pred = apply_operation_prior(final_pred, test)

    submission = pd.DataFrame({config.ID_COL: test.index, config.SUBMIT_COL: final_pred})
    submission = submission.merge(sample[[config.ID_COL]], on=config.ID_COL, how="right")
    submission[config.SUBMIT_COL] = submission[config.SUBMIT_COL].fillna(train[config.TARGET_COL].mean())
    submission[config.SUBMIT_COL] = submission[config.SUBMIT_COL].clip(0, 1)

    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Soumission écrite : {config.SUBMISSION_PATH}")
    print(submission[config.SUBMIT_COL].describe())

    assert submission.shape[0] == sample.shape[0]
    assert list(submission.columns) == [config.ID_COL, config.SUBMIT_COL]
    assert submission[config.ID_COL].is_unique
    assert submission[config.SUBMIT_COL].between(0, 1).all()
    assert set(submission[config.ID_COL]) == set(sample[config.ID_COL])
    print("Soumission valide.")
    return submission


def main(final_only: bool = False) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, test, sample = load_data()
    run_eda(train, test)

    report_path = config.REPORTS_DIR / "validation_report.json"
    if final_only and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        weights = report["ensemble_weights"]
        print("Reprise avec poids validés :", weights)
    else:
        report = validate_models(train, config.VAL_PERIOD_CUTOFF)
        weights = report["ensemble_weights"]

    train_final_and_submit(train, test, sample, weights)
    print("\nPipeline terminée.")


if __name__ == "__main__":
    import sys

    main(final_only="--final-only" in sys.argv)
