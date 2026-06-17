"""Tuning Optuna avec early stopping pour LightGBM, CatBoost, XGBoost.

Espace de recherche large + pruner Hyperband + TPESampler. Validation par
hold-out temporel unique (cutoff 96) pour rapidite. Pour la robustesse finale,
relancer la CV rolling avec les meilleurs hyperparam.

Usage:
    python optuna_tuning.py --model lgbm --trials 100
    python optuna_tuning.py --model catboost --trials 80
    python optuna_tuning.py --model xgboost --trials 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config


def _load_features():
    """Charge train+test puis prepare les features une seule fois."""
    from run_pipeline import prepare_datasets

    train = pd.read_csv(config.TRAIN_PATH).set_index(config.ID_COL)
    test = pd.read_csv(config.TEST_PATH).set_index(config.ID_COL)

    tr_part = train[train["period"] < config.VAL_PERIOD_CUTOFF].copy()
    va_part = train[train["period"] >= config.VAL_PERIOD_CUTOFF].copy()
    combined = pd.concat([train, test], axis=0)

    train_feat, val_feat, feature_cols = prepare_datasets(
        tr_part, va_part,
        te_source=tr_part,
        ref_unsupervised=combined,
        combined_for_expanding=combined,
    )
    y_tr = tr_part[config.TARGET_COL]
    y_va = va_part[config.TARGET_COL]
    return train_feat, val_feat, feature_cols, y_tr, y_va


def tune_lgbm(trials: int, train_feat, val_feat, feature_cols, y_tr, y_va) -> dict:
    import optuna
    from sklearn.metrics import average_precision_score
    from lightgbm import LGBMClassifier, early_stopping
    from run_pipeline import _prepare_lgbm_df, CAT_COLS

    X_tr = _prepare_lgbm_df(train_feat, feature_cols)
    X_va = _prepare_lgbm_df(val_feat, feature_cols, ref=train_feat)

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            objective="binary",
            metric="average_precision",
            n_estimators=3000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 511),
            max_depth=trial.suggest_int("max_depth", -1, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 15.0),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            max_bin=trial.suggest_int("max_bin", 127, 511),
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model = LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="average_precision",
            callbacks=[early_stopping(80, verbose=False)],
            categorical_feature=CAT_COLS,
        )
        pred = model.predict_proba(X_va)[:, 1]
        return float(average_precision_score(y_va, pred))

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return {"best_value": study.best_value, "best_params": study.best_params}


def tune_catboost(trials: int, train_feat, val_feat, feature_cols, y_tr, y_va) -> dict:
    import optuna
    from sklearn.metrics import average_precision_score
    from catboost import CatBoostClassifier, Pool
    from run_pipeline import _cat_indices

    cat_idx = _cat_indices(feature_cols)
    pool_tr = Pool(train_feat[feature_cols], y_tr, cat_features=cat_idx)
    pool_va = Pool(val_feat[feature_cols], y_va, cat_features=cat_idx)

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            iterations=4000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            border_count=trial.suggest_int("border_count", 32, 254),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
            auto_class_weights="Balanced",
            eval_metric="Logloss",
            random_seed=config.RANDOM_STATE,
            verbose=0,
            early_stopping_rounds=80,
        )
        model = CatBoostClassifier(**params)
        model.fit(pool_tr, eval_set=pool_va)
        pred = model.predict_proba(val_feat[feature_cols])[:, 1]
        return float(average_precision_score(y_va, pred))

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return {"best_value": study.best_value, "best_params": study.best_params}


def tune_xgboost(trials: int, train_feat, val_feat, feature_cols, y_tr, y_va) -> dict:
    import optuna
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier
    from run_pipeline import _encode_xgb_frames

    X_tr = _encode_xgb_frames(train_feat, train_feat, feature_cols)
    X_va = _encode_xgb_frames(val_feat, train_feat, feature_cols)

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=3000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 15.0),
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=80,
            tree_method="hist",
        )
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict_proba(X_va)[:, 1]
        return float(average_precision_score(y_va, pred))

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return {"best_value": study.best_value, "best_params": study.best_params}


TUNERS = {"lgbm": tune_lgbm, "catboost": tune_catboost, "xgboost": tune_xgboost}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(TUNERS))
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] preparing features...")
    train_feat, val_feat, feature_cols, y_tr, y_va = _load_features()
    print(f"[load] done in {time.time()-t0:.0f}s. n_features={len(feature_cols)}")

    print(f"[tune] model={args.model} trials={args.trials}")
    result = TUNERS[args.model](args.trials, train_feat, val_feat, feature_cols, y_tr, y_va)
    print(f"[tune] best AP = {result['best_value']:.4f}")
    print(f"[tune] best params = {json.dumps(result['best_params'], indent=2)}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / f"optuna_{args.model}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[tune] saved to {out}")
    print(f"[tune] total runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
