"""Stacking : meta-modele niveau-2 sur predictions OOF des base-models.

Genere des OOF preds via CV temporelle, puis entraine un meta-LR / meta-LGBM.
Le meta-modele apprend a re-ponderer les base models par regions du feature
space.

Usage typique:
    oof = generate_oof_predictions(train, cutoffs=[75, 81, 87, 93, 99])
    meta = train_meta(oof[base_models], oof['target'])
    test_preds = predict_with_meta(meta, base_preds_on_test)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def train_meta_lr(oof_preds: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """Logistic regression L2 sur les logit(p) des base models. Plus robuste
    que sur les probas directes.
    """
    eps = 1e-6
    logits = np.log(oof_preds.clip(eps, 1 - eps) / (1 - oof_preds.clip(eps, 1 - eps)))
    meta = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, n_jobs=-1, random_state=42,
    )
    meta.fit(logits, y)
    return meta


def predict_with_meta_lr(meta: LogisticRegression, base_preds: pd.DataFrame) -> np.ndarray:
    eps = 1e-6
    logits = np.log(base_preds.clip(eps, 1 - eps) / (1 - base_preds.clip(eps, 1 - eps)))
    return meta.predict_proba(logits)[:, 1]


def train_meta_lgbm(oof_preds: pd.DataFrame, y: pd.Series):
    """Meta LightGBM peu profond. Capture les interactions entre base models."""
    from lightgbm import LGBMClassifier

    pos = float(y.sum())
    neg = float(len(y) - pos)
    scale = neg / max(pos, 1)
    meta = LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=200,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    meta.fit(oof_preds, y)
    return meta


def predict_with_meta_lgbm(meta, base_preds: pd.DataFrame) -> np.ndarray:
    return meta.predict_proba(base_preds)[:, 1]


def optimize_blend_weights(
    oof_preds: pd.DataFrame, y: pd.Series, restarts: int = 5, seed: int = 42,
) -> dict[str, float]:
    """Nelder-Mead sous contrainte: poids >= 0, somme = 1. Optimise AP directement."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    cols = list(oof_preds.columns)
    n = len(cols)
    preds = oof_preds.values

    def neg_ap(w_raw: np.ndarray) -> float:
        w = np.abs(w_raw)
        s = w.sum()
        if s < 1e-9:
            return 0.0
        w = w / s
        blend = (preds * w).sum(axis=1)
        return -float(average_precision_score(y, blend))

    best_score = np.inf
    best_w: dict[str, float] = {}
    for _ in range(restarts):
        w0 = rng.dirichlet(np.ones(n))
        res = minimize(neg_ap, w0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5})
        if res.fun < best_score:
            best_score = res.fun
            w = np.abs(res.x)
            w = w / w.sum()
            best_w = {c: float(v) for c, v in zip(cols, w)}
    return best_w
