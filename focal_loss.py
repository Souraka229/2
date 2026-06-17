"""Focal loss pour LightGBM (objective custom).

FL(p,y) = -alpha * (1-p)^gamma * y*log(p) - (1-alpha) * p^gamma * (1-y)*log(1-p)

Gradient et hessian implementes pour usage via LGBMClassifier(objective=...).
Avec scale_pos_weight=1 et alpha=0.25, gamma=2 : valeurs canoniques de Lin et al.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def make_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Retourne un callable (y_pred_raw, dataset) -> (grad, hess) compatible LightGBM."""

    def focal_loss_obj(y_pred_raw, dataset):
        y_true = dataset.get_label()
        p = _sigmoid(y_pred_raw)

        # Gradient et hessian de la focal loss binaire vs raw_score x.
        # On utilise les formules stables :
        #   loss = -[alpha*y*(1-p)^g*log(p) + (1-alpha)*(1-y)*p^g*log(1-p)]
        a_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        p_t = np.where(y_true == 1, p, 1.0 - p)
        y_signed = np.where(y_true == 1, 1.0, -1.0)

        # dloss/dx
        log_p_t = np.log(np.clip(p_t, 1e-9, 1.0))
        term = gamma * (1.0 - p_t) * log_p_t - p_t
        grad = -a_t * y_signed * (1.0 - p_t) ** gamma * term  # vs raw x (dim 1)
        # Sign: For y=1, dloss/dx negative when p close to 1. Approx hessian:
        hess = a_t * (1.0 - p_t) ** gamma * (
            gamma * (1.0 - p_t) * (1.0 - log_p_t * gamma * p_t) + p_t * (1.0 - p_t)
        )
        hess = np.clip(hess, 1e-6, None)
        return grad, hess

    return focal_loss_obj


def focal_loss_eval(gamma: float = 2.0, alpha: float = 0.25):
    """Eval metric matching focal_loss_obj, retournee comme (name, value, is_higher_better)."""

    def _eval(y_pred_raw, dataset):
        y_true = dataset.get_label()
        p = _sigmoid(y_pred_raw)
        a_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        p_t = np.where(y_true == 1, p, 1.0 - p)
        loss = -a_t * (1.0 - p_t) ** gamma * np.log(np.clip(p_t, 1e-9, 1.0))
        return "focal_loss", float(loss.mean()), False

    return _eval
