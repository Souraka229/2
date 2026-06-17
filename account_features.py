"""Features account-level — profils, deviations, isolation forest, frequence.

Toutes calculees sur train+test (non supervisees → pas de fuite).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


EPS = 1e-6


def fit_account_profiles(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profil quantile + diversite operation par compte.

    Retourne (origin_profile, dest_profile) indexes par account.
    """
    def _profile(df: pd.DataFrame, account_col: str, prefix: str) -> pd.DataFrame:
        g = df.groupby(account_col, observed=True)
        prof = pd.DataFrame({
            f"{prefix}_amt_q25": g["amount"].quantile(0.25),
            f"{prefix}_amt_q50": g["amount"].quantile(0.50),
            f"{prefix}_amt_q75": g["amount"].quantile(0.75),
            f"{prefix}_amt_freq": g.size().astype(np.float32),
            f"{prefix}_op_nunique": g["operation"].nunique().astype(np.float32),
        })
        prof[f"{prefix}_amt_iqr"] = (prof[f"{prefix}_amt_q75"] - prof[f"{prefix}_amt_q25"]).clip(lower=0)
        # Part de chaque type d'op pour ce compte
        op_share = (
            df.groupby([account_col, "operation"], observed=True).size().unstack(fill_value=0)
        )
        op_share = op_share.div(op_share.sum(axis=1).clip(lower=1), axis=0)
        op_share.columns = [f"{prefix}_share_{c}" for c in op_share.columns]
        prof = prof.join(op_share, how="left").fillna(0)
        return prof

    origin_profile = _profile(combined, "origin_account", "op_orig")
    dest_profile = _profile(combined, "destination_account", "op_dest")
    return origin_profile, dest_profile


def apply_account_profiles(
    df: pd.DataFrame,
    origin_profile: pd.DataFrame,
    dest_profile: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    saved_index = out.index
    out = out.merge(origin_profile, left_on="origin_account", right_index=True, how="left")
    out = out.merge(dest_profile, left_on="destination_account", right_index=True, how="left")
    out.index = saved_index

    profile_cols = list(origin_profile.columns) + list(dest_profile.columns)
    for c in profile_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    # Deviations courantes : amount vs profil compte
    out["amt_vs_orig_q50"] = (df["amount"] / (out["op_orig_amt_q50"] + 1.0)).clip(0, 1e4)
    out["amt_vs_dest_q50"] = (df["amount"] / (out["op_dest_amt_q50"] + 1.0)).clip(0, 1e4)
    out["amt_orig_iqr_z"] = (
        (df["amount"] - out["op_orig_amt_q50"]) / (out["op_orig_amt_iqr"] + 1.0)
    ).clip(-100, 100)
    out["amt_dest_iqr_z"] = (
        (df["amount"] - out["op_dest_amt_q50"]) / (out["op_dest_amt_iqr"] + 1.0)
    ).clip(-100, 100)

    # Log-counts pour stabilite GBDT
    out["op_orig_amt_freq_log1p"] = np.log1p(out["op_orig_amt_freq"]).astype(np.float32)
    out["op_dest_amt_freq_log1p"] = np.log1p(out["op_dest_amt_freq"]).astype(np.float32)

    return out


def fit_isolation_forest(
    combined: pd.DataFrame,
    feature_cols: list[str],
    subsample: int = 200_000,
    random_state: int = 42,
) -> IsolationForest:
    """Entraine IsolationForest sur un sous-echantillon stratifie de combined."""
    n = len(combined)
    if n > subsample:
        sample = combined.sample(subsample, random_state=random_state)
    else:
        sample = combined
    iso = IsolationForest(
        n_estimators=200,
        max_samples=min(256, len(sample)),
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(sample[feature_cols].astype(np.float32))
    return iso


def score_isolation(iso: IsolationForest, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Score d'anomalie : plus negatif = plus anormal."""
    return iso.score_samples(df[feature_cols].astype(np.float32)).astype(np.float32)


ISO_FEATURES = [
    "amount",
    "amount_log1p",
    "origin_balance_before",
    "origin_balance_after",
    "destination_balance_before",
    "destination_balance_after",
    "origin_balance_mismatch",
    "dest_balance_mismatch",
    "amount_to_origin_before",
    "amount_to_destination_before",
]
