"""Features temporelles fines — rolling windows, inter-event time, bursts.

Toutes calculees avec shift(1) PAR compte pour rester strictement causales :
chaque ligne ne voit que ses propres transactions PRECEDENTES (meme periode +
meme id inclus dans le shift apres tri stable). Train precede test (train
period max = 105, test min = 106) → pas de fuite future train-vers-test.

Convention : on calcule cote origin_account ET cote destination_account
separement, puis on prefixe `o_` / `d_`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ID_COL = "id"
WINDOWS = (5, 20)


def _sorted_combined(combined: pd.DataFrame) -> pd.DataFrame:
    if ID_COL in combined.columns:
        return combined.sort_values(["period", ID_COL], kind="mergesort")
    return combined.assign(_k=combined.index).sort_values(["period", "_k"], kind="mergesort").drop(columns="_k")


def _rolling_features_per_account(
    df: pd.DataFrame,
    account_col: str,
    prefix: str,
    windows: tuple[int, ...] = WINDOWS,
) -> pd.DataFrame:
    """Pour chaque ligne : stats glissantes sur les `w` transactions PRECEDENTES
    du meme compte, plus inter-event time et burst flags.
    """
    g = df.groupby(account_col, sort=False)

    out = pd.DataFrame(index=df.index)

    # Inter-event : difference de period avec la tx precedente du compte.
    prev_period = g["period"].shift(1)
    out[f"{prefix}_dt_prev"] = (df["period"] - prev_period).fillna(-1).astype(np.float32)
    # 2eme et 3eme precedentes
    prev2 = g["period"].shift(2)
    out[f"{prefix}_dt_prev2"] = (df["period"] - prev2).fillna(-1).astype(np.float32)

    # Amount rolling sur les fenetres
    amount_shift = g["amount"].shift(1)
    for w in windows:
        roll = amount_shift.groupby(df[account_col], sort=False).rolling(w, min_periods=1)
        out[f"{prefix}_amt_mean_w{w}"] = roll.mean().reset_index(level=0, drop=True).astype(np.float32)
        out[f"{prefix}_amt_std_w{w}"] = roll.std().reset_index(level=0, drop=True).fillna(0).astype(np.float32)
        out[f"{prefix}_amt_max_w{w}"] = roll.max().reset_index(level=0, drop=True).astype(np.float32)
        out[f"{prefix}_amt_min_w{w}"] = roll.min().reset_index(level=0, drop=True).astype(np.float32)
        out[f"{prefix}_amt_sum_w{w}"] = roll.sum().reset_index(level=0, drop=True).astype(np.float32)

    # z-score dynamique : (amount - mean_w20) / (std_w20 + eps), clip pour stabilite.
    mu = out[f"{prefix}_amt_mean_w20"]
    sd = out[f"{prefix}_amt_std_w20"]
    z = (df["amount"] - mu) / (sd + 1.0)
    out[f"{prefix}_amt_zscore_w20"] = z.clip(-50, 50).astype(np.float32)

    # Ratio par rapport au max precedent (escalade montant), clip stable.
    ratio = df["amount"] / (out[f"{prefix}_amt_max_w20"].clip(lower=1.0))
    out[f"{prefix}_amt_ratio_max20"] = ratio.clip(0, 1e4).astype(np.float32)

    # Burst : nombre de tx du meme compte dans la fenetre [period-2, period-1]
    # → utile pour detecter une rafale.
    period_shift = g["period"].shift(1)
    same_account_recent = ((df["period"] - period_shift) <= 2).fillna(False).astype(np.int8)
    out[f"{prefix}_burst2"] = same_account_recent

    # Nb distinct counterparties rolling : couteux. On approxime par
    # cumulative distinct count - shift(prev_w cumcount).
    # On garde simple : flag "premiere apparition".
    cumcount = g.cumcount()
    out[f"{prefix}_is_first"] = (cumcount == 0).astype(np.int8)
    out[f"{prefix}_rank"] = cumcount.astype(np.float32)

    return out


def add_rolling_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Retourne un DataFrame indexe comme combined avec les features rolling
    cote origin et cote dest. NaN remplaces par -1 (sentinelle "premiere tx").
    """
    sorted_df = _sorted_combined(combined)
    o = _rolling_features_per_account(sorted_df, "origin_account", "o_roll", WINDOWS)
    d = _rolling_features_per_account(sorted_df, "destination_account", "d_roll", WINDOWS)
    feats = pd.concat([o, d], axis=1).fillna(-1.0).astype(np.float32)
    return feats.reindex(combined.index)
