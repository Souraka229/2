"""Feature engineering sans fuite de cible vers le test."""
from __future__ import annotations

import numpy as np
import pandas as pd

from graph_features import apply_graph_features, fit_graph_features
from temporal_features import add_rolling_features
from account_features import apply_account_profiles


EPS = 1e-6


def add_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amount_log1p"] = np.log1p(np.maximum(out["amount"], 0))
    out["amount_sqrt"] = np.sqrt(np.maximum(out["amount"], 0))

    out["origin_balance_change"] = out["origin_balance_after"] - out["origin_balance_before"]
    out["destination_balance_change"] = (
        out["destination_balance_after"] - out["destination_balance_before"]
    )

    out["origin_expected_debit"] = out["origin_balance_before"] - out["origin_balance_after"]
    out["dest_expected_credit"] = out["destination_balance_after"] - out["destination_balance_before"]

    out["origin_balance_mismatch"] = np.abs(out["origin_expected_debit"] - out["amount"])
    out["dest_balance_mismatch"] = np.abs(out["dest_expected_credit"] - out["amount"])

    out["amount_to_origin_before"] = out["amount"] / (np.abs(out["origin_balance_before"]) + EPS)
    out["amount_to_destination_before"] = out["amount"] / (
        np.abs(out["destination_balance_before"]) + EPS
    )
    out["amount_to_origin_after"] = out["amount"] / (np.abs(out["origin_balance_after"]) + EPS)
    out["amount_to_destination_after"] = out["amount"] / (
        np.abs(out["destination_balance_after"]) + EPS
    )

    out["origin_negative_before"] = (out["origin_balance_before"] < 0).astype(np.int8)
    out["origin_negative_after"] = (out["origin_balance_after"] < 0).astype(np.int8)
    out["dest_negative_before"] = (out["destination_balance_before"] < 0).astype(np.int8)
    out["dest_negative_after"] = (out["destination_balance_after"] < 0).astype(np.int8)

    out["origin_unchanged"] = (out["origin_balance_change"] == 0).astype(np.int8)
    out["dest_unchanged"] = (out["destination_balance_change"] == 0).astype(np.int8)
    out["origin_wiped"] = (out["origin_balance_after"] <= 0).astype(np.int8)
    out["dest_empty_before"] = (out["destination_balance_before"] == 0).astype(np.int8)
    out["dest_empty_after"] = (out["destination_balance_after"] == 0).astype(np.int8)

    out["origin_balance_ratio"] = out["origin_balance_after"] / (
        np.abs(out["origin_balance_before"]) + EPS
    )
    out["dest_balance_ratio"] = out["destination_balance_after"] / (
        np.abs(out["destination_balance_before"]) + EPS
    )

    out["total_balance_before"] = out["origin_balance_before"] + out["destination_balance_before"]
    out["total_balance_after"] = out["origin_balance_after"] + out["destination_balance_after"]
    out["balance_sum_change"] = out["total_balance_after"] - out["total_balance_before"]

    out["same_account"] = (
        out["origin_account"] == out["destination_account"]
    ).astype(np.int8)

    return out


def _smooth_rate(count: pd.Series, rate: pd.Series, global_mean: float, m: float) -> pd.Series:
    return (rate * count + global_mean * m) / (count + m)


def fit_target_encoding(
    train_df: pd.DataFrame,
    target: pd.Series,
    cols: list[str],
    smoothing: float,
) -> dict[str, pd.Series]:
    global_mean = float(target.mean())
    maps: dict[str, pd.Series] = {"__global__": global_mean}
    y = target.values
    for col in cols:
        tmp = pd.DataFrame({"k": train_df[col].values, "y": y})
        agg = tmp.groupby("k")["y"].agg(["mean", "count"])
        maps[col] = _smooth_rate(agg["count"], agg["mean"], global_mean, smoothing)
    return maps


def compute_oof_target_encoding(
    train_df: pd.DataFrame,
    target: pd.Series,
    cols: list[str],
    smoothing: float,
    n_splits: int = 5,
    groups: pd.Series | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Target encoding out-of-fold pour le train + maps full-fit pour le test.

    - Le train reçoit pour chaque ligne une valeur calculée sur les K-1 autres folds.
      Aucune ligne ne voit sa propre cible -> pas de fuite vers le validation.
    - Les maps retournees sont apprises sur tout le train, à appliquer telles
      quelles sur le test (pas de cible côté test, donc pas de fuite possible).
    - `groups` (ex: origin_account) -> StratifiedGroupKFold pour eviter qu'un
      compte soit a la fois en train et en val d'un meme fold.
    """
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

    global_mean = float(target.mean())
    oof_cols = {c: np.full(len(train_df), global_mean, dtype=np.float32) for c in cols}

    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(train_df, target, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(train_df, target)

    y_arr = target.values
    for fold_idx, (tr_idx, va_idx) in enumerate(split_iter):
        fold_train = train_df.iloc[tr_idx]
        fold_y = y_arr[tr_idx]
        fold_mean = float(fold_y.mean())
        for col in cols:
            tmp = pd.DataFrame({"k": fold_train[col].values, "y": fold_y})
            agg = tmp.groupby("k")["y"].agg(["mean", "count"])
            fold_map = _smooth_rate(agg["count"], agg["mean"], fold_mean, smoothing)
            mapped = train_df.iloc[va_idx][col].map(fold_map).fillna(fold_mean).values
            oof_cols[col][va_idx] = mapped

    oof_df = pd.DataFrame(
        {f"{c}_fraud_rate": oof_cols[c] for c in cols},
        index=train_df.index,
    )

    full_maps = fit_target_encoding(train_df, target, cols, smoothing)
    return oof_df, full_maps


def apply_target_encoding(df: pd.DataFrame, maps: dict[str, pd.Series], cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    global_mean = maps["__global__"]
    for col in cols:
        out[f"{col}_fraud_rate"] = df[col].map(maps[col]).fillna(global_mean)
    return out


def add_unsupervised_aggregations(
    df: pd.DataFrame,
    ref: pd.DataFrame,
) -> pd.DataFrame:
    """Comptages et stats non supervisées calculées sur ref (train ou train+test)."""
    out = df.copy()

    origin_stats = ref.groupby("origin_account").agg(
        origin_txn_count=("amount", "count"),
        origin_amount_sum=("amount", "sum"),
        origin_amount_mean=("amount", "mean"),
        origin_amount_std=("amount", "std"),
        origin_period_min=("period", "min"),
        origin_period_max=("period", "max"),
        origin_dest_nunique=("destination_account", "nunique"),
    )
    dest_stats = ref.groupby("destination_account").agg(
        dest_txn_count=("amount", "count"),
        dest_amount_sum=("amount", "sum"),
        dest_amount_mean=("amount", "mean"),
        dest_amount_std=("amount", "std"),
        dest_period_min=("period", "min"),
        dest_period_max=("period", "max"),
        dest_origin_nunique=("origin_account", "nunique"),
    )

    op_stats = ref.groupby("operation").agg(
        op_count=("amount", "count"),
        op_amount_mean=("amount", "mean"),
        op_amount_std=("amount", "std"),
        op_amount_median=("amount", "median"),
    )

    pair_stats = (
        ref.groupby(["origin_account", "destination_account"])
        .size()
        .rename("pair_count")
        .reset_index()
    )

    saved_index = out.index
    out = out.merge(origin_stats, left_on="origin_account", right_index=True, how="left")
    out = out.merge(dest_stats, left_on="destination_account", right_index=True, how="left")
    out = out.merge(op_stats, left_on="operation", right_index=True, how="left")

    out = out.merge(
        pair_stats,
        on=["origin_account", "destination_account"],
        how="left",
    )
    out.index = saved_index
    out["pair_count"] = out["pair_count"].fillna(0)

    out["origin_amount_std"] = out["origin_amount_std"].fillna(0)
    out["dest_amount_std"] = out["dest_amount_std"].fillna(0)
    out["op_amount_std"] = out["op_amount_std"].fillna(0)

    out["origin_txn_count"] = out["origin_txn_count"].fillna(0)
    out["dest_txn_count"] = out["dest_txn_count"].fillna(0)
    out["pair_count"] = out["pair_count"].fillna(0)

    out["origin_dest_ratio"] = out["origin_txn_count"] / (out["dest_txn_count"] + 1)
    out["amount_vs_origin_mean"] = out["amount"] / (out["origin_amount_mean"].fillna(0) + EPS)
    out["amount_vs_dest_mean"] = out["amount"] / (out["dest_amount_mean"].fillna(0) + EPS)
    out["amount_vs_op_mean"] = out["amount"] / (out["op_amount_mean"].fillna(0) + EPS)

    out["origin_period_span"] = out["origin_period_max"] - out["origin_period_min"]
    out["dest_period_span"] = out["dest_period_max"] - out["dest_period_min"]

    return out


def add_expanding_temporal_features(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Features temporelles expanding avec shift(1) sur données triées par period, id.
    Train (period<=105) précède test (period>=106) — pas de fuite future.
    """
    out = combined.copy()
    if ID_COL in out.columns:
        out = out.sort_values(["period", ID_COL])
    else:
        out = out.assign(_sort_key=out.index).sort_values(["period", "_sort_key"]).drop(
            columns="_sort_key"
        )

    for prefix, account_col in [
        ("origin", "origin_account"),
        ("dest", "destination_account"),
    ]:
        g = out.groupby(account_col, sort=False)
        out[f"{prefix}_txn_rank"] = g.cumcount()
        out[f"{prefix}_amount_cumsum_prior"] = g["amount"].transform(
            lambda s: s.cumsum().shift(1)
        )
        out[f"{prefix}_amount_cummean_prior"] = g["amount"].transform(
            lambda s: s.expanding().mean().shift(1)
        )
        out[f"{prefix}_amount_cummax_prior"] = g["amount"].transform(
            lambda s: s.expanding().max().shift(1)
        )
        out[f"{prefix}_period_first"] = g["period"].transform("first")
        out[f"{prefix}_period_last_prior"] = g["period"].transform(
            lambda s: s.shift(1).ffill()
        )
        out[f"{prefix}_period_since_last"] = out["period"] - out[f"{prefix}_period_last_prior"]
        out[f"{prefix}_period_since_first"] = out["period"] - out[f"{prefix}_period_first"]

    fill_cols = [c for c in out.columns if c.endswith("_prior") or c.endswith("_since_last")]
    out[fill_cols] = out[fill_cols].fillna(0)
    return out.reindex(combined.index)


def encode_operation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["operation_code"] = out["operation"].str.extract(r"(\d+)").astype(float).fillna(0)
    out["period_sin"] = np.sin(2 * np.pi * out["period"] / 144)
    out["period_cos"] = np.cos(2 * np.pi * out["period"] / 144)
    return out


ID_COL = "id"


def build_features(
    df: pd.DataFrame,
    *,
    ref_unsupervised: pd.DataFrame,
    te_maps: dict[str, pd.Series] | None = None,
    te_cols: list[str] | None = None,
    te_oof: pd.DataFrame | None = None,
    expanding_df: pd.DataFrame | None = None,
    graph_origin_stats: pd.DataFrame | None = None,
    graph_dest_stats: pd.DataFrame | None = None,
    rolling_df: pd.DataFrame | None = None,
    account_origin_profile: pd.DataFrame | None = None,
    account_dest_profile: pd.DataFrame | None = None,
    iso_score: pd.Series | None = None,
) -> pd.DataFrame:
    out = add_balance_features(df)
    out = encode_operation(out)
    out = add_unsupervised_aggregations(out, ref_unsupervised)

    if graph_origin_stats is not None and graph_dest_stats is not None:
        out = apply_graph_features(out, graph_origin_stats, graph_dest_stats)

    if account_origin_profile is not None and account_dest_profile is not None:
        out = apply_account_profiles(out, account_origin_profile, account_dest_profile)

    if iso_score is not None:
        out["iso_anomaly_score"] = iso_score.reindex(df.index).astype(np.float32).values

    if te_oof is not None:
        out = out.join(te_oof)
    elif te_maps is not None and te_cols:
        out = apply_target_encoding(out, te_maps, te_cols)

    if rolling_df is not None:
        out = out.join(rolling_df.loc[df.index])

    if expanding_df is not None:
        exp_cols = [
            c
            for c in expanding_df.columns
            if c.startswith(("origin_", "dest_"))
            and c
            not in {
                "origin_account",
                "destination_account",
                "origin_balance_before",
                "origin_balance_after",
                "destination_balance_before",
                "destination_balance_after",
            }
            and c in expanding_df.columns
            and c not in out.columns
        ]
        # colonnes expanding spécifiques
        keep = [
            "origin_txn_rank",
            "origin_amount_cumsum_prior",
            "origin_amount_cummean_prior",
            "origin_amount_cummax_prior",
            "origin_period_since_last",
            "origin_period_since_first",
            "dest_txn_rank",
            "dest_amount_cumsum_prior",
            "dest_amount_cummean_prior",
            "dest_amount_cummax_prior",
            "dest_period_since_last",
            "dest_period_since_first",
        ]
        keep = [c for c in keep if c in expanding_df.columns]
        out = out.join(expanding_df[keep])

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {
        ID_COL,
        "fraud_flag",
        "origin_account",
        "destination_account",
    }
    return [c for c in df.columns if c not in drop]
