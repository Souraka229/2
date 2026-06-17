"""Features graphe biparti origin -> destination (non supervisées, train+test)."""
from __future__ import annotations

import pandas as pd

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


def _build_directed_graph(ref: pd.DataFrame) -> "nx.DiGraph":
    edges = (
        ref.groupby(["origin_account", "destination_account"])
        .agg(weight=("amount", "sum"), count=("amount", "count"))
        .reset_index()
    )
    g = nx.DiGraph()
    for row in edges.itertuples(index=False):
        g.add_edge(
            row.origin_account,
            row.destination_account,
            weight=float(row.weight),
            count=int(row.count),
        )
    return g


def fit_graph_features(ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stats par compte apprises sur ref (train ou train+test)."""
    origin_out = ref.groupby("origin_account").agg(
        o_out_degree=("destination_account", "nunique"),
        o_out_count=("amount", "count"),
        o_out_amount_sum=("amount", "sum"),
        o_out_amount_mean=("amount", "mean"),
        o_period_span=("period", lambda s: s.max() - s.min() + 1),
    )
    origin_out["o_fan_out_rate"] = origin_out["o_out_degree"] / origin_out["o_period_span"].clip(lower=1)

    dest_in = ref.groupby("destination_account").agg(
        d_in_degree=("origin_account", "nunique"),
        d_in_count=("amount", "count"),
        d_in_amount_sum=("amount", "sum"),
        d_in_amount_mean=("amount", "mean"),
        d_period_span=("period", lambda s: s.max() - s.min() + 1),
    )
    dest_in["d_fan_in_rate"] = dest_in["d_in_degree"] / dest_in["d_period_span"].clip(lower=1)

    pairs = (
        ref.groupby(["origin_account", "destination_account"])
        .size()
        .reset_index(name="cnt")
    )
    rev_map = {
        (row.destination_account, row.origin_account): row.cnt
        for row in pairs.itertuples(index=False)
    }
    pairs["rev_cnt"] = [
        rev_map.get((row.origin_account, row.destination_account), 0)
        for row in pairs.itertuples(index=False)
    ]
    pairs["cycle2"] = pairs["cnt"] + pairs["rev_cnt"]
    pairs = pairs[pairs["origin_account"] < pairs["destination_account"]]
    cycle_origin = pairs.groupby("origin_account")["cycle2"].sum().rename("o_cycle2_count")
    cycle_dest = pairs.groupby("destination_account")["cycle2"].sum().rename("d_cycle2_count")

    origin_out = origin_out.join(cycle_origin, how="left").fillna(0)
    dest_in = dest_in.join(cycle_dest, how="left").fillna(0)

    if nx is not None and len(ref) > 0:
        g = _build_directed_graph(ref)
        try:
            pr = nx.pagerank(g, weight="weight", max_iter=50)
        except Exception:
            pr = {n: 0.0 for n in g.nodes}
        origin_out["o_pagerank"] = origin_out.index.map(pr).fillna(0)
        dest_in["d_pagerank"] = dest_in.index.map(pr).fillna(0)
    else:
        origin_out["o_pagerank"] = 0.0
        dest_in["d_pagerank"] = 0.0

    return origin_out, dest_in


def apply_graph_features(
    df: pd.DataFrame,
    origin_stats: pd.DataFrame,
    dest_stats: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    saved_index = out.index
    out = out.merge(origin_stats, left_on="origin_account", right_index=True, how="left")
    out = out.merge(
        dest_stats,
        left_on="destination_account",
        right_index=True,
        how="left",
        suffixes=("", "_dest"),
    )
    out.index = saved_index
    graph_cols = [c for c in out.columns if c.startswith(("o_", "d_"))]
    out[graph_cols] = out[graph_cols].fillna(0)
    out["graph_pagerank_gap"] = (
        out["o_pagerank"].rank(pct=True) - out["d_pagerank"].rank(pct=True)
    ).abs().fillna(0)
    return out
