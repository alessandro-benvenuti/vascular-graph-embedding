from __future__ import annotations

from itertools import combinations
import hashlib

import numpy as np
import pandas as pd
from scipy import stats

from .model import PredictionRecord


CHECKPOINT_PAIRS = tuple(combinations((200, 300, 400, 500), 2))


def graph_size_normalizer(gt_nodes: int, gt_edges: int, node_weight: float = 1.0, edge_weight: float = 1.0) -> float:
    return max(float(node_weight) * int(gt_nodes) + float(edge_weight) * int(gt_edges), 1.0)


def fixed_slot_disagreement(
    record: PredictionRecord,
    node_threshold: float,
    edge_threshold: float,
    *,
    node_weight: float = 1.0,
    edge_weight: float = 1.0,
) -> dict[str, float | int]:
    """Thresholded disagreement in the original 120 RelationFormer slots."""
    predicted_nodes = record.node_probabilities >= float(node_threshold)
    target_nodes = np.zeros(len(record.node_probabilities), dtype=bool)
    target_nodes[record.matched_queries] = True
    node_disagreement = int(np.count_nonzero(predicted_nodes != target_nodes))

    query_for_gt = {int(gt): int(query) for query, gt in zip(record.matched_queries, record.matched_gt)}
    target_edges = np.zeros(len(record.edge_pairs), dtype=bool)
    pair_to_index = {tuple(map(int, pair)): index for index, pair in enumerate(record.edge_pairs)}
    for gt_left, gt_right in record.gt_edges:
        query_pair = tuple(sorted((query_for_gt[int(gt_left)], query_for_gt[int(gt_right)])))
        target_edges[pair_to_index[query_pair]] = True
    predicted_edges = (
        predicted_nodes[record.edge_pairs[:, 0]]
        & predicted_nodes[record.edge_pairs[:, 1]]
        & (record.edge_probabilities >= float(edge_threshold))
    )
    edge_disagreement = int(np.count_nonzero(predicted_edges != target_edges))
    raw = float(node_weight) * node_disagreement + float(edge_weight) * edge_disagreement
    normalizer = graph_size_normalizer(len(record.gt_nodes), len(record.gt_edges), node_weight, edge_weight)
    return {
        "binary_node_disagreement": node_disagreement,
        "binary_edge_disagreement": edge_disagreement,
        "binary_local_total_raw": raw,
        "binary_local_total_normalized": raw / normalizer,
        "gt_size_normalizer": normalizer,
    }


def rank_normalized_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame and frame[column].notna().any()]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    ranked = pd.concat(
        [frame[column].rank(method="average", pct=True, na_option="keep").rename(column) for column in available],
        axis=1,
    )
    return ranked.mean(axis=1, skipna=True)


def paired_checkpoint_rows(
    frame: pd.DataFrame,
    score_columns: list[str],
    outcome_columns: list[str],
    *,
    threshold_label: float | str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, sample in frame.groupby("sample_id", sort=True):
        by_epoch = sample.drop_duplicates("epoch").set_index("epoch")
        for epoch_a, epoch_b in CHECKPOINT_PAIRS:
            if epoch_a not in by_epoch.index or epoch_b not in by_epoch.index:
                continue
            left, right = by_epoch.loc[epoch_a], by_epoch.loc[epoch_b]
            for outcome in outcome_columns:
                if outcome not in frame or pd.isna(left[outcome]) or pd.isna(right[outcome]):
                    continue
                outcome_direction = int(np.sign(float(right[outcome]) - float(left[outcome])))
                for score in score_columns:
                    if score not in frame or pd.isna(left[score]) or pd.isna(right[score]):
                        continue
                    score_direction = int(np.sign(float(right[score]) - float(left[score])))
                    outcome_tie = outcome_direction == 0
                    score_tie = score_direction == 0
                    correct = np.nan if outcome_tie or score_tie else float(score_direction == outcome_direction)
                    half_credit = np.nan if outcome_tie else (0.5 if score_tie else float(score_direction == outcome_direction))
                    rows.append({
                        "sample_id": str(sample_id), "threshold": threshold_label,
                        "epoch_a": epoch_a, "epoch_b": epoch_b, "outcome": outcome, "score": score,
                        "outcome_a": float(left[outcome]), "outcome_b": float(right[outcome]),
                        "score_a": float(left[score]), "score_b": float(right[score]),
                        "outcome_direction": outcome_direction, "score_direction": score_direction,
                        "outcome_tie": outcome_tie, "score_tie": score_tie,
                        "correct": correct, "half_credit_correct": half_credit,
                    })
    return pd.DataFrame(rows)


def ranking_accuracy(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame()
    rows = []
    keys = ["threshold", "outcome", "score"]
    for values, group in rankings.groupby(keys, dropna=False, sort=True):
        determinate = group.correct.dropna()
        half = group.half_credit_correct.dropna()
        rows.append({
            **dict(zip(keys, values)), "pairs": len(group),
            "outcome_ties": int(group.outcome_tie.sum()), "score_ties": int(group.score_tie.sum()),
            "determinate_pairs": len(determinate),
            "accuracy": float(determinate.mean()) if len(determinate) else np.nan,
            "half_credit_accuracy": float(half.mean()) if len(half) else np.nan,
        })
    return pd.DataFrame(rows)


def sequential_gains(accuracy: pd.DataFrame, score_order: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in accuracy.groupby(["threshold", "outcome"], dropna=False, sort=True):
        values = group.set_index("score").accuracy.to_dict()
        if not all(score in values for score in score_order):
            continue
        continuous, binary, saved, optimized = [float(values[score]) for score in score_order]
        rows.append({
            "threshold": keys[0], "outcome": keys[1],
            "thresholding_gain": binary - continuous,
            "edit_accounting_gain": saved - binary,
            "correspondence_optimization_gain": optimized - saved,
            "total_ged_gain": optimized - continuous,
        })
    return pd.DataFrame(rows)


def patch_bootstrap(
    rankings: pd.DataFrame,
    score_order: list[str],
    *,
    draws: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for keys, group in rankings.groupby(["threshold", "outcome"], dropna=False, sort=True):
        patches = sorted(group.sample_id.astype(str).unique())
        if not patches:
            continue
        rng = np.random.default_rng(seed + int.from_bytes(hashlib.sha256(str(keys).encode()).digest()[:4], "big"))
        chosen = rng.integers(len(patches), size=(int(draws), len(patches)))
        draw_values = {}
        for score in score_order:
            sums = np.zeros(len(patches)); counts = np.zeros(len(patches))
            score_rows = group[group.score == score]
            for index, patch in enumerate(patches):
                values = score_rows.loc[score_rows.sample_id.astype(str) == patch, "correct"].dropna().to_numpy(float)
                sums[index] = values.sum(); counts[index] = len(values)
            draw_sums = sums[chosen].sum(axis=1); draw_counts = counts[chosen].sum(axis=1)
            draw_values[score] = np.divide(draw_sums, draw_counts, out=np.full(int(draws), np.nan), where=draw_counts > 0)
        continuous, binary, saved, optimized = [draw_values[score] for score in score_order]
        gain_values = {
            "thresholding_gain": binary - continuous,
            "edit_accounting_gain": saved - binary,
            "correspondence_optimization_gain": optimized - saved,
            "total_ged_gain": optimized - continuous,
        }
        for name, values in {**draw_values, **gain_values}.items():
            clean = np.asarray(values, dtype=float); clean = clean[np.isfinite(clean)]
            rows.append({
                "threshold": keys[0], "outcome": keys[1], "estimand": name, "draws": len(clean),
                "low": float(np.quantile(clean, .025)) if len(clean) else np.nan,
                "high": float(np.quantile(clean, .975)) if len(clean) else np.nan,
            })
    return pd.DataFrame(rows)


def grouped_cv_extended(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    group: str,
    *,
    folds: int = 5,
    seed: int = 0,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    columns = [target, group, *predictors]
    clean = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    groups = sorted(clean[group].astype(str).unique())
    folds = min(int(folds), len(groups))
    if folds < 2:
        return {"n": len(clean), "groups": len(groups), "folds": folds, "r2": np.nan, "mae": np.nan, "spearman": np.nan}, pd.DataFrame()
    assignment = {value: int.from_bytes(hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "big") % folds for value in groups}
    clean["fold"] = clean[group].astype(str).map(assignment)
    predictions = pd.Series(np.nan, index=clean.index, dtype=float)
    for fold in range(folds):
        test = clean.fold == fold; train = ~test
        if not test.any() or train.sum() <= len(predictors):
            continue
        x_train = np.column_stack((np.ones(train.sum()), clean.loc[train, predictors].to_numpy(float)))
        coefficients = np.linalg.lstsq(x_train, clean.loc[train, target].to_numpy(float), rcond=None)[0]
        predictions.loc[test] = np.column_stack((np.ones(test.sum()), clean.loc[test, predictors].to_numpy(float))) @ coefficients
    output = clean.loc[predictions.notna()].copy()
    output["predicted"] = predictions.loc[output.index]
    output["residual"] = output[target] - output.predicted
    if output.empty:
        return {"n": 0, "groups": len(groups), "folds": folds, "r2": np.nan, "mae": np.nan, "spearman": np.nan}, output
    residual = output.residual.to_numpy(float); observed = output[target].to_numpy(float)
    denominator = np.square(observed - observed.mean()).sum()
    return {
        "n": len(output), "groups": len(groups), "folds": folds,
        "r2": float(1 - np.square(residual).sum() / denominator) if denominator > 0 else np.nan,
        "mae": float(np.abs(residual).mean()),
        "spearman": (
            float(stats.spearmanr(observed, output.predicted).statistic)
            if len(output) > 1 and np.unique(observed).size > 1 and output.predicted.nunique() > 1 else np.nan
        ),
    }, output


def multithreshold_aggregate(frame: pd.DataFrame, scores: list[str], outcomes: list[str]) -> pd.DataFrame:
    columns = [column for column in [*scores, *outcomes] if column in frame]
    grouped = frame.groupby(["sample_id", "epoch"], as_index=False)[columns].mean()
    grouped["threshold"] = "multithreshold"
    return grouped


def threshold_stability(frame: pd.DataFrame, scores: list[str]) -> pd.DataFrame:
    rows = []
    thresholds = sorted(frame.node_threshold.unique())
    for score in scores:
        pivot = frame.pivot_table(index=["sample_id", "epoch"], columns="node_threshold", values=score, aggfunc="first")
        for left, right in combinations(thresholds, 2):
            clean = pivot[[left, right]].dropna()
            can_correlate = len(clean) > 1 and clean[left].nunique() > 1 and clean[right].nunique() > 1
            rows.append({
                "record_type": "rank_correlation", "score": score, "threshold_a": left, "threshold_b": right,
                "spearman": float(stats.spearmanr(clean[left], clean[right]).statistic) if can_correlate else np.nan,
                "kendall_tau": float(stats.kendalltau(clean[left], clean[right]).statistic) if can_correlate else np.nan,
                "n": len(clean),
            })
        for (sample_id, epoch), values in pivot.iterrows():
            clean = values.dropna(); score_range = float(clean.max() - clean.min())
            rows.append({
                "record_type": "magnitude", "score": score, "sample_id": sample_id, "epoch": epoch,
                "score_range": score_range, "relative_score_range": score_range / max(abs(float(clean.median())), 1e-12),
            })
        flips = 0; determinate = 0
        for sample_id, sample in frame.groupby("sample_id"):
            for epoch_a, epoch_b in CHECKPOINT_PAIRS:
                directions = []
                for threshold in thresholds:
                    subset = sample[(sample.epoch.isin([epoch_a, epoch_b])) & (sample.node_threshold == threshold)].set_index("epoch")
                    if epoch_a in subset.index and epoch_b in subset.index:
                        directions.append(int(np.sign(float(subset.loc[epoch_b, score]) - float(subset.loc[epoch_a, score]))))
                nonzero = {value for value in directions if value != 0}
                if nonzero:
                    flipped = int(len(nonzero) > 1); determinate += 1; flips += flipped
                    rows.append({
                        "record_type": "checkpoint_pair_flip", "score": score, "sample_id": sample_id,
                        "epoch_a": epoch_a, "epoch_b": epoch_b, "checkpoint_order_flip": bool(flipped),
                        "directions": ",".join(map(str, directions)),
                    })
        rows.append({"record_type": "checkpoint_flip", "score": score, "flip_rate": flips / determinate if determinate else np.nan, "n": determinate})
    return pd.DataFrame(rows)


def mapping_change_count(saved: list[list[int]], optimized: list[list[int]]) -> int:
    return len({tuple(map(int, pair)) for pair in saved} ^ {tuple(map(int, pair)) for pair in optimized})
