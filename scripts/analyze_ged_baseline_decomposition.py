#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml

from vascular_graph_embedding.data.storage import atomic_json
from vascular_graph_embedding.ged.decomposition import (
    fixed_slot_disagreement, graph_size_normalizer, grouped_cv_extended,
    mapping_change_count, multithreshold_aggregate, paired_checkpoint_rows,
    patch_bootstrap, ranking_accuracy, rank_normalized_mean, sequential_gains,
    threshold_stability,
)
from vascular_graph_embedding.ged.loader import load_edge_pairs, parse_record, torch_load
from vascular_graph_embedding.ged.metrics import graph_summary
from vascular_graph_embedding.ged.model import HardGraph
from vascular_graph_embedding.ged.statistics import correlation


SCORES = [
    "continuous_total_score", "binary_local_total_normalized",
    "saved_matching_cost_normalized", "optimized_ged_normalized",
]
SCORE_LABELS = {
    "continuous_total_score": "continuous RF proxy",
    "binary_local_total_normalized": "binary local",
    "saved_matching_cost_normalized": "saved-matching edit",
    "optimized_ged_normalized": "optimized GED",
}
COLORS = {
    "continuous_total_score": "#4c78a8", "binary_local_total_normalized": "#f58518",
    "saved_matching_cost_normalized": "#54a24b", "optimized_ged_normalized": "#e45756",
}
OUTCOMES = [
    "abs_delta_beta0", "abs_delta_beta1", "topology_l1", "branchpoint_error",
    "endpoint_error", "degree_histogram_error", "connectivity_disagreement",
    "disconnected_gt_branches", "missing_bridge_edges", "false_cycle_closing_edges",
    "largest_component_error", "fragmentation", "combined_structural_damage",
]
TOLERANCE = 1e-7


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, **kwargs)
    temporary.replace(path)


def json_ready(value):
    if isinstance(value, dict): return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_ready(item) for item in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def load_task_results(run: Path) -> pd.DataFrame:
    expected = len(pd.read_csv(run / "task_manifest.csv"))
    paths = sorted((run / "results").glob("task_*.json"))
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} task JSON files, found {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    if any(row.get("record_status") != "complete" for row in rows):
        raise RuntimeError("not every GED task record is complete")
    frame = pd.DataFrame(rows)
    if frame.task_id.duplicated().any():
        raise RuntimeError("task IDs are duplicated")
    if not (frame.status == "optimal").all():
        counts = frame.status.value_counts().to_dict()
        raise RuntimeError(f"primary decomposition requires solver-optimal rows; found {counts}")
    if frame.upper_bound.isna().any() or frame.saved_matching_edit_cost.isna().any():
        raise RuntimeError("optimal or saved-matching costs are missing")
    return frame


def load_selected_records(run: Path, config: dict, keys: set[tuple[str, int]]) -> dict[tuple[str, int], object]:
    index_path = Path(config["output_root"]) / "prediction_index.csv"
    index = pd.read_csv(index_path, dtype={"sample_id": str})
    selected = index[index.apply(lambda row: (str(row.sample_id), int(row.epoch)) in keys, axis=1)].copy()
    if len(selected) != len(keys):
        raise RuntimeError(f"prediction index resolved {len(selected)} of {len(keys)} requested records")
    pairs = load_edge_pairs(Path(config["data"]["prediction_root"]))
    output = {}
    for shard_name, group in selected.groupby("shard", sort=True):
        shard = Path(shard_name); stat = shard.stat()
        if "shard_size_bytes" in group and not (group.shard_size_bytes.astype(int) == stat.st_size).all():
            raise RuntimeError(f"indexed shard size changed: {shard}")
        records = torch_load(shard)
        for row in group.itertuples():
            record = parse_record(records[int(row.offset)], pairs)
            output[(str(record.sample_id), int(record.epoch))] = record
    return output


def ambiguity_diagnostics(run: Path, row: pd.Series, record) -> dict[str, int | float]:
    artifact = np.load(run / str(row.hard_graph_path))
    coordinates = artifact["predicted_coordinates"]
    edges = artifact["predicted_edges"].reshape(-1, 2)
    slots = artifact["predicted_source_slots"]
    if len(record.gt_edges):
        lengths = np.linalg.norm(record.gt_nodes[record.gt_edges[:, 0]] - record.gt_nodes[record.gt_edges[:, 1]], axis=1)
        positive = lengths[lengths > 1e-8]
        scale = float(np.median(positive)) if len(positive) else np.sqrt(3.0)
    else:
        scale = np.sqrt(3.0)
    duplicate_candidates = 0
    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            duplicate_candidates += int(np.linalg.norm(coordinates[left] - coordinates[right]) <= .25 * scale)
    degree = np.zeros(len(coordinates), dtype=int)
    for left, right in edges:
        degree[int(left)] += 1; degree[int(right)] += 1
    matched_queries = set(map(int, record.matched_queries))
    subdivision_candidates = int(sum(int(degree[index] == 2 and int(slot) not in matched_queries) for index, slot in enumerate(slots)))
    gt_summary = graph_summary(HardGraph(record.gt_nodes, record.gt_edges))
    return {
        "duplicate_candidate_count": duplicate_candidates,
        "subdivision_candidate_count": subdivision_candidates,
        "largest_component_error": abs(float(row.largest_component_fraction) - float(gt_summary["largest_component_fraction"])),
    }


def build_score_table(run: Path, results: pd.DataFrame, records: dict, edge_weight: float) -> pd.DataFrame:
    structural = results[results.ged_variant == "structural"].copy()
    rows = []
    for row in structural.itertuples(index=False):
        values = row._asdict(); key = (str(row.sample_id), int(row.epoch)); record = records[key]
        binary = fixed_slot_disagreement(record, row.node_threshold, row.edge_threshold, edge_weight=edge_weight)
        normalizer = graph_size_normalizer(row.gt_node_count, row.gt_edge_count, 1.0, edge_weight)
        improvement = float(row.saved_matching_edit_cost) - float(row.upper_bound)
        if improvement < -TOLERANCE:
            raise RuntimeError(f"negative matching improvement {improvement} for task {row.task_id}")
        improvement = max(0.0, improvement)
        exact = row.relationformer_total_loss if pd.notna(row.relationformer_total_loss) else None
        total = float(exact) if exact is not None else float(row.audit_composite_proxy)
        saved_mapping = row.saved_matching_node_mapping
        optimized_mapping = row.node_mapping
        diagnostics = ambiguity_diagnostics(run, pd.Series(values), record)
        output = {
            "task_id": int(row.task_id), "sample_id": str(row.sample_id), "epoch": int(row.epoch),
            "checkpoint": row.checkpoint, "node_threshold": float(row.node_threshold),
            "edge_threshold": float(row.edge_threshold), "solver_status": row.status,
            "continuous_node_score": float(row.node_loss), "continuous_edge_score": float(row.edge_loss_proxy),
            "continuous_coordinate_score": float(row.coordinate_loss), "continuous_total_score": total,
            "relationformer_loss_proxy": float(row.audit_composite_proxy),
            "continuous_score_kind": "exact_training_loss" if exact is not None else "audit_proxy",
            **binary,
            "saved_matching_cost_raw": float(row.saved_matching_edit_cost),
            "saved_matching_cost_normalized": float(row.saved_matching_edit_cost) / normalizer,
            "saved_node_insertions": int(row.saved_matching_node_insertions),
            "saved_node_deletions": int(row.saved_matching_node_deletions),
            "saved_node_substitutions": int(row.saved_matching_node_substitutions),
            "saved_edge_insertions": int(row.saved_matching_edge_insertions),
            "saved_edge_deletions": int(row.saved_matching_edge_deletions),
            "saved_mapping": json.dumps(saved_mapping),
            "optimized_ged_raw": float(row.upper_bound),
            "optimized_ged_normalized": float(row.upper_bound) / normalizer,
            "optimized_node_insertions": int(row.node_insertions),
            "optimized_node_deletions": int(row.node_deletions),
            "optimized_node_substitutions": int(row.node_substitutions),
            "optimized_edge_insertions": int(row.edge_insertions),
            "optimized_edge_deletions": int(row.edge_deletions),
            "optimized_mapping": json.dumps(optimized_mapping),
            "matching_improvement_absolute": improvement,
            "matching_improvement_relative": improvement / max(float(row.saved_matching_edit_cost), 1e-12),
            "mapping_change_count": mapping_change_count(saved_mapping, optimized_mapping),
            "gt_node_count": int(row.gt_node_count), "gt_edge_count": int(row.gt_edge_count),
            "predicted_node_count": int(row.predicted_node_count), "predicted_edge_count": int(row.predicted_edge_count),
            "hard_graph_path": str(row.hard_graph_path),
            "abs_delta_beta0": row.abs_delta_beta0, "abs_delta_beta1": row.abs_delta_beta1,
            "topology_l1": row.topology_l1, "branchpoint_error": row.branchpoint_count_error,
            "endpoint_error": row.endpoint_count_error, "degree_histogram_error": row.degree_histogram_l1_error,
            "connectivity_disagreement": row.matched_gt_connectivity_disagreement,
            "disconnected_gt_branches": row.matched_gt_branches_disconnected,
            "missing_bridge_edges": row.missing_bridge_edges,
            "false_cycle_closing_edges": row.false_cycle_closing_edges,
            "fragmentation": row.fragmentation,
            **diagnostics,
        }
        rows.append(output)
    frame = pd.DataFrame(rows)
    components = ["topology_l1", "branchpoint_error", "connectivity_disagreement", "false_cycle_closing_edges", "fragmentation"]
    frame["combined_structural_damage"] = frame.groupby("node_threshold", group_keys=False)[components].apply(
        lambda group: rank_normalized_mean(group, components)
    )
    return frame


def correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold, group in frame.groupby("node_threshold"):
        for score in SCORES:
            for outcome in OUTCOMES:
                rows.append({"scope": "structural", "threshold": threshold, "left": score, "right": outcome, **correlation(group, score, outcome)})
        for left, right in (
            ("optimized_ged_normalized", "saved_matching_cost_normalized"),
            ("optimized_ged_normalized", "binary_local_total_normalized"),
            ("saved_matching_cost_normalized", "binary_local_total_normalized"),
        ):
            rows.append({"scope": "structural_redundancy", "threshold": threshold, "left": left, "right": right, **correlation(group, left, right)})
    return pd.DataFrame(rows)


def cv_table(frame: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    models = {
        "A_continuous": ["continuous_node_score", "continuous_edge_score", "continuous_coordinate_score"],
        "B_binary": ["binary_node_disagreement", "binary_edge_disagreement"],
        "C_saved_edit": ["saved_node_insertions", "saved_node_deletions", "saved_edge_insertions", "saved_edge_deletions"],
    }
    rows = []
    for threshold, group in frame.groupby("node_threshold"):
        for name, predictors in models.items():
            metrics, residuals = grouped_cv_extended(group, "optimized_ged_raw", predictors, "sample_id", folds=folds, seed=seed)
            row = {"threshold": threshold, "model": name, "predictors": "+".join(predictors), **metrics}
            if len(residuals):
                joined = residuals[["residual"]].join(group[OUTCOMES])
                for outcome in OUTCOMES:
                    row[f"residual_spearman_{outcome}"] = correlation(joined, "residual", outcome)["spearman"]
            rows.append(row)
    return pd.DataFrame(rows)


def improvement_summary(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    positive = frame[frame.matching_improvement_absolute > TOLERANCE]
    values = positive.matching_improvement_absolute
    relative = positive.matching_improvement_relative
    summary = {
        "rows": len(frame), "exact_zero_fraction": float((frame.matching_improvement_absolute <= TOLERANCE).mean()),
        "positive_fraction": float((frame.matching_improvement_absolute > TOLERANCE).mean()),
        "mean_positive": float(values.mean()) if len(values) else None,
        "median_positive": float(values.median()) if len(values) else None,
        "positive_q90": float(values.quantile(.9)) if len(values) else None,
        "positive_q95": float(values.quantile(.95)) if len(values) else None,
        "maximum": float(values.max()) if len(values) else None,
        "median_relative_positive": float(relative.median()) if len(relative) else None,
        "relative_q90_positive": float(relative.quantile(.9)) if len(relative) else None,
    }
    columns = list(dict.fromkeys([
        "sample_id", "checkpoint", "epoch", "node_threshold", "edge_threshold", "saved_mapping", "optimized_mapping",
        "saved_node_insertions", "saved_node_deletions", "saved_edge_insertions", "saved_edge_deletions",
        "optimized_node_insertions", "optimized_node_deletions", "optimized_edge_insertions", "optimized_edge_deletions",
        "matching_improvement_absolute", "matching_improvement_relative", "mapping_change_count",
        "gt_node_count", "gt_edge_count", "predicted_node_count", "predicted_edge_count",
        "duplicate_candidate_count", "subdivision_candidate_count", "branchpoint_error",
        "connectivity_disagreement", "false_cycle_closing_edges", "continuous_total_score", *OUTCOMES,
    ]))
    cases = positive.sort_values(["matching_improvement_absolute", "sample_id", "epoch"], ascending=[False, True, True])[columns]
    return summary, cases


def improvement_bootstrap(frame: pd.DataFrame, draws: int, seed: int) -> dict:
    patches = sorted(frame.sample_id.unique()); rng = np.random.default_rng(seed)
    positives = np.asarray([int((frame.loc[frame.sample_id == patch, "matching_improvement_absolute"] > TOLERANCE).sum()) for patch in patches])
    totals = np.asarray([int((frame.sample_id == patch).sum()) for patch in patches])
    chosen = rng.integers(len(patches), size=(draws, len(patches)))
    values = positives[chosen].sum(axis=1) / totals[chosen].sum(axis=1)
    return {"estimand": "positive_matching_improvement_frequency", "estimate": float((frame.matching_improvement_absolute > TOLERANCE).mean()), "low": float(np.quantile(values, .025)), "high": float(np.quantile(values, .975)), "draws": draws}


def correlation_bootstrap(frame: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    comparisons = [
        *( (score, "combined_structural_damage") for score in SCORES ),
        ("optimized_ged_normalized", "saved_matching_cost_normalized"),
        ("optimized_ged_normalized", "binary_local_total_normalized"),
    ]
    rows = []
    for threshold, group in frame.groupby("node_threshold"):
        patches = sorted(group.sample_id.unique())
        for left, right in comparisons:
            observed = correlation(group, left, right)["spearman"]
            rng = np.random.default_rng(seed + int(1000 * threshold) + sum(map(ord, left + right)))
            values = []
            for _ in range(draws):
                chosen = rng.choice(patches, len(patches), replace=True)
                sampled = pd.concat([group[group.sample_id == patch] for patch in chosen], ignore_index=True)
                value = correlation(sampled, left, right)["spearman"]
                if np.isfinite(value): values.append(value)
            rows.append({
                "threshold": threshold, "outcome": right,
                "estimand": f"spearman:{left}:{right}", "estimate": observed, "draws": len(values),
                "low": float(np.quantile(values, .025)) if values else np.nan,
                "high": float(np.quantile(values, .975)) if values else np.nan,
            })
    return pd.DataFrame(rows)


def improvement_group_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy(); working["improved"] = working.matching_improvement_absolute > TOLERANCE
    measures = [
        "gt_node_count", "gt_edge_count", "predicted_node_count", "predicted_edge_count",
        "duplicate_candidate_count", "subdivision_candidate_count", "branchpoint_error",
        "connectivity_disagreement", "false_cycle_closing_edges", "continuous_total_score",
    ]
    rows = []
    for group_name, group in working.groupby("improved"):
        for measure in measures:
            rows.append({"comparison": "improved_status", "group": bool(group_name), "measure": measure, "n": len(group), "mean": float(group[measure].mean()), "median": float(group[measure].median())})
    for threshold, group in working.groupby("node_threshold"):
        rate = float(group.improved.mean())
        rows.append({"comparison": "threshold", "group": str(threshold), "measure": "positive_improvement_frequency", "n": len(group), "mean": rate, "median": rate})
    for epoch, group in working.groupby("epoch"):
        rate = float(group.improved.mean())
        rows.append({"comparison": "checkpoint", "group": str(epoch), "measure": "positive_improvement_frequency", "n": len(group), "mean": rate, "median": rate})
    return pd.DataFrame(rows)


def disagreement_table(rankings: pd.DataFrame, baseline: str) -> pd.DataFrame:
    primary = rankings[rankings.outcome == "combined_structural_damage"]
    keys = ["sample_id", "threshold", "epoch_a", "epoch_b", "outcome"]
    left = primary[primary.score == baseline].set_index(keys)
    right = primary[primary.score == "optimized_ged_normalized"].set_index(keys)
    joined = left.add_prefix("baseline_").join(right.add_prefix("ged_"), how="inner").reset_index()
    joined = joined[(joined.baseline_score_direction != joined.ged_score_direction) & ~joined.baseline_outcome_tie]
    joined["optimized_ged_correct"] = joined.ged_correct
    joined["simpler_baseline_correct"] = joined.baseline_correct
    joined["absolute_score_disagreement"] = abs(joined.ged_score_direction - joined.baseline_score_direction)
    joined["structural_damage_difference"] = abs(joined.ged_outcome_b - joined.ged_outcome_a)
    return joined.sort_values(
        ["absolute_score_disagreement", "structural_damage_difference", "sample_id", "epoch_a", "epoch_b"],
        ascending=[False, False, True, True, True],
    )


def geometric_decomposition(results: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["sample_id", "epoch", "node_threshold", "edge_threshold"]
    geometric = results[results.ged_variant == "geometric"].copy()
    keep = keys + ["upper_bound", "saved_matching_edit_cost", "coordinate_loss", "coordinate_cost"]
    geometric = geometric[keep].rename(columns={
        "upper_bound": "geometric_optimized_raw", "saved_matching_edit_cost": "geometric_saved_raw",
        "coordinate_loss": "continuous_coordinate_loss", "coordinate_cost": "direct_coordinate_error",
    })
    joined = scores.merge(geometric, on=keys, how="inner", validate="one_to_one")
    joined["geometric_matching_improvement"] = joined.geometric_saved_raw - joined.geometric_optimized_raw
    joined["structural_plus_direct_coordinate"] = joined.optimized_ged_raw + joined.direct_coordinate_error
    return joined[keys + [
        "optimized_ged_raw", "geometric_saved_raw", "geometric_optimized_raw", "geometric_matching_improvement",
        "continuous_coordinate_loss", "direct_coordinate_error", "structural_plus_direct_coordinate", *OUTCOMES,
    ]]


def save_figure(figure, path: Path) -> None:
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(path.with_suffix(f".{suffix}"), dpi=180)
    plt.close(figure)


def create_plots(scores: pd.DataFrame, accuracy: pd.DataFrame, gains: pd.DataFrame, bootstrap: pd.DataFrame, stability: pd.DataFrame, multi_accuracy: pd.DataFrame, geometric: pd.DataFrame, plots: Path) -> None:
    plots.mkdir(parents=True, exist_ok=True)
    subset = accuracy[(accuracy.outcome == "combined_structural_damage") & (accuracy.threshold.astype(str) != "multithreshold")].copy()
    thresholds = sorted(map(float, subset.threshold.unique())); x = np.arange(len(thresholds)); width = .19
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for index, score in enumerate(SCORES):
        values = subset[subset.score == score].set_index("threshold").reindex(thresholds).accuracy.to_numpy(float)
        position = x + (index - 1.5) * width
        axis.bar(position, values, width, label=SCORE_LABELS[score], color=COLORS[score])
        intervals = bootstrap[(bootstrap.outcome == "combined_structural_damage") & (bootstrap.estimand == score)]
        lows=[]; highs=[]
        for threshold, value in zip(thresholds, values):
            row = intervals[np.isclose(pd.to_numeric(intervals.threshold, errors="coerce"), threshold)]
            low = float(row.iloc[0].low) if len(row) else value; high = float(row.iloc[0].high) if len(row) else value
            lows.append(max(0, value-low)); highs.append(max(0, high-value))
        axis.errorbar(position, values, yerr=np.asarray([lows, highs]), fmt="none", ecolor="black", capsize=2, linewidth=.8)
    axis.set_xticks(x, thresholds); axis.set(xlabel="threshold", ylabel="ranking accuracy"); axis.legend(fontsize=8)
    save_figure(figure, plots / "ranking_accuracy_by_score_threshold")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    gain_names = ["thresholding_gain", "edit_accounting_gain", "correspondence_optimization_gain"]
    gain_subset = gains[(gains.outcome == "combined_structural_damage") & (gains.threshold.astype(str) != "multithreshold")].set_index("threshold")
    for index, gain in enumerate(gain_names):
        values = gain_subset.reindex(thresholds)[gain].to_numpy(float); position = x + (index - 1) * .25
        axis.bar(position, values, .25, label=gain.replace("_", " "))
        intervals = bootstrap[(bootstrap.outcome == "combined_structural_damage") & (bootstrap.estimand == gain)]
        lows=[]; highs=[]
        for threshold, value in zip(thresholds, values):
            row = intervals[np.isclose(pd.to_numeric(intervals.threshold, errors="coerce"), threshold)]
            low = float(row.iloc[0].low) if len(row) else value; high = float(row.iloc[0].high) if len(row) else value
            lows.append(max(0, value-low)); highs.append(max(0, high-value))
        axis.errorbar(position, values, yerr=np.asarray([lows, highs]), fmt="none", ecolor="black", capsize=2, linewidth=.8)
    axis.set_xticks(x, thresholds); axis.legend(fontsize=8)
    axis.axhline(0, color="black", linewidth=.8); axis.set_ylabel("sequential ranking gain")
    save_figure(figure, plots / "sequential_ranking_gains")

    for x, name in (("binary_local_total_normalized", "optimized_ged_vs_binary_local"), ("saved_matching_cost_normalized", "optimized_ged_vs_saved_matching")):
        figure, axis = plt.subplots(figsize=(6, 5))
        for threshold, group in scores.groupby("node_threshold"):
            axis.scatter(group[x], group.optimized_ged_normalized, s=14, alpha=.5, label=f"threshold {threshold}")
        axis.set(xlabel=SCORE_LABELS[x], ylabel="optimized structural GED"); axis.legend()
        save_figure(figure, plots / name)

    positive = scores.matching_improvement_absolute[scores.matching_improvement_absolute > TOLERANCE]
    figure, axis = plt.subplots(figsize=(6, 4)); axis.hist(positive, bins=30); axis.set(xlabel="positive matching improvement", ylabel="count")
    save_figure(figure, plots / "matching_improvement_histogram")

    figure, axis = plt.subplots(figsize=(6, 4)); scores.boxplot(column="matching_improvement_relative", by="node_threshold", ax=axis); figure.suptitle(""); axis.set_ylabel("relative matching improvement")
    save_figure(figure, plots / "relative_matching_improvement_by_threshold")

    for x, name in (("predicted_node_count", "matching_improvement_vs_graph_size"), ("combined_structural_damage", "matching_improvement_vs_structural_damage")):
        figure, axis = plt.subplots(figsize=(6, 4)); axis.scatter(scores[x], scores.matching_improvement_absolute, s=14, alpha=.45)
        axis.set(xlabel=x.replace("_", " "), ylabel="matching improvement")
        save_figure(figure, plots / name)

    correlations = stability[stability.record_type == "rank_correlation"]
    labels = SCORES; matrix = np.full((len(labels), 3), np.nan); pairs = sorted({(row.threshold_a, row.threshold_b) for row in correlations.itertuples()})
    for i, score in enumerate(labels):
        for j, pair in enumerate(pairs):
            row = correlations[(correlations.score == score) & (correlations.threshold_a == pair[0]) & (correlations.threshold_b == pair[1])]
            if len(row): matrix[i, j] = row.iloc[0].spearman
    figure, axis = plt.subplots(figsize=(7, 4)); image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_yticks(range(len(labels)), [SCORE_LABELS[x] for x in labels]); axis.set_xticks(range(len(pairs)), [f"{a}–{b}" for a, b in pairs]); figure.colorbar(image, ax=axis, label="Spearman")
    save_figure(figure, plots / "cross_threshold_rank_correlation_heatmap")

    flips = stability[stability.record_type == "checkpoint_flip"].set_index("score").reindex(SCORES)
    figure, axis = plt.subplots(figsize=(7, 4)); axis.bar([SCORE_LABELS[x] for x in SCORES], flips.flip_rate, color=[COLORS[x] for x in SCORES]); axis.tick_params(axis="x", rotation=20); axis.set_ylabel("checkpoint-ranking flip rate")
    save_figure(figure, plots / "checkpoint_ranking_flip_rate")

    multi = multi_accuracy[multi_accuracy.outcome == "combined_structural_damage"].set_index("score").reindex(SCORES)
    figure, axis = plt.subplots(figsize=(7, 4)); axis.bar([SCORE_LABELS[x] for x in SCORES], multi.accuracy, color=[COLORS[x] for x in SCORES]); axis.tick_params(axis="x", rotation=20); axis.set_ylabel("multithreshold ranking accuracy")
    save_figure(figure, plots / "multithreshold_ranking_comparison")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(geometric.optimized_ged_raw, geometric.structural_plus_direct_coordinate, s=12, alpha=.4)
    axes[0].set(xlabel="geometric GED", ylabel="structural GED + direct coordinate error")
    axes[1].scatter(geometric.continuous_coordinate_loss, geometric.geometric_optimized_raw, s=12, alpha=.4)
    axes[1].set(xlabel="continuous coordinate loss", ylabel="geometric GED")
    save_figure(figure, plots / "structural_vs_geometric_decomposition")


def plot_graph(axis, coordinates: np.ndarray, edges: np.ndarray, title: str) -> None:
    axis.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], s=16)
    for left, right in np.asarray(edges, dtype=int).reshape(-1, 2):
        points = coordinates[[left, right]]; axis.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=1)
    axis.set_title(title, fontsize=8); axis.set_axis_off()


def create_disagreement_graphs(run: Path, scores: pd.DataFrame, disagreements: dict[str, pd.DataFrame], plots: Path) -> None:
    lookup = scores.set_index(["sample_id", "epoch", "node_threshold"])
    destination = plots / "disagreement_graphs"; destination.mkdir(exist_ok=True)
    for category, cases in disagreements.items():
        threshold_cases = cases[cases.threshold.astype(str) != "multithreshold"].head(10)
        for rank, case in enumerate(threshold_cases.itertuples()):
            key_a = (str(case.sample_id), int(case.epoch_a), float(case.threshold))
            key_b = (str(case.sample_id), int(case.epoch_b), float(case.threshold))
            if key_a not in lookup.index or key_b not in lookup.index: continue
            row_a, row_b = lookup.loc[key_a], lookup.loc[key_b]
            artifact_a = np.load(run / str(row_a.hard_graph_path)); artifact_b = np.load(run / str(row_b.hard_graph_path))
            figure = plt.figure(figsize=(12, 4))
            plot_graph(figure.add_subplot(131, projection="3d"), artifact_a["gt_coordinates"], artifact_a["gt_edges"], "ground truth")
            plot_graph(figure.add_subplot(132, projection="3d"), artifact_a["predicted_coordinates"], artifact_a["predicted_edges"], f"epoch {case.epoch_a}")
            plot_graph(figure.add_subplot(133, projection="3d"), artifact_b["predicted_coordinates"], artifact_b["predicted_edges"], f"epoch {case.epoch_b}")
            safe_sample = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(case.sample_id))
            save_figure(figure, destination / f"{category}_{rank:02d}_{safe_sample}_{case.threshold}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-run", required=True)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args(); run = Path(args.main_run).resolve(); output = Path(args.output).resolve() if args.output else run / "baseline_decomposition"
    required = [run / "COMPLETED.json", run / "resolved_config.yaml", run / "task_manifest.csv", run / "results"]
    missing = [str(path) for path in required if not path.exists()]
    if missing: raise FileNotFoundError(f"main run is incomplete; missing {missing}")
    if (output / "COMPLETED.json").is_file() and not args.overwrite:
        raise FileExistsError(f"completed decomposition exists: {output}; pass --overwrite to replace it")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"nonempty decomposition directory exists: {output}; pass --overwrite")
    if args.overwrite and output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True); config = yaml.safe_load((run / "resolved_config.yaml").read_text())
    results = load_task_results(run); keys = set(zip(results.sample_id.astype(str), results.epoch.astype(int)))
    records = load_selected_records(run, config, keys); edge_weight = float(config["costs"]["edge_weight"])
    scores = build_score_table(run, results, records, edge_weight)
    atomic_csv(scores, output / "score_table.csv.gz", compression="gzip")

    rankings = pd.concat([
        paired_checkpoint_rows(group, SCORES, OUTCOMES, threshold_label=float(threshold))
        for threshold, group in scores.groupby("node_threshold")
    ], ignore_index=True)
    multi = multithreshold_aggregate(scores, SCORES, OUTCOMES)
    multi_rankings = paired_checkpoint_rows(multi, SCORES, OUTCOMES, threshold_label="multithreshold")
    all_rankings = pd.concat([rankings, multi_rankings], ignore_index=True)
    accuracy = ranking_accuracy(all_rankings); gains = sequential_gains(accuracy, SCORES)
    bootstrap = patch_bootstrap(all_rankings, SCORES, draws=args.bootstrap_draws, seed=int(config["analysis"]["seed"]))
    improvement_ci = improvement_bootstrap(scores, args.bootstrap_draws, int(config["analysis"]["seed"]))
    correlation_ci = correlation_bootstrap(scores, args.bootstrap_draws, int(config["analysis"]["seed"]))
    bootstrap = pd.concat([bootstrap, pd.DataFrame([improvement_ci]), correlation_ci], ignore_index=True)
    atomic_csv(all_rankings, output / "paired_rankings.csv.gz", compression="gzip")
    atomic_csv(accuracy, output / "ranking_accuracy.csv"); atomic_csv(gains, output / "ranking_gains.csv")
    atomic_csv(bootstrap, output / "bootstrap_intervals.csv")

    correlations = correlation_table(scores); cv = cv_table(scores, int(config["analysis"]["cv_folds"]), int(config["analysis"]["seed"]))
    stability = threshold_stability(scores, SCORES)
    atomic_csv(correlations, output / "correlations.csv"); atomic_csv(cv, output / "grouped_cv_results.csv")
    atomic_csv(stability, output / "threshold_stability.csv"); atomic_csv(multi, output / "multithreshold_results.csv")

    improvement, cases = improvement_summary(scores); atomic_csv(cases, output / "matching_improvement_cases.csv.gz", compression="gzip")
    atomic_csv(improvement_group_comparison(scores), output / "matching_improvement_group_comparison.csv")
    mapping_cases = scores[(scores.mapping_change_count > 0) & (scores.matching_improvement_absolute <= TOLERANCE)].sort_values(["mapping_change_count", "sample_id", "epoch"], ascending=[False, True, True])
    atomic_csv(mapping_cases, output / "mapping_change_cases.csv.gz", compression="gzip")
    disagreements = {}
    improvement_lookup = scores.set_index(["sample_id", "node_threshold", "epoch"])["matching_improvement_absolute"]
    range_lookup = scores.groupby(["sample_id", "epoch"]).optimized_ged_normalized.agg(lambda value: float(value.max() - value.min()))
    for name, baseline in (("continuous_vs_ged", SCORES[0]), ("binary_vs_ged", SCORES[1]), ("saved_matching_vs_ged", SCORES[2])):
        disagreements[name] = disagreement_table(all_rankings, baseline)
        if len(disagreements[name]):
            disagreements[name]["matching_improvement"] = disagreements[name].apply(
                lambda row: max(
                    float(improvement_lookup.get((row.sample_id, float(row.threshold), int(row.epoch_a)), 0)),
                    float(improvement_lookup.get((row.sample_id, float(row.threshold), int(row.epoch_b)), 0)),
                ) if str(row.threshold) != "multithreshold" else np.nan,
                axis=1,
            )
            disagreements[name]["cross_threshold_score_range"] = disagreements[name].apply(
                lambda row: max(float(range_lookup.get((row.sample_id, int(row.epoch_a)), 0)), float(range_lookup.get((row.sample_id, int(row.epoch_b)), 0))), axis=1
            )
            disagreements[name] = disagreements[name].sort_values(
                ["absolute_score_disagreement", "structural_damage_difference", "matching_improvement", "cross_threshold_score_range", "sample_id"],
                ascending=[False, False, False, True, True],
            )
        atomic_csv(disagreements[name], output / f"{name}_disagreements.csv")

    geometric = geometric_decomposition(results, scores); atomic_csv(geometric, output / "geometric_decomposition.csv.gz", compression="gzip")
    geometric_rows = []
    for threshold, group in geometric.groupby("node_threshold"):
        for left in ("geometric_saved_raw", "optimized_ged_raw", "continuous_coordinate_loss", "direct_coordinate_error", "structural_plus_direct_coordinate"):
            geometric_rows.append({"scope": "geometric", "threshold": threshold, "left": left, "right": "geometric_optimized_raw", **correlation(group, left, "geometric_optimized_raw")})
        for outcome in OUTCOMES:
            geometric_rows.append({"scope": "geometric_outcome", "threshold": threshold, "left": "geometric_optimized_raw", "right": outcome, **correlation(group, "geometric_optimized_raw", outcome)})
    correlations = pd.concat([correlations, pd.DataFrame(geometric_rows)], ignore_index=True)
    atomic_csv(correlations, output / "correlations.csv")
    create_plots(scores, accuracy, gains, bootstrap, stability, ranking_accuracy(multi_rankings), geometric, output / "plots")
    create_disagreement_graphs(run, scores, disagreements, output / "plots")

    primary = accuracy[accuracy.outcome == "combined_structural_damage"].pivot(index="threshold", columns="score", values="accuracy")
    primary_gains = gains[gains.outcome == "combined_structural_damage"].set_index("threshold")
    multi_gain = float(primary_gains.loc["multithreshold", "correspondence_optimization_gain"])
    ci_row = bootstrap[(bootstrap.threshold.astype(str) == "multithreshold") & (bootstrap.outcome == "combined_structural_damage") & (bootstrap.estimand == "correspondence_optimization_gain")]
    gain_low = float(ci_row.iloc[0].low) if len(ci_row) else np.nan; gain_high = float(ci_row.iloc[0].high) if len(ci_row) else np.nan
    model_b = float(cv[cv.model == "B_binary"].r2.median()); model_c = float(cv[cv.model == "C_saved_edit"].r2.median())
    multi_row = primary.loc["multithreshold"]
    multi_binary_advantage = float(multi_row["optimized_ged_normalized"] - multi_row["binary_local_total_normalized"])
    multi_saved_advantage = float(multi_row["optimized_ged_normalized"] - multi_row["saved_matching_cost_normalized"])
    no_go = (model_b >= .90 or model_c >= .95) and (multi_gain < .02 or (np.isfinite(gain_low) and gain_low <= 0)) and multi_saved_advantage < .02
    decision = "no_go_for_greed" if no_go else ("evidence_supporting_further_encoder_work" if multi_gain >= .05 and gain_low > 0 else "inconclusive_do_not_train_yet")
    summary = {
        "schema": "ged-baseline-decomposition/v1", "source_run": str(run), "rows": len(scores),
        "source_patches": int(scores.sample_id.nunique()), "solver_optimal_only": True,
        "score_normalization": "raw score divided by max(GT nodes + edge_weight * GT edges, 1)",
        "combined_structural_damage": "mean of within-threshold percentile ranks (average ties; missing components omitted) for topology_l1, branchpoint_error, connectivity_disagreement, false_cycle_closing_edges, fragmentation",
        "binary_vs_saved_max_absolute_difference": float((scores.binary_local_total_raw - scores.saved_matching_cost_raw).abs().max()),
        "matching_improvement": improvement,
        "binary_model_median_cv_r2": model_b, "saved_edit_model_median_cv_r2": model_c,
        "multithreshold_accuracies": {SCORE_LABELS[key]: float(multi_row[key]) for key in SCORES},
        "multithreshold_binary_advantage": multi_binary_advantage,
        "multithreshold_saved_matching_advantage": multi_saved_advantage,
        "multithreshold_correspondence_gain": multi_gain,
        "multithreshold_correspondence_gain_ci95": [gain_low, gain_high],
        "positive_improvement_frequency_ci95": [improvement_ci["low"], improvement_ci["high"]],
        "decision": decision, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "RelationFormer total training loss is unavailable; the existing audit proxy is labelled relationformer_loss_proxy.",
            "Duplicate and subdivision candidate counts are prespecified graph-only heuristics, not ground-truth ambiguity labels.",
            "No image volumes are required or loaded; disagreement inspection uses exported graph artifacts.",
        ],
    }
    summary = json_ready(summary)
    atomic_json(output / "summary.json", summary)
    report = f"""# GED baseline decomposition

This analysis reuses the 1,200 certified-optimal labels from `{run.name}`; it launches no GED solver and trains no encoder.

## Central decomposition

The four successive scores are the existing RelationFormer audit proxy, thresholded fixed-slot disagreement, the valid edit cost under the saved Hungarian matching, and optimized structural GED. All thresholds are analysed separately; structural and geometric GED are never pooled.

1. **How much comes from thresholding?** The grouped-CV binary baseline predicts optimized GED with median R² `{model_b:.3f}`. Threshold-specific ranking gains are in `ranking_gains.csv`.
2. **Does edit accounting help?** The maximum raw difference between binary local disagreement and saved-matching structural cost is `{summary['binary_vs_saved_max_absolute_difference']:.6g}` under this implementation's convention, which charges incident edge operations separately.
3. **Does correspondence optimization help?** Positive cost improvement occurs in `{improvement['positive_fraction']:.1%}` of rows. The multithreshold correspondence-ranking gain is `{multi_gain:.3f}` with patch-bootstrap 95% CI `[{gain_low:.3f}, {gain_high:.3f}]`.
4. **Patch-stable?** Every confidence interval resamples the {summary['source_patches']} source patches, keeping checkpoints and thresholds clustered.
5. **Meaningful ambiguities?** `matching_improvement_cases.csv.gz` reports duplicate/subdivision heuristics, mapping changes, graph sizes, losses, and structural outcomes; these heuristics are diagnostic rather than ground truth.
6. **Does multithreshold GED beat simple baselines?** Its ranking advantage is `{multi_binary_advantage:.3f}` over binary local and `{multi_saved_advantage:.3f}` over saved-matching edits for the prespecified combined damage outcome.
7. **Is GREED justified?** **`{decision}`**. The decision follows the requested practical correspondence-gain thresholds and requires superiority over the simple multithreshold baselines.

## Accounting and normalization

Raw structural operations use node weight 1 and the configured edge weight `{edge_weight:g}`. Normalized scores divide by `max(GT nodes + edge_weight × GT edges, 1)`. The combined damage outcome is the mean of within-threshold percentile ranks with average ranks for ties; missing components are omitted row-wise. Outcome and score ties are counted explicitly; primary accuracy excludes either tie, and half-credit results are reported separately.

## Geometric secondary analysis

`geometric_decomposition.csv.gz` compares geometric optimized and saved-matching GED against continuous coordinate loss, hard structural GED, and structural GED plus direct coordinate error. It is secondary and is not pooled with structural GED.
"""
    (output / "report.md").write_text(report)
    atomic_json(output / "COMPLETED.json", {"status": "complete", "summary": "summary.json", "report": "report.md"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
