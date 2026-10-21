#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from vascular_graph_embedding.data.storage import atomic_json
from vascular_graph_embedding.ged.statistics import correlation, grouped_bootstrap_correlation, grouped_linear_cv


ASSOCIATION_TARGETS = (
    "relationformer_total_loss", "audit_composite_proxy", "node_loss", "edge_loss_proxy", "coordinate_loss",
    "saved_matching_edit_cost", "abs_delta_beta0", "abs_delta_beta1", "topology_l1",
    "branchpoint_count_error", "matched_gt_connectivity_disagreement",
    "false_cycle_closing_edges", "fragmentation",
)


def json_default(value):
    """Convert NumPy values produced by pandas rows to JSON-native values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    temporary = path.with_name(path.name + ".tmp"); frame.to_csv(temporary, index=False, **kwargs); temporary.replace(path)


def load_results(run: Path, expected: int) -> pd.DataFrame:
    rows = []
    for path in sorted((run / "results").glob("task_*.json")):
        row = json.loads(path.read_text())
        if row.get("record_status") == "complete": rows.append(row)
    if len(rows) != expected: raise RuntimeError(f"expected {expected} completed task records, found {len(rows)}")
    frame = pd.json_normalize(rows, sep="__")
    if frame.task_id.duplicated().any(): raise RuntimeError("duplicate task result IDs")
    return frame


def associations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    filters = {"optimal_only": frame[frame.status == "optimal"], "all_incumbents": frame[frame.upper_bound.notna()]}
    for filter_name, filtered in filters.items():
        for (variant, node_threshold, edge_threshold), group in filtered.groupby(["ged_variant", "node_threshold", "edge_threshold"]):
            for target in ASSOCIATION_TARGETS:
                if target in group:
                    rows.append({"status_filter": filter_name, "ged_variant": variant, "node_threshold": node_threshold, "edge_threshold": edge_threshold, "metric": target, **correlation(group, "upper_bound", target)})
    return pd.DataFrame(rows)


def ranking_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []; metrics = ("topology_l1", "matched_gt_connectivity_disagreement", "branchpoint_count_error", "false_cycle_closing_edges", "combined_structural_score")
    optimal = frame[frame.status == "optimal"]
    for keys, group in optimal.groupby(["sample_id", "ged_variant", "node_threshold", "edge_threshold"]):
        values = group.sort_values("epoch").drop_duplicates("epoch")
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                left, right = values.iloc[left_index], values.iloc[right_index]
                ged_direction = np.sign(float(right.upper_bound) - float(left.upper_bound))
                loss_direction = np.sign(float(right.audit_composite_proxy) - float(left.audit_composite_proxy))
                for metric in metrics:
                    structural_direction = np.sign(float(right[metric]) - float(left[metric]))
                    rows.append({
                        "sample_id": keys[0], "ged_variant": keys[1], "node_threshold": keys[2], "edge_threshold": keys[3],
                        "epoch_a": int(left.epoch), "epoch_b": int(right.epoch), "structural_metric": metric,
                        "ged_agrees": bool(ged_direction == structural_direction) if structural_direction else None,
                        "loss_agrees": bool(loss_direction == structural_direction) if structural_direction else None,
                        "ged_direction": ged_direction, "loss_direction": loss_direction, "structural_direction": structural_direction,
                    })
    return pd.DataFrame(rows)


def disagreement_cases(frame: pd.DataFrame, loss_tolerance: float, ged_difference: float) -> pd.DataFrame:
    rows = []; optimal = frame[frame.status == "optimal"]
    for keys, group in optimal.groupby(["sample_id", "ged_variant", "node_threshold", "edge_threshold"]):
        values = group.sort_values("epoch")
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                left, right = values.iloc[left_index], values.iloc[right_index]
                loss_delta = abs(float(left.audit_composite_proxy) - float(right.audit_composite_proxy))
                ged_delta = abs(float(left.upper_bound) - float(right.upper_bound))
                structural_delta = abs(float(left.topology_l1) - float(right.topology_l1)) + abs(float(left.matched_gt_connectivity_disagreement) - float(right.matched_gt_connectivity_disagreement))
                reason = None
                if loss_delta <= loss_tolerance and ged_delta >= ged_difference: reason = "similar_loss_different_ged"
                elif ged_delta <= ged_difference * .25 and structural_delta >= 2: reason = "similar_ged_different_structural_damage"
                if reason:
                    rows.append({
                        "sample_id": keys[0], "ged_variant": keys[1], "node_threshold": keys[2], "edge_threshold": keys[3],
                        "epoch_a": int(left.epoch), "epoch_b": int(right.epoch), "loss_a": left.audit_composite_proxy,
                        "loss_b": right.audit_composite_proxy, "ged_a": left.upper_bound, "ged_b": right.upper_bound,
                        "topology_l1_a": left.topology_l1, "topology_l1_b": right.topology_l1,
                        "connectivity_a": left.matched_gt_connectivity_disagreement,
                        "connectivity_b": right.matched_gt_connectivity_disagreement,
                        "operations_a": json.dumps({key: left[key] for key in ("node_insertions", "node_deletions", "edge_insertions", "edge_deletions")}, default=json_default),
                        "operations_b": json.dumps({key: right[key] for key in ("node_insertions", "node_deletions", "edge_insertions", "edge_deletions")}, default=json_default),
                        "node_mapping_a": json.dumps(left.node_mapping, default=json_default),
                        "node_mapping_b": json.dumps(right.node_mapping, default=json_default),
                        "reason_selected": reason, "rank_score": ged_delta + structural_delta,
                    })
    return pd.DataFrame(rows).sort_values("rank_score", ascending=False) if rows else pd.DataFrame()


def save_scatter(frame: pd.DataFrame, x: str, y: str, plots: Path, name: str) -> None:
    if x not in frame or y not in frame: return
    figure, axis = plt.subplots(figsize=(6.5, 5))
    for (variant, status), group in frame.groupby(["ged_variant", "status"]):
        marker = "o" if status == "optimal" else "x"
        axis.scatter(group[x], group[y], s=22, alpha=.65, marker=marker, label=f"{variant}, {status}")
    axis.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " ")); axis.legend(fontsize=7); figure.tight_layout()
    for suffix in ("png", "pdf"): figure.savefig(plots / f"{name}.{suffix}", dpi=180)
    plt.close(figure)


def plots(frame: pd.DataFrame, residuals: pd.DataFrame, ranking: pd.DataFrame, cases: pd.DataFrame, destination: Path) -> None:
    destination.mkdir(exist_ok=True)
    for metric in ("audit_composite_proxy", "node_loss", "edge_loss_proxy", "coordinate_loss"):
        save_scatter(frame, metric, "upper_bound", destination, f"ged_vs_{metric}")
    save_scatter(frame, "saved_matching_edit_cost", "upper_bound", destination, "optimized_vs_saved_matching")
    for metric in ("topology_l1", "abs_delta_beta0", "abs_delta_beta1"):
        save_scatter(frame, metric, "upper_bound", destination, f"ged_vs_{metric}")
    if len(residuals): save_scatter(residuals.assign(ged_variant="cv", status="held_out"), "topology_l1", "residual_ged", destination, "ged_residual_vs_topology_l1")
    save_scatter(frame, "predicted_node_count", "runtime_seconds", destination, "solver_runtime_vs_graph_size")
    save_scatter(frame, "predicted_node_count", "relative_gap", destination, "solver_gap_vs_graph_size")
    for column, name in (("epoch", "ged_by_checkpoint"), ("node_threshold", "ged_by_threshold")):
        figure, axis = plt.subplots(figsize=(7, 4)); frame.boxplot(column="upper_bound", by=column, ax=axis); figure.suptitle(""); figure.tight_layout()
        for suffix in ("png", "pdf"): figure.savefig(destination / f"{name}.{suffix}", dpi=180)
        plt.close(figure)
    if len(ranking):
        summary = ranking.groupby("structural_metric")[["ged_agrees", "loss_agrees"]].mean()
        axis = summary.plot.bar(figsize=(8, 4)); axis.set_ylabel("ranking agreement"); axis.figure.tight_layout()
        for suffix in ("png", "pdf"): axis.figure.savefig(destination / f"ranking_agreement.{suffix}", dpi=180)
        plt.close(axis.figure)
    if len(cases):
        identities = set()
        for row in cases.itertuples():
            identities.add((str(row.sample_id), int(row.epoch_a), str(row.ged_variant), float(row.node_threshold), float(row.edge_threshold)))
            identities.add((str(row.sample_id), int(row.epoch_b), str(row.ged_variant), float(row.node_threshold), float(row.edge_threshold)))
        highlighted = np.asarray([
            (str(row.sample_id), int(row.epoch), str(row.ged_variant), float(row.node_threshold), float(row.edge_threshold)) in identities
            for row in frame.itertuples()
        ])
        figure, axis = plt.subplots(figsize=(6.5, 5))
        axis.scatter(frame.loc[~highlighted, "audit_composite_proxy"], frame.loc[~highlighted, "upper_bound"], s=15, alpha=.25, color="grey", label="other predictions")
        axis.scatter(frame.loc[highlighted, "audit_composite_proxy"], frame.loc[highlighted, "upper_bound"], s=42, alpha=.9, color="crimson", label="matched-loss disagreement")
        axis.set(xlabel="audit loss proxy", ylabel="GED / incumbent upper bound"); axis.legend(); figure.tight_layout()
        for suffix in ("png", "pdf"): figure.savefig(destination / f"matched_loss_disagreement_cases.{suffix}", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True); args = parser.parse_args(); run = Path(args.run)
    config = __import__("yaml").safe_load((run / "resolved_config.yaml").read_text())
    tasks = pd.read_csv(run / "task_manifest.csv"); frame = load_results(run, len(tasks))
    atomic_csv(frame, run / "ged_results.csv.gz", compression="gzip"); atomic_csv(frame, run / "joined_analysis.csv.gz", compression="gzip")
    association_frame = associations(frame); atomic_csv(association_frame, run / "associations.csv")
    optimal = frame[frame.status == "optimal"].copy()
    cv_rows = []; residual_parts = []
    for keys, group in optimal.groupby(["ged_variant", "node_threshold", "edge_threshold"]):
        metrics, residual = grouped_linear_cv(group, "upper_bound", ["node_loss", "edge_loss_proxy", "coordinate_loss"], "sample_id", folds=int(config["analysis"]["cv_folds"]), seed=int(config["analysis"]["seed"]))
        cv_row = {"ged_variant": keys[0], "node_threshold": keys[1], "edge_threshold": keys[2], **metrics}
        if len(residual):
            residual = residual.join(group[["topology_l1", "matched_gt_connectivity_disagreement", "branchpoint_count_error", "false_cycle_closing_edges", "combined_structural_score"]])
            for metric in ("topology_l1", "matched_gt_connectivity_disagreement", "branchpoint_count_error", "false_cycle_closing_edges", "combined_structural_score"):
                cv_row[f"residual_spearman_{metric}"] = correlation(residual, "residual_ged", metric)["spearman"]
            residual_parts.append(residual)
        cv_rows.append(cv_row)
    cv = pd.DataFrame(cv_rows); residuals = pd.concat(residual_parts, ignore_index=True) if residual_parts else pd.DataFrame()
    ranking = ranking_analysis(frame); cases = disagreement_cases(frame, float(config["analysis"]["similar_loss_tolerance"]), float(config["analysis"]["substantial_ged_difference"]))
    atomic_csv(ranking, run / "ranking_results.csv"); atomic_csv(cases, run / "disagreement_cases.csv")
    diagnostics = frame[["task_id", "sample_id", "epoch", "ged_variant", "node_threshold", "edge_threshold", "status", "runtime_seconds", "lower_bound", "upper_bound", "relative_gap", "predicted_node_count", "predicted_edge_count"]]
    atomic_csv(diagnostics, run / "solver_diagnostics.csv")
    bootstrap_rows = []
    for (variant, node_threshold, edge_threshold), group in optimal.groupby(["ged_variant", "node_threshold", "edge_threshold"]):
        for metric in ("audit_composite_proxy", "saved_matching_edit_cost", "topology_l1"):
            bootstrap_rows.append({"ged_variant": variant, "node_threshold": node_threshold, "edge_threshold": edge_threshold, "metric": metric, **grouped_bootstrap_correlation(group, "upper_bound", metric, "sample_id", draws=int(config["analysis"]["bootstrap_draws"]), seed=int(config["analysis"]["seed"]))})
    if len(ranking):
        ranking_numeric = ranking.dropna(subset=["ged_agrees", "loss_agrees"]).copy()
        if len(ranking_numeric):
            ranking_numeric["agreement_advantage"] = ranking_numeric.ged_agrees.astype(float) - ranking_numeric.loss_agrees.astype(float)
            for metric, group in ranking_numeric.groupby("structural_metric"):
                grouped = list(group.groupby("sample_id", sort=True)); rng = np.random.default_rng(int(config["analysis"]["seed"])); draws = []
                for _ in range(int(config["analysis"]["bootstrap_draws"])):
                    chosen = rng.integers(len(grouped), size=len(grouped)); draws.append(float(pd.concat([grouped[int(index)][1] for index in chosen]).agreement_advantage.mean()))
                bootstrap_rows.append({
                    "ged_variant": "all", "node_threshold": np.nan, "edge_threshold": np.nan,
                    "metric": f"ranking_advantage_{metric}", "estimate": float(group.agreement_advantage.mean()),
                    "low": float(np.quantile(draws, .025)), "high": float(np.quantile(draws, .975)),
                    "groups": len(grouped), "draws": len(draws),
                })
    atomic_csv(pd.DataFrame(bootstrap_rows), run / "bootstrap_intervals.csv")
    plots(frame, residuals, ranking, cases, run / "plots")
    optimal_rate = float((frame.status == "optimal").mean()); median_runtime = float(frame.runtime_seconds.median())
    cv_r2 = pd.to_numeric(cv.get("r2", pd.Series(dtype=float)), errors="coerce").dropna()
    saved_improvement = optimal.saved_matching_edit_cost - optimal.upper_bound
    ranking_complete = ranking.dropna(subset=["ged_agrees", "loss_agrees"]) if len(ranking) else pd.DataFrame()
    ged_ranking = float(ranking_complete.ged_agrees.astype(float).mean()) if len(ranking_complete) else None
    loss_ranking = float(ranking_complete.loss_agrees.astype(float).mean()) if len(ranking_complete) else None
    ranking_advantage = (ged_ranking - loss_ranking) if ged_ranking is not None else None
    threshold_ranges = frame.groupby(["sample_id", "epoch", "ged_variant"]).upper_bound.agg(lambda value: value.max() - value.min())
    typical_ged = float(frame.upper_bound.dropna().median()) if frame.upper_bound.notna().any() else None
    relative_threshold_range = float(threshold_ranges.median() / max(typical_ged, 1e-12)) if typical_ged is not None else None
    residual_topology = float(pd.to_numeric(cv.get("residual_spearman_combined_structural_score", pd.Series(dtype=float)), errors="coerce").abs().median()) if "residual_spearman_combined_structural_score" in cv else None
    if residual_topology is not None and not np.isfinite(residual_topology): residual_topology = None
    stage_name = str(config["stage"]["name"])
    signals = {
        "not_redundant": bool(len(cv_r2) and float(cv_r2.median()) < .90),
        "mapping_material": bool(len(saved_improvement) and float(saved_improvement.median()) > .5),
        "ranking_advantage": bool(ranking_advantage is not None and ranking_advantage > .05),
        "threshold_stable": bool(relative_threshold_range is not None and relative_threshold_range < .25),
        "adequate_optimal_labels": optimal_rate >= .70,
        "structural_residual": bool(residual_topology is not None and residual_topology > .20),
    }
    if stage_name == "smoke": decision = "pipeline_only_no_scientific_decision"
    elif sum(signals.values()) >= 5: decision = "go_evidence_supports_considering_encoder_labels"
    elif (len(cv_r2) and float(cv_r2.median()) >= .95 and len(saved_improvement) and float(saved_improvement.median()) < .1) or optimal_rate < .30: decision = "no_go"
    else: decision = "inconclusive"
    summary = {
        "rows": len(frame), "samples": int(frame.sample_id.nunique()), "optimal_rate": optimal_rate,
        "median_runtime_seconds": median_runtime, "median_relative_gap": float(frame.relative_gap.dropna().median()) if frame.relative_gap.notna().any() else None,
        "cross_validated_models": json.loads(cv.to_json(orient="records")), "disagreement_cases": len(cases),
        "median_cross_validated_r2": float(cv_r2.median()) if len(cv_r2) else None,
        "median_saved_matching_improvement": float(saved_improvement.median()) if len(saved_improvement) else None,
        "fraction_optimized_better_than_saved_matching": float((saved_improvement > 1e-6).mean()) if len(saved_improvement) else None,
        "ged_structural_ranking_agreement": ged_ranking, "loss_structural_ranking_agreement": loss_ranking,
        "ranking_agreement_advantage": ranking_advantage, "relative_median_threshold_range": relative_threshold_range,
        "median_abs_residual_structural_spearman": residual_topology,
        "decision_signals": signals, "go_no_go_decision": decision,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminology": {"optimal": "certified GED", "timeout": "GED incumbent upper bound"},
    }
    atomic_json(run / "summary.json", summary)
    def shown(value, suffix=""):
        return "unavailable" if value is None or not np.isfinite(value) else f"{value:.3f}{suffix}"
    report = f"""# Conventional GED go/no-go report

This study uses saved real RelationFormer validation predictions, not handmade corruptions.

## Solver evidence

- Optimal solve rate: {optimal_rate:.1%}
- Median runtime: {median_runtime:.2f} seconds
- Non-optimal incumbents are reported only as GED upper bounds.

## Decision questions

1. **Predictable from existing losses?** Median grouped-CV R² is {shown(float(cv_r2.median()) if len(cv_r2) else None)}. High held-out R² is evidence against continuing.
2. **Different from saved matching?** Median certified improvement over the constrained saved matching is {shown(float(saved_improvement.median()) if len(saved_improvement) else None)} edits; the optimized cost is lower in {shown(100 * float((saved_improvement > 1e-6).mean()) if len(saved_improvement) else None, '%')} of certified rows.
3. **Better structural ranking?** GED agreement is {shown(ged_ranking)} versus {shown(loss_ranking)} for the audit-loss proxy; advantage is {shown(ranking_advantage)}.
4. **Stable across thresholds?** Median within-prediction threshold range divided by typical GED is {shown(relative_threshold_range)}.
5. **Geometric value beyond coordinate loss?** Structural/geometric grouped-CV rows and their held-out structural residual correlations are recorded in `summary.json`; median absolute residual structural Spearman is {shown(residual_topology)}.
6. **How often truly optimal?** {optimal_rate:.1%}; all other finite results are incumbents.
7. **Train a GREED-style encoder?** Automated staged assessment: `{decision}`. Only residual information, correspondence changes, ranking, threshold stability, and solver label quality jointly support a go decision.

## Loss limitation

Node classification, matched-coordinate L1, and cardinality are reconstructed from saved tensors. Box predictions and the training-time random directed relation sampling were not saved, so edge and total values are explicitly audit proxies, not exact RelationFormer training loss.
"""
    (run / "report.md").write_text(report)
    atomic_json(run / "COMPLETED.json", {"status": "complete", "summary": "summary.json", "report": "report.md"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
