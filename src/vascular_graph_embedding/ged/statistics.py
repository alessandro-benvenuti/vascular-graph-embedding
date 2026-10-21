from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy import stats


def correlation(frame: pd.DataFrame, left: str, right: str) -> dict[str, float | int]:
    clean = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2 or clean[left].nunique() < 2 or clean[right].nunique() < 2:
        return {"n": len(clean), "pearson": np.nan, "spearman": np.nan}
    return {
        "n": len(clean), "pearson": float(stats.pearsonr(clean[left], clean[right]).statistic),
        "spearman": float(stats.spearmanr(clean[left], clean[right]).statistic),
    }


def grouped_bootstrap_correlation(
    frame: pd.DataFrame, left: str, right: str, group: str,
    *, draws: int = 500, seed: int = 0,
) -> dict[str, float | int]:
    groups = list(frame.groupby(group, sort=True)); rng = np.random.default_rng(seed)
    observed = correlation(frame, left, right)["spearman"]
    values = []
    for _ in range(int(draws)):
        selected = rng.integers(len(groups), size=len(groups)); parts = []
        for replicate, index in enumerate(selected):
            part = groups[int(index)][1].copy(); part[group] = f"bootstrap-{replicate}"; parts.append(part)
        value = correlation(pd.concat(parts, ignore_index=True), left, right)["spearman"]
        if np.isfinite(value): values.append(value)
    return {
        "estimate": observed, "low": float(np.quantile(values, .025)) if values else np.nan,
        "high": float(np.quantile(values, .975)) if values else np.nan,
        "groups": len(groups), "draws": len(values),
    }


def _fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def grouped_linear_cv(
    frame: pd.DataFrame, target: str, predictors: list[str], group: str,
    *, folds: int = 5, seed: int = 0,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    columns = [target, group, *predictors]
    clean = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    unique_groups = clean[group].astype(str).unique()
    folds = min(int(folds), len(unique_groups))
    if folds < 2: return {"n": len(clean), "groups": len(unique_groups), "r2": np.nan, "mae": np.nan}, pd.DataFrame()
    clean["fold"] = clean[group].astype(str).map(lambda value: _fold(value, folds, seed))
    predictions = np.full(len(clean), np.nan)
    x_all = clean[predictors].to_numpy(float); y_all = clean[target].to_numpy(float)
    for fold in range(folds):
        test = clean.fold.to_numpy() == fold; train = ~test
        if not test.any() or train.sum() <= len(predictors): continue
        x_train = np.column_stack((np.ones(train.sum()), x_all[train]))
        coefficients = np.linalg.lstsq(x_train, y_all[train], rcond=None)[0]
        predictions[test] = np.column_stack((np.ones(test.sum()), x_all[test])) @ coefficients
    valid = np.isfinite(predictions); residual = y_all[valid] - predictions[valid]
    denominator = np.square(y_all[valid] - y_all[valid].mean()).sum()
    r2 = 1.0 - np.square(residual).sum() / denominator if denominator > 0 else np.nan
    output = clean.loc[valid].copy(); output["predicted_ged"] = predictions[valid]; output["residual_ged"] = residual
    return {"n": int(valid.sum()), "groups": len(unique_groups), "folds": folds, "r2": float(r2), "mae": float(np.abs(residual).mean())}, output
