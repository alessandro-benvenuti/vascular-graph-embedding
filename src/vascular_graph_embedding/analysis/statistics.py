from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def associations(frame: pd.DataFrame, x: str, y: str) -> dict[str, float | int]:
    clean=frame[[x,y]].dropna(); n=len(clean)
    if n<2:return {"n":n,"pearson":float("nan"),"spearman":float("nan"),"ties":0}
    return {"n":n,"pearson":float(stats.pearsonr(clean[x],clean[y]).statistic),"spearman":float(stats.spearmanr(clean[x],clean[y]).statistic),"ties":int(clean[x].duplicated().sum())}


def cluster_bootstrap_mean(frame: pd.DataFrame, value: str, cluster: str, seed: int=0, draws: int=1000) -> dict[str,float]:
    aggregates=frame.groupby(cluster)[value].agg(["sum","count"]);rng=np.random.default_rng(seed);clusters=len(aggregates)
    if clusters<2:return {"mean":float(frame[value].mean()),"low":float("nan"),"high":float("nan"),"clusters":clusters}
    sums=aggregates["sum"].to_numpy(float);counts=aggregates["count"].to_numpy(float);chosen=rng.integers(clusters,size=(draws,clusters));means=sums[chosen].sum(axis=1)/counts[chosen].sum(axis=1)
    return {"mean":float(frame[value].mean()),"low":float(np.quantile(means,.025)),"high":float(np.quantile(means,.975)),"clusters":clusters}

