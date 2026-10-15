from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def associations(frame: pd.DataFrame, x: str, y: str) -> dict[str, float | int]:
    clean=frame[[x,y]].dropna(); n=len(clean)
    if n<2:return {"n":n,"pearson":float("nan"),"spearman":float("nan"),"ties":0}
    return {"n":n,"pearson":float(stats.pearsonr(clean[x],clean[y]).statistic),"spearman":float(stats.spearmanr(clean[x],clean[y]).statistic),"ties":int(clean[x].duplicated().sum())}


def cluster_bootstrap_mean(frame: pd.DataFrame, value: str, cluster: str, seed: int=0, draws: int=1000) -> dict[str,float]:
    groups=list(frame.groupby(cluster));rng=np.random.default_rng(seed);means=[]
    if len(groups)<2:return {"mean":float(frame[value].mean()),"low":float("nan"),"high":float("nan"),"clusters":len(groups)}
    for _ in range(draws):
        chosen=rng.integers(len(groups),size=len(groups));sample=pd.concat([groups[i][1] for i in chosen]);means.append(sample[value].mean())
    return {"mean":float(frame[value].mean()),"low":float(np.quantile(means,.025)),"high":float(np.quantile(means,.975)),"clusters":len(groups)}

