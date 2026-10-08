"""Cluster uncertainty for saved measurements; no image/model operations.

Percentile intervals resample whole sequences/sources, retaining their rows and
NaNs. The estimand remains the mean of valid pair measurements, not a pool of
keypoints. Synthetic intervals describe only the eight frozen source images.
"""
import numpy as np
import pandas as pd

BOOTSTRAP = {"replicates": 2000, "seed": 0, "confidence": 0.95,
             "method": "percentile cluster bootstrap; resample all rows of a cluster together",
             "hpatches_unit": "sequence, separately within illumination/viewpoint",
             "synthetic_unit": "source_image; eight fixed sources, descriptive uncertainty only"}


def cluster_summary(frame, keys, metrics, cluster, replicates=2000, seed=0):
    if replicates < 1:
        raise ValueError("Positive bootstrap replicate count required")
    if frame[cluster].isna().any():
        raise ValueError("Missing sampling unit")
    rows = []
    for labels, group in frame.groupby(keys, dropna=False, sort=True):
        labels = labels if isinstance(labels, tuple) else (labels,)
        units = sorted(group[cluster].unique())
        rng = np.random.default_rng(seed)
        # Same resampling multiplicities for every metric within a group.
        weights = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=replicates)
        for metric in metrics:
            values = group[metric].astype(float)
            if np.isinf(values).any():
                raise ValueError(f"Infinite measurement: {metric}")
            unit = pd.DataFrame({"unit": group[cluster], "value": values}).groupby("unit").value
            sums = unit.sum().reindex(units).to_numpy(float)
            counts = unit.count().reindex(units).to_numpy(float)
            denominators = weights @ counts
            samples = np.divide(weights @ sums, denominators, out=np.full(replicates, np.nan), where=denominators > 0)
            usable = samples[np.isfinite(samples)]
            valid_units = int((counts > 0).sum())
            low, high = np.quantile(usable, [.025, .975]) if valid_units >= 2 and len(usable) else (np.nan, np.nan)
            remaining = counts.sum() - counts
            leave_one_out = np.divide(sums.sum() - sums, remaining,
                                      out=np.full(len(units), np.nan), where=remaining > 0)
            finite_loo = leave_one_out[np.isfinite(leave_one_out)]
            rows.append({**dict(zip(keys, labels)), "metric": metric, "cluster_unit": cluster,
                         "cluster_count": len(units), "valid_cluster_count": valid_units,
                         "row_count": len(group), "valid_count": int(values.count()),
                         "mean": values.mean(), "median": values.median(), "std": values.std(),
                         "ci95_low": low, "ci95_high": high,
                         "leave_one_cluster_out_min": finite_loo.min() if len(finite_loo) else np.nan,
                         "leave_one_cluster_out_max": finite_loo.max() if len(finite_loo) else np.nan,
                         "bootstrap_replicates": replicates, "bootstrap_valid_replicates": len(usable),
                         "seed": seed})
    return pd.DataFrame(rows)
