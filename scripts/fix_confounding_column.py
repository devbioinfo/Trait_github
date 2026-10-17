"""Add the confounding level to benchmark_v2 output written before it was recorded.

Within each replicate, baseline rows were written as three consecutive blocks
(confounding 0, 1, 2), each covering every effect size and method.
Usage: python -I scripts/fix_confounding_column.py results/benchmark_v2.csv
"""
import sys

import numpy as np
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)
if "confounding" not in df:
    df["confounding"] = 1.0
    for rep, idx in df[df.scenario == "baseline"].groupby("rep").groups.items():
        idx = np.sort(np.asarray(idx))
        block = len(idx) // 3
        df.loc[idx, "confounding"] = np.repeat([0.0, 1.0, 2.0], block)
    df.to_csv(path, index=False)
print(df.groupby(["scenario", "confounding"]).size())
