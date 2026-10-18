"""Compare signature methods on saved z-scores (from scripts/signatures.py --save-z).

Exact accuracy = share of correctly detected convergent genes whose predicted set of
active layers equals the set implied by how they were simulated.
Usage: python -I scripts/signature_eval.py results/signatures/treeA_z.npz results/signatures/treeB_z.npz
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlconv.configs import fit_configurations  # noqa: E402
from mlconv.signatures import fit_signatures, pattern_active  # noqa: E402

EXPECTED = {"omega": (0, 1, 1), "rate": (1, 1, 0), "both": (1, 1, 1)}
NAMES = ["rate_nucleotide", "rate_protein", "selection_proxy"]

for path in sys.argv[1:]:
    d = np.load(path)
    z, R, kinds, truth, det = d["z"], d["R"], d["kinds"], d["truth"], d["detected"]
    tp = det & truth
    exp = np.array([EXPECTED[k] for k in kinds[tp]], dtype=bool)

    def acc(active):
        a = active[tp]
        by = {k: (a[kinds[tp] == k] == exp[kinds[tp] == k]).all(1).mean() for k in EXPECTED}
        return f"{(a == exp).all(1).mean():.2f} (" + ", ".join(f"{k} {v:.2f}" for k, v in by.items()) + ")"

    sig = fit_signatures(z, R, NAMES)
    ebcm = fit_configurations(z, R, taus=(1, 2, 3, 5, 8, 12, 20, 30))
    print(f"== {Path(path).stem}: {tp.sum()} true detections, {(det & ~truth).sum()} false")
    print("  learned directions (k = 2^L - 1):", acc(pattern_active(sig)))
    print("  EBCM, wide grid:                 ", acc(ebcm.configs[ebcm.map_config]))
    print("  per-layer rule p < 0.05:         ", acc(st.norm.sf(z) < 0.05))
    print("  learned directions with weight > 0.002:")
    for (n, _), w in sorted(zip(sig.patterns, sig.pattern_weights), key=lambda x: -x[1]):
        if n.startswith("learned") and w > 0.002:
            print(f"    {w:.4f}  {n}")
