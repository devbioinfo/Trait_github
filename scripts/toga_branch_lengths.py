"""Per-gene branch lengths on the species tree from TOGA multiple codon alignments.

For each gene alignment (human hg38 reference, TOGA MultipleCodonAlignments):
  1. keep the selected species that have exactly one sequence (the REFERENCE is human);
  2. skip the gene if fewer than --min-frac of the selected species remain;
  3. prune the Zoonomia species tree to those species and fit branch lengths on that fixed
     topology with IQ-TREE 2: GTR+G4 on codon sites and LG+G4 on the translated protein;
  4. map every branch of the pruned tree to the master tree (selected species). A master
     branch receives a length only when it maps one-to-one onto a pruned branch; branches
     merged by pruning, and branches above absent species, are missing (NaN).
Results are appended to a JSON-lines file (one gene per line) so the run can resume.

Usage:
  python -I scripts/toga_branch_lengths.py --species species.tsv --out-dir results/toga \
      --mca /path/MultipleCodonAlignments --tree hg38.447way.scientificNames.nh.txt
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlconv.tree import from_newick, prune_newick  # noqa: E402

IQTREE = shutil.which("iqtree2") or shutil.which("iqtree3") or "iqtree2"
CODON = {a + b + c: aa for (a, b, c), aa in zip(
    ((x, y, z) for x in "TCAG" for y in "TCAG" for z in "TCAG"),
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")}

CTX: dict = {}


def init(ctx: dict) -> None:
    CTX.update(ctx)


def translate(seq: str) -> str:
    out = []
    for i in range(0, len(seq) - 2, 3):
        c = seq[i:i + 3].upper()
        out.append("-" if c == "---" else CODON.get(c, "X"))
    return "".join(out).replace("*", "X")


def read_alignment(path: Path, asm2sp: dict) -> dict[str, str]:
    """Sequences of the chosen assembly of each selected species (1:1 orthologs only)."""
    seqs: dict[str, list] = {}
    seen: dict[str, int] = {}
    cur = None
    with gzip.open(path, "rt") as fh:
        for ln in fh:
            if ln.startswith(">"):
                lab = ln[1:].split()[0]
                cur = asm2sp.get(lab[3:] if lab.startswith("vs_") else lab)
                if cur:
                    seen[cur] = seen.get(cur, 0) + 1
                    seqs.setdefault(cur, [])
            elif cur:
                seqs[cur].append(ln.strip())
    return {k: "".join(v) for k, v in seqs.items() if seen[k] == 1}   # 1:1 orthologs only


def run_iqtree(aln: Path, tree: Path, model: str, prefix: str, cwd: Path) -> str:
    subprocess.run([IQTREE, "-s", str(aln), "-te", str(tree), "-m", model, "-nt", "1",
                    "--prefix", prefix, "--redo", "-quiet"], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return (cwd / f"{prefix}.treefile").read_text().strip()


def process(path_str: str) -> dict | None:
    path = Path(path_str)
    gene = path.name.replace(".fasta.gz", "")
    try:
        seqs = read_alignment(path, CTX["asm2sp"])
    except (OSError, EOFError):
        return {"gene": gene, "status": "unreadable"}
    present = sorted(seqs)
    if len(present) < CTX["min_n"]:
        return {"gene": gene, "status": "too_few_species", "n": len(present)}
    L = len(seqs[present[0]])
    cols = [i for i in range(0, L - L % 3, 3)
            if sum(seqs[s][i:i + 3] not in ("---", "NNN") for s in present) >= 4]
    if len(cols) < 50:
        return {"gene": gene, "status": "too_short", "codons": len(cols)}
    nt = {s: "".join(seqs[s][i:i + 3] for i in cols) for s in present}
    aa = {s: translate(v) for s, v in nt.items()}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "nt.fa").write_text("".join(f">{s}\n{v}\n" for s, v in nt.items()))
        (tmp / "aa.fa").write_text("".join(f">{s}\n{v}\n" for s, v in aa.items()))
        (tmp / "t.nwk").write_text(prune_newick(CTX["tree"], CTX["names"], set(present)) + "\n")
        try:
            t_nt = run_iqtree(tmp / "nt.fa", tmp / "t.nwk", "GTR+G4", "nt", tmp)
            t_aa = run_iqtree(tmp / "aa.fa", tmp / "t.nwk", "LG+G4", "aa", tmp)
        except subprocess.CalledProcessError:
            return {"gene": gene, "status": "iqtree_failed"}
    idx = CTX["index"]
    present_idx = frozenset(idx[s] for s in present)
    anchor = min(present_idx)

    def canon(S: frozenset) -> frozenset:
        """Split of the present species, identified by the side without the anchor species."""
        return present_idx - S if anchor in S else S

    # master branch -> split of the present species (None if it does not split them)
    keys = []
    for D_b in CTX["master_tipsets"]:
        S = D_b & present_idx
        keys.append(None if (not S or S == present_idx) else canon(S))
    counts: dict = {}
    for k in keys:
        if k is not None:
            counts[k] = counts.get(k, 0) + 1
    out = {"gene": gene, "status": "ok", "n": len(present), "codons": len(cols)}
    for layer, nwk in (("nt", t_nt), ("aa", t_aa)):
        gtree, gnames = from_newick(nwk)
        Dg = gtree.tip_sets()
        est: dict = {}
        for b in range(gtree.n_branches):
            S = frozenset(idx[gnames[t]] for t in np.nonzero(Dg[b])[0])
            if S and S != present_idx:
                k = canon(S)
                est[k] = est.get(k, 0.0) + float(gtree.length[b])   # root edges of a rooted output add up
        # identified only if exactly one master branch maps to this split
        out[layer] = [None if (k is None or counts[k] > 1 or k not in est) else round(est[k], 8)
                      for k in keys]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True,
                    help="species<TAB>assembly per line (Zoonomia tree names; human = REFERENCE)")
    ap.add_argument("--mca", required=True)
    ap.add_argument("--overview", help="unused; kept for compatibility")
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-frac", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [ln.split("\t") for ln in Path(args.species).read_text().splitlines() if ln.strip()]
    species = [sp for sp, _ in pairs]
    asm2sp = {asm: sp for sp, asm in pairs}
    full, full_names = from_newick(Path(args.tree).read_text())
    master_nwk = prune_newick(full, full_names, set(species))
    (out_dir / "master.nwk").write_text(master_nwk + "\n")
    master, names = from_newick(master_nwk)
    index = {n: i for i, n in enumerate(names)}
    D = master.tip_sets()
    tipsets = [frozenset(np.nonzero(D[b])[0].tolist()) for b in range(master.n_branches)]
    ctx = dict(asm2sp=asm2sp, tree=full, names=full_names, index=index,
               master_tipsets=tipsets, min_n=int(np.ceil(args.min_frac * len(species))))

    done_file = out_dir / "branch_lengths.jsonl"
    done = set()
    if done_file.exists():
        done = {json.loads(ln)["gene"] for ln in done_file.read_text().splitlines() if ln.strip()}
    files = sorted(Path(args.mca).glob("*.fasta.gz"))
    todo = [str(f) for f in files if f.name.replace(".fasta.gz", "") not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(species)} species, {master.n_branches} master branches; {len(done)} genes done, "
          f"{len(todo)} to do", flush=True)
    with Pool(args.workers, initializer=init, initargs=(ctx,)) as pool, open(done_file, "a") as fh:
        for i, res in enumerate(pool.imap_unordered(process, todo, chunksize=4), 1):
            if res:
                fh.write(json.dumps(res) + "\n")
                fh.flush()
            if i % 250 == 0:
                print(f"{i}/{len(todo)} genes", flush=True)


if __name__ == "__main__":
    main()
