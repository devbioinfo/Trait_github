"""Gene-loss matrix (genes x species) from TOGA's per-assembly ortholog classification.

For each selected species, downloads loss_summ_data.tsv.gz of the chosen assembly (hg38
reference) and reads the TRANSCRIPT-level status. A gene's status in a species is that of
its most intact transcript (TOGA's own gene-level rule), in the order
    I (intact) > PI (partially intact) > UL (uncertain loss) > L (lost) > PM, M, PG (missing).
Output (tab-separated, gzipped): one row per gene symbol, one column per species, values
I/PI/UL/L/M; the human reference column is all I.

Usage: python -I scripts/toga_loss_matrix.py --species species.tsv --overview overview.table.tsv \
           --cache /path/loss_tables --out loss_status.tsv.gz
"""
from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

BASE = "https://genome.senckenberg.de/download/TOGA/human_hg38_reference"
RANK = {"I": 0, "PI": 1, "UL": 2, "L": 3, "PM": 4, "M": 4, "PG": 4}
SHORT = {"PM": "M", "PG": "M"}


def fetch(url: str, dest: Path | None = None, tries: int = 4) -> str:
    for k in range(tries):
        cmd = ["curl", "-sfL", "--max-time", "600", url] + (["-o", str(dest)] if dest else [])
        r = subprocess.run(cmd, capture_output=True, text=dest is None)
        if r.returncode == 0:
            return r.stdout if dest is None else ""
        time.sleep(2 ** (k + 1))
    raise RuntimeError(f"download failed: {url}")


def assembly_dirs(orders: set[str]) -> dict[str, str]:
    """assembly name -> '<order>/<Species__common__assembly>' from the directory listings."""
    out = {}
    for order in sorted(orders):
        for d in re.findall(r'href="([^"?/][^"]*)/"', fetch(f"{BASE}/{order}/")):
            out[d.rsplit("__", 1)[-1]] = f"{order}/{d}"
    return out


def gene_status(path: Path) -> pd.Series:
    best: dict[str, str] = {}
    with gzip.open(path, "rt") as fh:
        for ln in fh:
            if not ln.startswith("TRANSCRIPT"):
                continue
            _, tx, s = ln.rstrip("\n").split("\t")[:3]
            sym = tx.split(".", 1)[1]
            if sym not in best or RANK.get(s, 4) < RANK.get(best[sym], 4):
                best[sym] = s
    return pd.Series({g: SHORT.get(s, s) for g, s in best.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--overview", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    pairs = [ln.split("\t") for ln in Path(args.species).read_text().splitlines() if ln.strip()]
    query = [(sp, asm) for sp, asm in pairs if asm != "REFERENCE"]
    # TOGA groups assemblies into top-level directories named after orders or clades
    top = set(re.findall(r'href="([A-Z][A-Za-z]+)/"', fetch(f"{BASE}/"))) - {"Matrix", "MultipleCodonAlignments"}
    dirs = assembly_dirs(top)
    cols = {}
    for sp, asm in query:
        dest = cache / f"{asm}.loss_summ_data.tsv.gz"
        if not dest.exists():
            if asm not in dirs:
                raise SystemExit(f"no TOGA directory for {sp} ({asm})")
            fetch(f"{BASE}/{dirs[asm]}/loss_summ_data.tsv.gz", dest.with_suffix(".part"))
            dest.with_suffix(".part").rename(dest)
        cols[sp] = gene_status(dest)
        print(f"{sp}: {(cols[sp] == 'L').sum()} lost, {(cols[sp] == 'UL').sum()} uncertain", flush=True)
    m = pd.DataFrame(cols).fillna("M")
    ref = [sp for sp, asm in pairs if asm == "REFERENCE"]
    for sp in ref:
        m[sp] = "I"
    m = m[[sp for sp, _ in pairs]]
    m.index.name = "symbol"
    m.to_csv(args.out, sep="\t", compression="gzip")
    print(f"{m.shape[0]} genes x {m.shape[1]} species -> {args.out}")


if __name__ == "__main__":
    main()
