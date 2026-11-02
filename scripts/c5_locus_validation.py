"""C5: independent sequence validation of the LGSN and CRYGS losses (run on the Mac).

For each species that lost LGSN or CRYGS, the assembly TOGA used and, where they exist, independently produced
assemblies of the same species are searched with miniprot (protein-to-genome alignment, an annotation method
independent of TOGA) for the human LGSN and CRYGS proteins and for the three nearest protein-coding neighbours on
each side of each gene in the human genome (within 2 Mb). For the best hit of each target protein the script reports:
  - stop codons and frameshifts, found by translating the aligned coding segments (residue = human protein position);
  - protein coverage, identity, N content of the locus and the soft-masked (repeat) fraction of the coding bases;
  - the second-best hit elsewhere in the genome (paralogy check);
  - synteny: how many of the six human neighbours align to the same scaffold within 3 Mb;
and compares each lesion with TOGA's inactivating mutations for the same species (unmasked stop codons and
frameshifts; a lesion matches if it lies within 2 codons).

Inputs downloaded on first run (cached in data/c5/): human proteins (Ensembl REST), genomes (NCBI FTP), TOGA
geneInactivatingMutations (hg38 reference).
Outputs: results/c5/locus_validation.csv, results/c5/lesions.csv, results/c5/summary.md, results/c5/VERSIONS.txt
Requires: miniprot on the PATH (conda install -c bioconda miniprot).
Usage: python3 scripts/c5_locus_validation.py [--threads 8]
"""
import argparse
import gzip
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from marine_detail import read_mutations  # noqa: E402
from toga_loss_matrix import BASE, assembly_dirs, fetch  # noqa: E402

DATA = ROOT / "data" / "c5"
OUT = ROOT / "results" / "c5"
TARGETS = {"LGSN": "ENST00000370657", "CRYGS": "ENST00000392499"}
# species: (TOGA assembly name, genes TOGA calls lost, [(accession, assembly name, role)])
SPECIES = {
    "Heterocephalus_glaber": ("HLhetGla3", ["LGSN"], [
        ("GCA_000247695.1", "HetGla_female_1.0", "short-read assembly (Broad 2012); TOGA used the DNA Zoo Hi-C version HLhetGla3, presumably built on it"),
        ("GCA_988278965.1", "mHetGla1.1.hap1", "independent long-read assembly (2026)"),
        ("GCA_053883595.1", "H.glaber_assembly_v1.0", "independent long-read assembly (2025)")]),
    "Fukomys_damarensis": ("HLfukDam2", ["LGSN"], [
        ("GCA_012274545.1", "DMR_v1.0_HiC", "assembly used by TOGA (no independent assembly exists)")]),
    "Nannospalax_galili": ("nanGal1", ["LGSN", "CRYGS"], [
        ("GCA_000622305.1", "S.galili_v1.0", "assembly used by TOGA (no independent assembly exists)")]),
    "Condylura_cristata": ("conCri1", ["LGSN", "CRYGS"], [
        ("GCA_000260355.1", "ConCri1.0", "assembly used by TOGA (no independent assembly exists)")]),
    "Scalopus_aquaticus": ("HLscaAqu1", ["LGSN", "CRYGS"], [
        ("GCA_004024925.1", "ScaAqu_v1_BIUU", "assembly used by TOGA (no independent assembly exists)")]),
    "Chrysochloris_asiatica": ("chrAsi1", ["LGSN", "CRYGS"], [
        ("GCA_000296735.1", "ChrAsi1.0", "assembly used by TOGA"),
        ("GCA_004027935.1", "ChrAsi_v1_BIUU", "independent assembly (Broad 2019)")]),
}
ORDERS = {"Rodentia", "Eulipotyphla", "Afrotheria"}
CODON = {a + b + c: aa for (a, b, c), aa in zip(
    [(x, y, z) for x in "TCAG" for y in "TCAG" for z in "TCAG"],
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")}
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def rest(path, tries=5):
    """Ensembl REST GET with retries (the service returns occasional 5xx errors under load)."""
    import time
    for k in range(tries):
        try:
            req = urllib.request.Request("https://rest.ensembl.org" + path, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise RuntimeError(f"Ensembl REST failed for {path}: {e}")
            time.sleep(5 * (k + 1))


def human_proteins():
    """Target proteins plus the three nearest protein-coding neighbours on each side (Ensembl, GRCh38)."""
    f = DATA / "human_proteins.faa"
    meta = DATA / "human_proteins.json"
    if f.exists() and meta.exists():
        return json.loads(meta.read_text())
    recs, info = [], {}
    for gene, tx in TARGETS.items():
        t = rest(f"/lookup/id/{tx}")
        recs.append((f"{gene}|target|{tx}", rest(f"/sequence/id/{tx}?type=protein")["seq"]))
        chrom, start, end = t["seq_region_name"], t["start"], t["end"]
        # Ensembl limits region queries to 5 Mb, so the window is +-2 Mb around the gene
        genes = rest(f"/overlap/region/human/{chrom}:{max(1, start - 2_000_000)}-{end + 2_000_000}"
                     f"?feature=gene;biotype=protein_coding")
        genes = [g for g in genes if g.get("external_name") not in (gene, None)]
        up = sorted([g for g in genes if g["end"] < start], key=lambda g: start - g["end"])[:3]
        down = sorted([g for g in genes if g["start"] > end], key=lambda g: g["start"] - end)[:3]
        info[gene] = {"chrom": chrom, "start": start, "end": end, "neighbours": []}
        for side, lst in (("up", up), ("down", down)):
            for k, g in enumerate(lst, 1):
                full = rest(f"/lookup/id/{g['id']}?expand=1")
                can = [x for x in full.get("Transcript", []) if x.get("is_canonical")]
                if not can:
                    continue
                seq = rest(f"/sequence/id/{can[0]['id']}?type=protein")["seq"]
                name = f"{gene}|nbr_{side}{k}|{g['external_name']}"
                recs.append((name, seq))
                info[gene]["neighbours"].append(name)
    f.write_text("".join(f">{n}\n{s}\n" for n, s in recs))
    info["lengths"] = {n: len(s) for n, s in recs}
    meta.write_text(json.dumps(info, indent=1))
    return info


def genome(acc, name):
    d = DATA / "genomes"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{acc}_{name}_genomic.fna.gz"
    if f.exists():
        return f
    p = acc[:3] + "/" + "/".join(re.findall(r"\d{3}", acc.split("_")[1].split(".")[0]))
    listing = subprocess.run(["curl", "-sfL", f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{p}/"],
                             capture_output=True, text=True).stdout
    dirs = [x for x in re.findall(r'href="([^"/]+)/"', listing) if x.startswith(acc)]
    if not dirs:
        raise SystemExit(f"NCBI folder for {acc} not found")
    url = f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{p}/{dirs[0]}/{dirs[0]}_genomic.fna.gz"
    print("downloading", url, flush=True)
    subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(f) + ".part", url], check=True)
    Path(str(f) + ".part").rename(f)
    return f


def run_miniprot(gfile, threads):
    out = gfile.with_suffix(".miniprot.gff")
    if not out.exists():
        with open(out.with_suffix(".part"), "w") as fh:
            subprocess.run(["miniprot", "-t", str(threads), "-I", "--gff", str(gfile), str(DATA / "human_proteins.faa")],
                           stdout=fh, check=True)
        out.with_suffix(".part").rename(out)
    return out


def parse_gff(path):
    mrna, cds = {}, {}
    for ln in open(path):
        if ln.startswith("#"):
            continue
        f = ln.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        a = dict(kv.split("=", 1) for kv in f[8].split(";") if "=" in kv)
        tgt = a.get("Target", "").split()
        if f[2] == "mRNA":
            mrna[a["ID"]] = dict(id=a["ID"], seq=f[0], start=int(f[3]), end=int(f[4]), strand=f[6],
                                 rank=int(a.get("Rank", 99)), identity=float(a.get("Identity", "nan")),
                                 protein=tgt[0], pstart=int(tgt[1]), pend=int(tgt[2]))
        elif f[2] == "CDS":
            cds.setdefault(a["Parent"], []).append(dict(start=int(f[3]), end=int(f[4]), phase=int(f[7]) if f[7] != "." else 0,
                                                        ps=int(tgt[1]), pe=int(tgt[2])))
    return mrna, cds


def load_scaffolds(gfile, names):
    seqs, cur, buf = {}, None, []
    with gzip.open(gfile, "rt") as fh:
        for ln in fh:
            if ln.startswith(">"):
                if cur:
                    seqs[cur] = "".join(buf)
                n = ln[1:].split()[0]
                cur, buf = (n if n in names else None), []
            elif cur:
                buf.append(ln.strip())
        if cur:
            seqs[cur] = "".join(buf)
    return seqs


def lesions(m, segs, seq):
    """Stops and frameshifts along the aligned coding segments, in human protein positions."""
    segs = sorted(segs, key=lambda s: s["ps"])
    out, cds_bases, lower = [], 0, 0
    for i, s in enumerate(segs):
        nt = seq[s["start"] - 1:s["end"]]
        if m["strand"] == "-":
            nt = nt.translate(COMP)[::-1]
        cds_bases += len(nt)
        lower += sum(c.islower() for c in nt)
        nt = nt.upper()
        for k in range(s["phase"], len(nt) - 2, 3):
            if CODON.get(nt[k:k + 3]) == "*":
                out.append(("stop", s["ps"] + (k - s["phase"]) // 3 + (1 if s["phase"] else 0)))
        if i:
            p = segs[i - 1]
            gap = (s["start"] - p["end"] - 1) if m["strand"] == "+" else (p["start"] - s["end"] - 1)
            if gap < 30:
                out.append(("frameshift", p["pe"]))
    return out, cds_bases, lower


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ver = subprocess.run(["miniprot", "--version"], capture_output=True, text=True).stdout.strip()
    info = human_proteins()
    lengths = info["lengths"]
    # TOGA inactivating mutations of the species' TOGA assemblies
    dirs = assembly_dirs(ORDERS)
    toga = []
    for sp, (tasm, lost, _) in SPECIES.items():
        f = DATA / f"{tasm}.geneInactivatingMutations.tsv.gz"
        if not f.exists():
            fetch(f"{BASE}/{dirs[tasm]}/geneInactivatingMutations.tsv.gz", f.with_suffix(".part"))
            f.with_suffix(".part").rename(f)
        for r in read_mutations(f, set(TARGETS)):
            if not r["masked"] and any(k in r["type"].upper() for k in ("STOP", "FS_")):
                toga.append(dict(species=sp, gene=r["gene"], codon=r["codon"],
                                 kind="stop" if "STOP" in r["type"].upper() else "frameshift", chain=r["chain"]))
    toga = pd.DataFrame(toga)
    rows, lrows = [], []
    for sp, (tasm, lost, asms) in SPECIES.items():
        for acc, name, role in asms:
            g = genome(acc, name)
            mrna, cds = parse_gff(run_miniprot(g, args.threads))
            by_prot = {}
            for m in mrna.values():
                by_prot.setdefault(m["protein"], []).append(m)
            for v in by_prot.values():
                v.sort(key=lambda m: (m["rank"], -m["identity"]))
            best = {gene: by_prot.get(f"{gene}|target|{tx}", []) for gene, tx in TARGETS.items()}
            need = {v[0]["seq"] for v in best.values() if v}
            seqs = load_scaffolds(g, need)
            for gene, tx in TARGETS.items():
                hits = best[gene]
                r = dict(species=sp, gene=gene, toga_called_lost=gene in lost, accession=acc, assembly=name, role=role,
                         miniprot=ver)
                if not hits:
                    rows.append(dict(r, found=False))
                    continue
                m = hits[0]
                seq = seqs[m["seq"]]
                les, nb, low = lesions(m, cds[m["id"]], seq)
                span = seq[m["start"] - 1:m["end"]]
                cov = sum(c["pe"] - c["ps"] + 1 for c in cds[m["id"]]) / lengths[f"{gene}|target|{tx}"]
                second = next((h for h in hits[1:] if h["seq"] != m["seq"] or abs(h["start"] - m["start"]) > 100_000), None)
                syn = 0
                for nb_name in info[gene]["neighbours"]:
                    for h in by_prot.get(nb_name, [])[:1]:
                        if h["seq"] == m["seq"] and min(abs(h["start"] - m["end"]), abs(m["start"] - h["end"])) < 3_000_000:
                            syn += 1
                tg = toga[(toga.species == sp) & (toga.gene == gene)] if len(toga) else pd.DataFrame()
                matched = sum(any(abs(t.codon - p) <= 2 and t.kind == k for k, p in les) for t in tg.itertuples())
                rows.append(dict(r, found=True, scaffold=m["seq"], start=m["start"], end=m["end"], strand=m["strand"],
                                 identity=m["identity"], protein_coverage=round(min(cov, 1), 3),
                                 stops=sum(k == "stop" for k, _ in les), frameshifts=sum(k == "frameshift" for k, _ in les),
                                 lesion_positions=";".join(f"{k}@{p}" for k, p in les),
                                 toga_lesions=len(tg), toga_lesions_reproduced=matched,
                                 locus_N_fraction=round(span.upper().count("N") / max(len(span), 1), 4),
                                 cds_softmasked_fraction=round(low / max(nb, 1), 3),
                                 synteny_neighbours_same_scaffold=f"{syn}/{len(info[gene]['neighbours'])}",
                                 second_hit=(f"{second['seq']}:{second['start']} identity {second['identity']}" if second else "")))
                for k, p in les:
                    lrows.append(dict(species=sp, gene=gene, assembly=name, role=role, kind=k, position=p,
                                      in_toga=any(abs(t.codon - p) <= 2 and t.kind == k for t in tg.itertuples())))
            print("done", sp, name, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "locus_validation.csv", index=False)
    pd.DataFrame(lrows).to_csv(OUT / "lesions.csv", index=False)
    toga.to_csv(OUT / "toga_lesions.csv", index=False)
    (OUT / "VERSIONS.txt").write_text(f"{ver}\npython {sys.version.split()[0]}\npandas {pd.__version__}\n"
                                      f"Ensembl REST (GRCh38) proteins: data/c5/human_proteins.faa\n")
    cols = ["species", "gene", "toga_called_lost", "assembly", "role", "found", "identity", "protein_coverage", "stops",
            "frameshifts", "toga_lesions", "toga_lesions_reproduced", "locus_N_fraction", "cds_softmasked_fraction",
            "synteny_neighbours_same_scaffold"]
    s = t[[c for c in cols if c in t]].to_string(index=False)
    (OUT / "summary.md").write_text("# C5 locus validation\n\n```\n" + s + "\n```\n")
    print(s)


if __name__ == "__main__":
    main()
