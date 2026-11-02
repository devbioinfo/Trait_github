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
    part = str(f) + ".part"
    for attempt in range(1, 11):
        # resume (-C -) after stalls or dropped connections; a transfer below 10 kB/s for 60 s counts as stalled
        r = subprocess.run(["curl", "-fL", "--retry", "5", "--retry-all-errors", "--speed-limit", "10000",
                            "--speed-time", "60", "-C", "-", "-o", part, url])
        if r.returncode == 0:
            break
        print(f"download interrupted (curl exit {r.returncode}); resuming, attempt {attempt + 1} of 10", flush=True)
    else:
        raise SystemExit(f"download of {url} failed after 10 attempts")
    # integrity check before use: a resumed download must decompress completely
    if subprocess.run(["gzip", "-t", part]).returncode != 0:
        Path(part).unlink()
        raise SystemExit(f"{part} is corrupt and was removed; rerun to download it again")
    Path(part).rename(f)
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
                                 mp_frameshift=int(a.get("Frameshift", 0)), mp_stop=int(a.get("StopCodon", 0)),
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


CS_OP = re.compile(r"(:\d+)|(\*[a-z]+[A-Z*])|(\+[A-Z*]+)|(-[a-z]+)|(~[a-z]{2}\d+[a-z]{2})")


def lesions_from_cs(cs, start0):
    """Stops and frameshifts from miniprot's cs tag (human protein positions, 1-based).
    ':N' N matched residues; '*codonA' one codon aligned to residue A; '+AA' residues without genomic sequence;
    '-nnn' genomic bases without residues; '~xxNNNyy' an intron. A partial codon that is completed across an intron
    ('*tE~gt..ag-ga') is a split codon, not a frameshift; any other 1- or 2-nt codon, or a '-' run that is not a
    multiple of 3, is a frameshift; a stop codon aligned to a residue is a premature stop."""
    ops = [m.group(0) for m in CS_OP.finditer(cs)]
    pos, out, i = start0, [], 0
    while i < len(ops):
        op = ops[i]
        if op[0] == ":":
            pos += int(op[1:])
        elif op[0] == "*":
            codon = op[1:-1].upper()
            pos += 1
            if len(codon) < 3 and i + 2 < len(ops) and ops[i + 1][0] == "~" and ops[i + 2][0] == "-" \
                    and len(codon) + len(ops[i + 2]) - 1 == 3:
                codon = codon + ops[i + 2][1:].upper()      # split codon across the intron
                i += 2
            if len(codon) < 3:
                out.append(("frameshift", pos))
            elif codon in ("TAA", "TAG", "TGA"):
                out.append(("stop", pos))
        elif op[0] == "+":
            pos += len(op) - 1
        elif op[0] == "-":
            if (len(op) - 1) % 3:
                out.append(("frameshift", pos))
        i += 1
    return out


def run_miniprot_paf(gfile, threads):
    out = gfile.with_suffix(".miniprot.paf")
    if not out.exists():
        with open(out.with_suffix(".part"), "w") as fh:
            subprocess.run(["miniprot", "-t", str(threads), "-I", str(gfile), str(DATA / "human_proteins.faa")],
                           stdout=fh, check=True)
        out.with_suffix(".part").rename(out)
    best = {}
    for ln in open(out):
        if ln.startswith("#"):
            continue
        f = ln.rstrip("\n").split("\t")
        tags = dict((t[:2], t[5:]) for t in f[12:])
        score = int(tags.get("AS", 0))
        if f[0] not in best or score > best[f[0]]["score"]:
            best[f[0]] = dict(score=score, contig=f[5], start=int(f[7]), end=int(f[8]), pstart0=int(f[2]),
                              fs=int(tags.get("fs", 0)), st=int(tags.get("st", 0)), cs=tags.get("cs", ""))
    return best


def masking(m, segs, seq):
    """Coding bases of the best alignment and how many of them are soft-masked (repeats)."""
    nb = low = 0
    for sg in segs:
        nt = seq[sg["start"] - 1:sg["end"]]
        nb += len(nt)
        low += sum(c.islower() for c in nt)
    return nb, low


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
            paf = run_miniprot_paf(g, args.threads)
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
                pb = paf.get(f"{gene}|target|{tx}")
                les = lesions_from_cs(pb["cs"], pb["pstart0"]) if pb and pb["contig"] == m["seq"] else []
                nb, low = masking(m, cds[m["id"]], seq)
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
                                 miniprot_frameshifts=m["mp_frameshift"], miniprot_stops=m["mp_stop"],
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
    cols = ["species", "gene", "toga_called_lost", "assembly", "found", "identity", "protein_coverage", "stops",
            "frameshifts", "miniprot_stops", "miniprot_frameshifts", "toga_lesions", "toga_lesions_reproduced", "locus_N_fraction", "cds_softmasked_fraction",
            "synteny_neighbours_same_scaffold"]
    s = t[[c for c in cols if c in t]].to_string(index=False)
    (OUT / "summary.md").write_text("# C5 locus validation\n\n```\n" + s + "\n```\n")
    print(s)


if __name__ == "__main__":
    main()
