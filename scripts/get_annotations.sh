#!/bin/bash
# Downloads the annotation files used by scripts/function_enrichment.py and records their versions.
set -e
D=${1:-data/annot}; mkdir -p "$D"; cd "$D"
[ -f rna_tissue_consensus.tsv ] || { curl -fsSL -o hpa.zip https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip && unzip -o hpa.zip && rm hpa.zip; }
[ -f ReactomePathways.gmt ] || { curl -fsSL -o rea.zip https://reactome.org/download/current/ReactomePathways.gmt.zip && unzip -o rea.zip && rm rea.zip; }
[ -f goa_human.gaf.gz ] || curl -fsSL -o goa_human.gaf.gz https://current.geneontology.org/annotations/goa_human.gaf.gz
[ -f go-basic.obo ] || curl -fsSL -o go-basic.obo https://purl.obolibrary.org/obo/go/go-basic.obo
{ date; for f in *; do echo "$f $(shasum -a 256 "$f" | cut -c1-16)"; done; grep -m1 "^data-version" go-basic.obo || true; } > VERSIONS.txt
cat VERSIONS.txt
