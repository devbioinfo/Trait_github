"""Choose the species set for the echolocation analysis from TOGA gene coverage.

Rules (fixed before looking at any convergence result):
  * only species in the Zoonomia 447-way tree; one assembly per species (highest coverage);
  * minimum gene coverage (1:1 orthologs) of --min-cov;
  * all bats and all cetaceans that pass, since they define the trait and its closest
    controls (pteropodid bats, baleen whales);
  * shrews (Soricidae), tenrecs (Tenrecidae) and solenodons, which include species with
    simple echolocation, are excluded so that the background holds no echolocators;
  * a backbone of the best-covered species from each other mammalian group, up to the
    quota for that group.

Usage: python -I scripts/toga_select_species.py coverage.tsv out_species.tsv
"""
import sys

import pandas as pd

QUOTAS = [  # (lineage keyword, maximum species)
    ("Chiroptera", 99), ("Cetacea", 99),
    ("Hippopotamidae", 2), ("Ruminantia", 8), ("Suina", 3), ("Tylopoda", 2),
    ("Carnivora", 12), ("Perissodactyla", 4), ("Pholidota", 2), ("Eulipotyphla", 4),
    ("Primates", 8), ("Rodentia", 12), ("Lagomorpha", 3), ("Scandentia", 1), ("Dermoptera", 1),
    ("Afrotheria", 6), ("Xenarthra", 3), ("Metatheria", 3), ("Prototheria", 1),
]
EXCLUDE = ("Soricidae", "Tenrecidae", "Solenodontidae")


def main(cov_file: str, out_file: str, min_cov: float = 0.85) -> None:
    d = pd.read_csv(cov_file, sep="\t")
    d = d[d.in_zoonomia_tree & (d.coverage >= min_cov)]
    d = d[~d.lineage.str.contains("|".join(EXCLUDE))]
    d = d.sort_values("coverage", ascending=False).drop_duplicates("species")
    chosen = ["Homo_sapiens"]                      # the TOGA reference is always present
    for key, quota in QUOTAS:
        grp = d[d.lineage.str.contains(key) & ~d.species.isin(chosen)]
        chosen += grp.species.head(quota).tolist()
    chosen = list(dict.fromkeys(chosen))
    best = d.set_index("species").assembly                 # highest-coverage assembly per species
    with open(out_file, "w") as fh:                        # "species<TAB>assembly" (human = REFERENCE)
        for sp in chosen:
            fh.write(f"{sp}\t{'REFERENCE' if sp == 'Homo_sapiens' else best[sp]}\n")
    sel = d[d.species.isin(chosen)]
    print(f"{len(chosen)} species; median coverage {sel.coverage.median():.3f}")
    for key, _ in QUOTAS:
        print(f"  {key:15s} {sel.lineage.str.contains(key).sum()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(float(a) for a in sys.argv[3:4]))
