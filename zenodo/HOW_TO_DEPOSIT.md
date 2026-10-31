# Depositing code and data on Zenodo

Two records, each with its own DOI:

| Record | What | Licence | How |
|---|---|---|---|
| Data | `zenodo/convergence_mammals_data_v1.tar.gz` (45 MB, built by `bash scripts/make_zenodo_data.sh`; description in `zenodo/README_data.md`) | CC BY 4.0 | upload by hand |
| Code | this GitHub repository at release v1.0.0 (metadata in `.zenodo.json`, `CITATION.cff`, `LICENSE`) | MIT | GitHub-Zenodo integration |

## Before starting
Author names, affiliations and ORCID iDs are filled in `.zenodo.json`, `CITATION.cff` and `LICENSE`
(Konhar, Afzoon, Biswal); check them once before publishing.

## 1. Data record (do this first)
1. Log in at https://zenodo.org (ORCID or GitHub login).
2. **New upload** -> drag in `convergence_mammals_data_v1.tar.gz`.
3. "Do you already have a DOI?" -> **No** -> **Get a DOI now!** (reserves the DOI; it can
   go into the manuscript before anything is published).
4. Resource type **Dataset**; title "Data for: Convergent evolution in mammals is written in
   gene loss"; creators; description (first two paragraphs of `README_data.md`); licence
   **Creative Commons Attribution 4.0**; keywords as in `.zenodo.json`.
5. **Save draft.** Publish when the manuscript is submitted. To keep the files private
   until acceptance, choose access **Restricted** or an **embargo** date; the DOI and the
   description are public either way.

## 2. Code record
1. The GitHub repository must be **public** (Zenodo archives only public repositories).
2. Zenodo -> your name -> **GitHub** -> switch on `devbioinfo/Trait_github`.
3. On GitHub: **Releases -> Draft a new release**, tag `v1.0.0`, title "v1.0.0 (manuscript
   submission)", **Publish release**. Zenodo archives the release and mints a DOI within a
   few minutes; it appears under Zenodo -> GitHub.
4. Add the data DOI to `.zenodo.json` (`isSupplementedBy`) before tagging, or edit the
   Zenodo record afterwards.

## 3. After both DOIs exist
Put them into the manuscript's Data availability and Code availability statements,
`CITATION.cff` (`doi:`) and the README.
