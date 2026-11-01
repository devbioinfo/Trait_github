"""Check each manuscript reference against Crossref (title, first author, journal year, volume, first page).

Input : manuscript_refs/references.txt (one numbered reference per line, as in the manuscript).
Output: results/review/reference_check.csv and a summary on screen. Uses only the Python standard library.
Run   : python3 scripts/check_references.py      (needs internet; ~1 request per second, about 1 minute)
"""
import csv, difflib, json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "manuscript_refs" / "references.txt"
OUT = ROOT / "results" / "review" / "reference_check.csv"
DOIS = ROOT / "manuscript_refs" / "dois.tsv"   # optional: reference number -> DOI (direct lookup)
doi_of = {}
if DOIS.exists():
    for l in DOIS.read_text().splitlines():
        if l.strip() and not l.startswith("#"):
            k, v = l.split("\t")[:2]
            doi_of[k.strip()] = v.strip()


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse(line):
    n, rest = line.split(". ", 1)
    rest = rest.strip()
    m = re.search(r"\*([^*]+)\*\s*(?:\*\*([^*]+)\*\*)?,?\s*([^ (]*)\s*\((\d{4})\)", rest)
    journal, vol, page, year = (m.group(1), m.group(2) or "", m.group(3).rstrip(","), m.group(4)) if m else ("", "", "", "")
    pre = rest[: m.start()] if m else rest
    parts = re.split(r"(?:et al\.|[A-Z]\.|Consortium\.)\s+(?=[A-Z\"“])", pre)
    title = parts[-1].strip().rstrip(".")
    first = re.match(r"([^,.]+)", rest).group(1).strip()
    doi = re.search(r"doi:(\S+)", rest)
    return dict(n=n, title=title, first=first, journal=journal, vol=vol, page=page, year=year,
                doi=doi.group(1).rstrip(".") if doi else "")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "reference-check/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def lookup(r):
    if r["doi"]:
        return get("https://api.crossref.org/works/" + urllib.parse.quote(r["doi"]))["message"]
    q = urllib.parse.urlencode({"query.bibliographic": f'{r["title"]} {r["journal"]} {r["year"]}',
                                "query.author": r["first"], "rows": 1})
    items = get("https://api.crossref.org/works?" + q)["message"]["items"]
    return items[0] if items else None


def first_page(p):
    return re.split(r"[-–]", p or "")[0].lstrip("e")


rows = []
for line in REFS.read_text().splitlines():
    if not line.strip():
        continue
    r = parse(line)
    if r["n"] in doi_of:
        r["doi"] = doi_of[r["n"]]
    try:
        it = lookup(r)
    except Exception as e:  # network or HTTP error
        rows.append({**r, "status": "LOOKUP FAILED", "issues": str(e)[:80], "crossref_doi": ""})
        continue
    if not it:
        rows.append({**r, "status": "NOT FOUND", "issues": "", "crossref_doi": ""})
        continue
    ct = (it.get("title") or [""])[0]
    sim = difflib.SequenceMatcher(None, norm(r["title"]), norm(ct)).ratio()
    y = str((it.get("issued", {}).get("date-parts") or [[""]])[0][0])
    v = it.get("volume", "")
    p = it.get("page", "") or it.get("article-number", "")
    fam = [a.get("family", a.get("name", "")) for a in it.get("author", [])][:1]
    issues = []
    if sim < 0.9:
        issues.append(f'title differs ({sim:.2f}): "{ct}"')
    yp = str((it.get("published-print", {}).get("date-parts") or [[""]])[0][0])
    if r["year"] and y and r["year"] not in (y, yp):
        issues.append(f"year {y}")
    if r["vol"] and v and v != r["vol"]:
        issues.append(f"volume {v}")
    if r["page"] and p and first_page(r["page"]) != first_page(p):
        issues.append(f"page {p}")
    if fam and fam[0] and norm(fam[0]) not in norm(r["first"]) and "consortium" not in r["first"].lower():
        issues.append(f"first author {fam[0]}")
    rows.append({**r, "status": "OK" if not issues else "CHECK", "issues": "; ".join(issues), "crossref_doi": it.get("DOI", "")})
    time.sleep(1.0)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["n", "status", "issues", "crossref_doi", "first", "title", "journal", "vol", "page", "year", "doi"])
    w.writeheader()
    w.writerows(rows)
ok = sum(r["status"] == "OK" for r in rows)
print(f"{ok} of {len(rows)} references match Crossref; details in {OUT.relative_to(ROOT)}")
for r in rows:
    if r["status"] != "OK":
        print(f'[{r["n"]}] {r["status"]}: {r["issues"]}')
