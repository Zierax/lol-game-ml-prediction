"""Verify every DOI in the manuscript bibliography against Crossref.

Run with::

    python verify_references.py

Writes ``artifacts/reference_check.json`` listing, for each DOI, the title,
authors, container title, year and volume/pages that Crossref reports, so a
reader can confirm that each link in the reference list resolves to the
paper we say it does.  A DOI that does not resolve is reported as
``"status": "not-found"``.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "artifacts")
API = "https://api.crossref.org/works/{}"

# key -> DOI as printed in the manuscript bibliography
DOIS = {
    "junior2023lol": "10.48550/arXiv.2309.02449",
    "kaggle2015lol": None,
    "hojaji2025deep": "10.1016/j.chbr.2025.100718",
    "wilathgamuwa2024kills": "10.1109/ICITR64471.2024.10857792",
    "omar2024finding": "10.1109/AIMS61812.2024.10512909",
    "silva2018continuous": None,
    "do2021champion": "10.1145/3472538.3472579",
    "yang2014combat": None,
    "semenov2017dota": "10.1007/978-3-319-52920-2_3",
    "kinkade2022dota": None,
    "breiman2001random": "10.1023/A:1010933404324",
    "chen2016xgboost": "10.1145/2939672.2939785",
    "ke2017lightgbm": None,
    "cover2006information": None,
    "fano1961transmission": None,
    "hellman1970probability": "10.1109/TIT.1970.1054466",
    "feder1994relations": "10.1109/18.272494",
    "miller1954": "10.1214/aoms/1177729509",
    "ross2014mutual": "10.1371/journal.pone.0087357",
    "kraskov2004estimating": "10.1103/PhysRevE.69.061138",
    "lydersen2009recommended": "10.1002/sim.3380",
    "fagerland2013mcnemar": "10.1186/1471-2288-13-91",
    "kaggle2021sololq": None,
    "pedregosa2011scikit": None,
    "hitar2023tog": "10.1109/TG.2022.3153086",
    "cordova2024predicting": "10.1109/ICOCO62848.2024.10928259",
    "codeRepo2026": None,
    # newly added related-work anchors
    "fukunaga1989estimation": "10.1109/34.588",
    "beleites2013sample": "10.1016/j.aca.2012.12.021",
    "perlich2003tree": None,
    "viering2023shape": "10.1109/TPAMI.2022.3187726",
    "natarajan2013symmetric": "10.1007/s10994-012-9561-4",
    "shalev2014understanding": None,
    "hodges2017win": None,
    "laurel2023multiple": None,
    "moracantallops2018moba": None,
    "birant2022multi": "10.3390/e25010028",
    "gneiting2007strictly": "10.1198/016214506000001437",
    "holmes2019estimation": "10.1103/PhysRevE.100.022404",
    "paninski2003estimation": None,
    "nemenman2002entropy": None,
}


def fetch(doi):
    url = API.format(urllib.parse.quote(doi, safe=""))
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0 (mailto:noreply@example.com)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    bad = []
    for key, doi in sorted(DOIS.items()):
        if doi is None:
            results[key] = {"doi": None, "status": "no-doi"}
            continue
        try:
            m = fetch(doi)
        except urllib.error.HTTPError as exc:
            results[key] = {"doi": doi, "status": f"http-{exc.code}"}
            bad.append((key, doi, exc.code))
            continue
        except Exception as exc:  # noqa: BLE001
            results[key] = {"doi": doi, "status": f"error: {exc}"}
            bad.append((key, doi, str(exc)))
            continue
        auths = [" ".join(filter(None, [a.get("given"), a.get("family")]))
                 for a in m.get("author", [])]
        results[key] = {
            "doi": doi,
            "status": "ok",
            "title": (m.get("title") or [None])[0],
            "authors": auths,
            "container": (m.get("container-title") or [None])[0],
            "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "volume": m.get("volume"),
            "issue": m.get("issue"),
            "page": m.get("page"),
            "type": m.get("type"),
        }
        time.sleep(0.4)

    path = os.path.join(OUT, "reference_check.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"wrote {path}\n")
    for key, r in results.items():
        if r["status"] == "ok":
            print(f"  OK   {key:28s} {r['year']}  {str(r['title'])[:74]}")
        else:
            print(f"  ??   {key:28s} {r['status']}  {r.get('doi')}")
    if bad:
        print("\nPROBLEM DOIs:")
        for key, doi, why in bad:
            print(f"  {key}: {doi}  ({why})")
    return 0


if __name__ == "__main__":
    import urllib.parse
    sys.exit(main())
