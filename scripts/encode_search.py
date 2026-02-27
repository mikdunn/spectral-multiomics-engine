from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", default="ATAC-seq")
    ap.add_argument("--biosample", default="GM12878")
    ap.add_argument("--assembly", default="hg19")
    ap.add_argument("--file_format", default="bedgraph")
    ap.add_argument("--limit", default=25, type=int)
    args = ap.parse_args()

    params = {
        "type": "File",
        "assay_title": args.assay,
        "biosample_ontology.term_name": args.biosample,
        "assembly": args.assembly,
        "file_format": args.file_format,
        "limit": str(args.limit),
        "format": "json",
    }
    url = "https://www.encodeproject.org/search/?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    with urllib.request.urlopen(req, timeout=60) as r:
        obj = json.load(r)

    g = obj.get("@graph", [])
    print("URL:", url)
    print("Returned:", len(g))

    for f in g:
        print(
            f.get("accession"),
            "|",
            f.get("output_type"),
            "|",
            f.get("file_type"),
            "|",
            f.get("status"),
            "|",
            f.get("href"),
        )


if __name__ == "__main__":
    main()
