"""Pull a curated subset of ai4bharat/MSMARCO-XI for offline indexing.

The full dataset is ~55GB across 10M+ (query, language) rows. For a demo-scale
RAG system we only need a few thousand unique queries with their passage sets
and relevance labels. We stream the validation split (has ground-truth
`is_selected` labels for retrieval eval) and keep rows where
target_lang == "hin_Deva", which both deduplicates to one row per unique
query_id and gives us a real Hindi translation of the query/answer for the
voice-input demo, at no extra cost.
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

TARGET_LANG = "hin_Deva"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parent.parent / "data" / "raw" / "msmarco_xi.jsonl")
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ai4bharat/MSMARCO-XI", split=args.split, streaming=True)

    seen_query_ids = set()
    kept = 0
    scanned = 0

    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            scanned += 1
            if row["target_lang"] != TARGET_LANG:
                continue
            qid = row["query_id"]
            if qid in seen_query_ids:
                continue
            seen_query_ids.add(qid)

            passages = row["passages"]
            english_passages = passages.get("English_passages") or []
            is_selected = passages.get("is_selected") or []
            if not english_passages:
                continue

            record = {
                "query_id": qid,
                "query_type": row.get("query_type", "UNKNOWN"),
                "eng_query": row.get("Eng_Query", "").strip(),
                "eng_answer": row.get("Eng_Answer", "").strip(),
                "hin_query": row.get("query", "").strip(),
                "hin_answer": row.get("Answer", "").strip(),
                "passages": english_passages,
                "is_selected": is_selected,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

            if kept % 500 == 0:
                print(f"scanned={scanned} kept={kept}", file=sys.stderr)

            if kept >= args.limit:
                break

    print(f"Done. scanned={scanned} kept={kept} -> {out_path}")


if __name__ == "__main__":
    main()
