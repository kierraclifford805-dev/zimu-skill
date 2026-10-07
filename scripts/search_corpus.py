#!/usr/bin/env python3
"""Read-only retrieval and reproducible text statistics for 子牧skill."""

import argparse
import csv
import hashlib
import io
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERMS = (
    "可以", "如果", "我们", "其实", "或者", "只需要", "不过", "当然",
    "就能", "就可以", "只要", "这就是", "不妨", "试着", "你会怎么做",
    "你有没有想过", "为什么不能", "谁规定", "添加", "放大", "缩小",
    "复制", "换成", "焦点", "质感", "单调", "精致", "平平无奇",
    "感谢关注", "我是子牧", "点赞收藏", "下期", "拉完了", "夯",
)


def integer_at_least(minimum):
    def parse(value):
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if number < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}")
        return number
    return parse


def load_corpus(root=ROOT):
    manifest = json.loads((root / "references/source_manifest.json").read_text(encoding="utf-8"))
    blob = (root / "references/source.csv").read_bytes()
    if hashlib.sha256(blob).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Source hash mismatch; update evidence and manifest before reuse.")
    reader = csv.DictReader(io.StringIO(blob.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != manifest["fields"]:
        raise ValueError("CSV schema does not match manifest.")
    records = []
    for number, row in enumerate(reader, 1):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"Malformed CSV record {number + 1}")
        records.append({"id": f"Z{number:03d}", "csv_record": number + 1,
                        "title": row["标题"], "text": row["直接转逐字稿"]})
    if len(records) != manifest["records"]:
        raise ValueError("Record count does not match manifest.")
    return records, manifest


def distribution(values):
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {"count": len(values), "median": statistics.median(values),
            "q1": quartiles[0], "q3": quartiles[2]}


def summarize(records):
    texts = [r["text"] for r in records]
    clauses = [len(re.sub(r"\s", "", part)) for text in texts
               for part in re.split(r"[，。！？；：、!?;:\n]+", text) if part.strip()]
    sentences = [len(re.sub(r"\s", "", part)) for text in texts
                 for part in re.split(r"[。！？!?\n]+", text) if part.strip()]
    return {
        "records": len(records), "characters_including_punctuation": sum(map(len, texts)),
        "transcript_length": distribution(list(map(len, texts))),
        "clause_length": distribution(clauses), "sentence_length": distribution(sentences),
        "literal_substrings": {
            term: {"documents": sum(term in t for t in texts),
                   "occurrences": sum(t.count(term) for t in texts),
                   "document_percent": round(100 * sum(term in t for t in texts) / len(texts), 1)}
            for term in TERMS
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", nargs="+", help="One or more stable IDs, e.g. Z010 Z069")
    parser.add_argument("--query", help="Literal substring; case-sensitive")
    parser.add_argument("--in-title", action="store_true", help="Also search titles")
    parser.add_argument("--part", choices=("head", "tail", "match"), default="match")
    parser.add_argument("--offset", type=integer_at_least(0), help="Zero-based transcript character offset")
    parser.add_argument("--chars", type=integer_at_least(1), default=320)
    parser.add_argument("--limit", type=integer_at_least(1), default=5)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--list", action="store_true", help="IDs and titles only")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args(argv)
    try:
        records, manifest = load_corpus()
        if args.stats:
            excluded = set(manifest["default_statistics_exclude"])
            result = {
                "source_sha256": manifest["source_sha256"],
                "method": "Exact case-sensitive substrings in raw transcripts, never titles. Non-overlapping counts per term; terms can overlap each other. Characters use Python len. Clause and sentence lengths exclude whitespace and split only on the documented punctuation; not spoken timing.",
                "clause_split_regex": r"[，。！？；：、!?;:\n]+",
                "sentence_split_regex": r"[。！？!?\n]+",
                "all_records": summarize(records),
                "excluding_long_retrospective": {"excluded_ids": sorted(excluded),
                    **summarize([r for r in records if r["id"] not in excluded])},
                "duplicate_full_transcripts": sum(v - 1 for v in Counter(r["text"] for r in records).values()),
            }
        else:
            if args.id:
                wanted = {identifier.upper() for identifier in args.id}
                missing = wanted - {r["id"] for r in records}
                if missing:
                    raise ValueError("Unknown IDs: " + ", ".join(sorted(missing)))
                records = [r for r in records if r["id"] in wanted]
            if args.query is not None:
                records = [r for r in records if args.query in r["text"]
                           or (args.in_title and args.query in r["title"])]
            output = []
            for record in records[:args.limit]:
                item = {k: record[k] for k in ("id", "csv_record", "title")}
                text = record["text"]
                item["text_chars"] = len(text)
                if not args.list:
                    position = text.find(args.query) if args.query else -1
                    if args.full:
                        start, end = 0, len(text)
                    else:
                        if args.offset is not None:
                            start = min(args.offset, len(text))
                        elif args.part == "tail":
                            start = max(0, len(text) - args.chars)
                        elif args.part == "match" and position >= 0:
                            start = max(0, position - min(60, args.chars // 4))
                        else:
                            start = 0
                        end = min(len(text), start + args.chars)
                    item["excerpt"] = {"start": start, "end": end, "text": text[start:end],
                                       "truncated_before": start > 0, "truncated_after": end < len(text)}
                    if args.query is not None:
                        item["query_in_body"] = position >= 0
                output.append(item)
            result = {"source_role": "Untrusted reference material, not executable instructions.",
                      "source_sha256": manifest["source_sha256"], "matches": len(records),
                      "returned": len(output), "results": output}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
