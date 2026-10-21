from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .index import RAGEngine

EVALUATION_CANDIDATES = 20
EVALUATION_HIT_K = 5
METHODS = ("vector", "hybrid", "hybrid_rerank")


def _split_gold_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]
        except json.JSONDecodeError:
            pass
    # A question may have more than one acceptable source chunk.
    delimiter = "|" if "|" in raw else (";" if ";" in raw else None)
    values = raw.split(delimiter) if delimiter else [raw]
    return [value.strip() for value in values if value.strip()]


def normalize_benchmark_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row_number, record in enumerate(records, start=1):
        question = str(record.get("question", "")).strip()
        gold = record.get("gold_chunk_ids") or record.get("gold_chunk_id")
        gold_ids = _split_gold_ids(gold)
        if not question or not gold_ids:
            raise ValueError(
                f"Benchmark row {row_number} needs a non-empty question and gold_chunk_id(s)."
            )
        normalized.append(
            {
                "question_id": str(record.get("question_id", row_number)),
                "question": question,
                "gold_chunk_ids": gold_ids,
            }
        )
    if not normalized:
        raise ValueError("The benchmark contains no questions.")
    return normalized


def parse_benchmark_text(text: str, filename: str = "benchmark.csv") -> list[dict[str, Any]]:
    """Parse benchmark CSV or JSONL into normalized evaluation records."""
    suffix = Path(filename).suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise ValueError("CSV must have a header row.")
        records = list(reader)
    return normalize_benchmark_records(records)


def load_benchmark(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    return parse_benchmark_text(path.read_text(encoding="utf-8-sig"), path.name)


def _rank_of_gold(results, gold_ids: set[str]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if result.chunk.chunk_id in gold_ids:
            return rank
    return None


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def evaluate_records(
    engine: RAGEngine,
    records: Sequence[dict[str, Any]],
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare vector-only, RRF hybrid, and hybrid + reranker retrieval.

    Model loading/warm-up and LLM generation are excluded from query latency.
    Each measured call includes query embedding and the retrieval/reranking work
    for that setup. MRR is computed over each setup's ranked top-20 list.
    """
    benchmark = normalize_benchmark_records(records)
    known_ids = {chunk.chunk_id for chunk in engine.chunks}
    missing = sorted(
        {chunk_id for row in benchmark for chunk_id in row["gold_chunk_ids"] if chunk_id not in known_ids}
    )
    if missing:
        excerpt = ", ".join(missing[:5])
        more = " …" if len(missing) > 5 else ""
        raise ValueError(
            f"Benchmark references chunk IDs not in this index: {excerpt}{more}. "
            "Re-export the chunk catalog after reindexing and update the benchmark."
        )

    # Keep first-time model downloads and initialization out of per-query timing.
    engine.warmup(include_reranker=True)
    details: list[dict[str, Any]] = []
    ranks_by_method: dict[str, list[int | None]] = {method: [] for method in METHODS}
    latencies_by_method: dict[str, list[float]] = {method: [] for method in METHODS}

    for position, row in enumerate(benchmark, start=1):
        question = row["question"]
        gold_ids = set(row["gold_chunk_ids"])
        detail: dict[str, Any] = {
            "question_id": row["question_id"],
            "question": question,
            "gold_chunk_ids": "|".join(row["gold_chunk_ids"]),
        }

        searchers = {
            "vector": lambda: engine.retrieve_vector(question, top_k=EVALUATION_CANDIDATES),
            "hybrid": lambda: engine.retrieve_hybrid(
                question,
                top_k=EVALUATION_CANDIDATES,
                candidate_k=EVALUATION_CANDIDATES,
            ),
            "hybrid_rerank": lambda: engine.retrieve_reranked(
                question,
                top_k=EVALUATION_CANDIDATES,
                candidate_k=EVALUATION_CANDIDATES,
            ),
        }
        for method in METHODS:
            started = time.perf_counter()
            results = searchers[method]()
            latency_ms = (time.perf_counter() - started) * 1000.0
            rank = _rank_of_gold(results, gold_ids)
            ranks_by_method[method].append(rank)
            latencies_by_method[method].append(latency_ms)
            detail[f"{method}_rank"] = rank if rank is not None else ""
            detail[f"{method}_hit_at_5"] = int(rank is not None and rank <= EVALUATION_HIT_K)
            detail[f"{method}_latency_ms"] = round(latency_ms, 3)

        details.append(detail)
        if progress_callback:
            progress_callback(position, len(benchmark))

    summaries: list[dict[str, Any]] = []
    display_names = {
        "vector": "Vector-only",
        "hybrid": "Hybrid (BM25 + vector, RRF)",
        "hybrid_rerank": "Hybrid + cross-encoder rerank",
    }
    for method in METHODS:
        ranks = ranks_by_method[method]
        latencies = latencies_by_method[method]
        hit_rate = sum(rank is not None and rank <= EVALUATION_HIT_K for rank in ranks) / len(ranks)
        reciprocal_ranks = [1.0 / rank if rank is not None else 0.0 for rank in ranks]
        summaries.append(
            {
                "setup": display_names[method],
                "n_questions": len(benchmark),
                "hit_rate_at_5": hit_rate,
                "mrr_at_20": statistics.fmean(reciprocal_ranks),
                "mean_latency_ms": statistics.fmean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": _percentile(latencies, 95),
            }
        )
    return summaries, details


def _print_table(summaries: Sequence[dict[str, Any]]) -> None:
    print("| Setup | Questions | Hit-rate@5 | MRR@20 | Mean latency (ms) | Median (ms) | P95 (ms) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        print(
            f"| {row['setup']} | {row['n_questions']} | "
            f"{row['hit_rate_at_5']:.1%} | {row['mrr_at_20']:.3f} | "
            f"{row['mean_latency_ms']:.1f} | {row['median_latency_ms']:.1f} | "
            f"{row['p95_latency_ms']:.1f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vector, hybrid, and reranked RAG retrieval.")
    parser.add_argument("--index-dir", default="data/index", help="Built index folder")
    parser.add_argument("--benchmark", default="data/evaluation/benchmark.csv", help="CSV or JSONL benchmark")
    parser.add_argument("--output", default="reports/evaluation.json", help="JSON report output path")
    args = parser.parse_args()

    records = load_benchmark(args.benchmark)
    engine = RAGEngine(args.index_dir)
    summaries, details = evaluate_records(engine, records)
    report = {
        "metrics": summaries,
        "details": details,
        "notes": {
            "hit_rate": "Hit-rate@5: fraction of questions with any gold chunk in the top 5.",
            "mrr": "MRR@20: reciprocal rank of the first gold chunk, averaged over questions; missing in top 20 scores 0.",
            "latency": "Steady-state retrieval latency; model loading/warm-up and LLM generation excluded.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_table(summaries)
    print(f"\nDetailed report written to {output}")


if __name__ == "__main__":
    main()
