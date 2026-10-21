from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import TokenChunker
from .loaders import load_corpus
from .models import Chunk, SearchResult

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


def _lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def _load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("Embedding support requires sentence-transformers. Install requirements.txt.") from exc
    return SentenceTransformer(model_name)


def build_index(
    corpus_dir: str | Path,
    index_dir: str | Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 350,
    overlap: int = 50,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Load, chunk, embed, and persist a corpus as a FAISS + metadata index."""
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("Vector indexing requires faiss-cpu. Install requirements.txt.") from exc

    segments = load_corpus(corpus_dir)
    if not segments:
        raise ValueError(
            f"No readable documents found in {corpus_dir}. Supported formats: PDF, DOCX, TXT, and Markdown."
        )

    model = _load_embedder(embedding_model)
    max_sequence_length = getattr(model, "max_seq_length", None)
    if max_sequence_length and chunk_size + 2 > int(max_sequence_length):
        raise ValueError(
            f"Chunk size {chunk_size} plus special tokens exceeds {embedding_model}'s "
            f"maximum sequence length ({max_sequence_length}). Use a smaller chunk size or a model "
            "with a longer context window."
        )

    chunker = TokenChunker(embedding_model, chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.split(segments)
    if not chunks:
        raise ValueError("The documents contained no extractable text. Scanned PDFs may need OCR first.")

    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(chunks):
        raise RuntimeError("The embedding model returned an unexpected array shape.")

    vector_index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    vector_index.add(embeddings)

    output = Path(index_dir)
    output.mkdir(parents=True, exist_ok=True)
    faiss.write_index(vector_index, str(output / "index.faiss"))
    with (output / "chunks.jsonl").open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    manifest = {
        "version": 1,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunk_count": len(chunks),
        "source_count": len({chunk.source for chunk in chunks}),
        "sources": sorted({chunk.source for chunk in chunks}),
        "vector_index": "IndexFlatIP (cosine via normalized vectors)",
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@dataclass(frozen=True)
class _Filter:
    sources: set[str] | None = None
    modified_after: date | None = None
    modified_before: date | None = None


def _as_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _parse_filter(filters: dict[str, Any] | None) -> _Filter:
    filters = filters or {}
    sources = filters.get("sources")
    return _Filter(
        sources=set(sources) if sources else None,
        modified_after=_as_date(filters.get("modified_after")),
        modified_before=_as_date(filters.get("modified_before")),
    )


class RAGEngine:
    """Loaded vector/BM25 index with optional cross-encoder reranking."""

    def __init__(self, index_dir: str | Path, reranker_model: str = DEFAULT_RERANKER_MODEL):
        try:
            import faiss
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError("Install project dependencies with: pip install -r requirements.txt") from exc

        self.index_dir = Path(index_dir)
        manifest_path = self.index_dir / "manifest.json"
        chunks_path = self.index_dir / "chunks.jsonl"
        vector_path = self.index_dir / "index.faiss"
        if not all(path.exists() for path in (manifest_path, chunks_path, vector_path)):
            raise FileNotFoundError(f"No complete index found in {self.index_dir}. Build the index first.")

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.embedding_model_name = self.manifest["embedding_model"]
        self.reranker_model_name = reranker_model
        self.chunks = [
            Chunk.from_dict(json.loads(line))
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.vector_index = faiss.read_index(str(vector_path))
        if self.vector_index.ntotal != len(self.chunks):
            raise ValueError("FAISS index and chunks.jsonl are out of sync. Rebuild the index.")

        self._embedder = None
        self._cross_encoder = None
        tokenized_corpus = [_lexical_tokens(chunk.text) for chunk in self.chunks]
        if not tokenized_corpus or not any(tokenized_corpus):
            raise ValueError("The index contains no lexical tokens. Rebuild it with readable text.")
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.sources = sorted({chunk.source for chunk in self.chunks})
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = _load_embedder(self.embedding_model_name)
        return self._embedder

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise RuntimeError("Reranking requires sentence-transformers. Install requirements.txt.") from exc
            self._cross_encoder = CrossEncoder(self.reranker_model_name)
        return self._cross_encoder

    def warmup(self, include_reranker: bool = False) -> None:
        """Load model weights and run a tiny inference outside latency timings."""
        self.embedder.encode(["warmup"], convert_to_numpy=True, normalize_embeddings=True)
        if include_reranker:
            self.cross_encoder.predict([("warmup", "warmup")], show_progress_bar=False)

    @staticmethod
    def _passes_filter(chunk: Chunk, filters: _Filter) -> bool:
        if filters.sources is not None and chunk.source not in filters.sources:
            return False
        if filters.modified_after is not None or filters.modified_before is not None:
            chunk_date = _as_date(chunk.modified_at)
            if chunk_date is None:
                return False
            if filters.modified_after is not None and chunk_date < filters.modified_after:
                return False
            if filters.modified_before is not None and chunk_date > filters.modified_before:
                return False
        return True

    def _query_vector(self, query: str) -> np.ndarray:
        vector = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)

    def retrieve_vector(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query.strip() or top_k <= 0 or not self.chunks:
            return []
        parsed_filter = _parse_filter(filters)
        has_filters = any((parsed_filter.sources, parsed_filter.modified_after, parsed_filter.modified_before))
        search_k = self.vector_index.ntotal if has_filters else min(top_k, self.vector_index.ntotal)
        query_vector = self._query_vector(query)
        scores, indices = self.vector_index.search(query_vector, search_k)

        results: list[SearchResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            if self._passes_filter(chunk, parsed_filter):
                results.append(SearchResult(chunk=chunk, score=float(score), method="vector"))
                if len(results) >= top_k:
                    break
        return results

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query.strip() or top_k <= 0 or not self.chunks:
            return []
        parsed_filter = _parse_filter(filters)
        query_tokens = _lexical_tokens(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1]
        results: list[SearchResult] = []
        for index in ranked_indices:
            score = float(scores[int(index)])
            if score <= 0:
                continue
            chunk = self.chunks[int(index)]
            if self._passes_filter(chunk, parsed_filter):
                results.append(SearchResult(chunk=chunk, score=score, method="bm25"))
                if len(results) >= top_k:
                    break
        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 20,
        candidate_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Fuse top vector and BM25 lists using reciprocal rank fusion."""
        if top_k <= 0 or candidate_k <= 0:
            return []
        vector_results = self.retrieve_vector(query, top_k=candidate_k, filters=filters)
        bm25_results = self.retrieve_bm25(query, top_k=candidate_k, filters=filters)
        fused_scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}

        for ranked_list in (vector_results, bm25_results):
            for rank, result in enumerate(ranked_list, start=1):
                chunk_id = result.chunk.chunk_id
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
                chunks[chunk_id] = result.chunk

        ordered_ids = sorted(fused_scores, key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id))
        return [
            SearchResult(chunk=chunks[chunk_id], score=fused_scores[chunk_id], method="hybrid")
            for chunk_id in ordered_ids[:top_k]
        ]

    def retrieve_reranked(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        candidates = self.retrieve_hybrid(
            query,
            top_k=candidate_k,
            candidate_k=candidate_k,
            filters=filters,
        )
        if not candidates:
            return []
        pairs = [(query, result.chunk.text) for result in candidates]
        scores = np.asarray(self.cross_encoder.predict(pairs, show_progress_bar=False)).reshape(-1)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: (-float(pair[1]), pair[0].chunk.chunk_id),
        )
        return [
            SearchResult(chunk=result.chunk, score=float(score), method="hybrid_rerank")
            for result, score in ranked[:top_k]
        ]

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        top_k: int = 5,
        candidate_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if mode == "vector":
            return self.retrieve_vector(query, top_k=top_k, filters=filters)
        if mode == "hybrid":
            return self.retrieve_hybrid(query, top_k=top_k, candidate_k=candidate_k, filters=filters)
        if mode in {"hybrid_rerank", "rerank"}:
            return self.retrieve_reranked(query, top_k=top_k, candidate_k=candidate_k, filters=filters)
        raise ValueError(f"Unknown retrieval mode: {mode}")

    def catalog(self) -> list[dict[str, Any]]:
        """Return chunk metadata for benchmark labeling / traceability."""
        return [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "section": chunk.section,
                "page": chunk.page,
                "modified_at": chunk.modified_at,
                "text_preview": chunk.text[:500].replace("\n", " "),
            }
            for chunk in self.chunks
        ]
