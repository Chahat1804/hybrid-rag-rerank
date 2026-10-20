import uuid
import tiktoken
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    section: str


# -------------------------------------------------------------
# 1. Text Chunking
# -------------------------------------------------------------
def chunk_text(
    text: str,
    source: str,
    section: str,
    target_tokens: int = 400,
    overlap_tokens: int = 50,
    tokenizer_name: str = "cl100k_base"
) -> List[DocumentChunk]:
    """Splits text into chunks of 300-500 tokens with 50-token overlap."""
    enc = tiktoken.get_encoding(tokenizer_name)
    tokens = enc.encode(text)

    chunks = []
    step = target_tokens - overlap_tokens

    for i in range(0, len(tokens), step):
        chunk_slice = tokens[i : i + target_tokens]
        chunk_text = enc.decode(chunk_slice).strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4())[:8],
                    text=chunk_text,
                    source=source,
                    section=section,
                )
            )
    return chunks


# -------------------------------------------------------------
# 2, 3, 4, 5. Hybrid Retriever & Reranker Engine
# -------------------------------------------------------------
class HybridRAG:
    def __init__(
        self,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print("Initializing Embeddings & Cross-Encoder models...")
        self.embed_model = SentenceTransformer(embed_model_name)
        self.reranker = CrossEncoder(rerank_model_name)

        # In-memory ChromaDB vector index
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="hybrid_rag_corpus")

        # BM25 Keyword index
        self.bm25 = None
        self.bm25_corpus: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []

    def index_documents(self, chunks: List[DocumentChunk]):
        """Populate both Vector (Chroma) and BM25 indices."""
        self.bm25_corpus = chunks
        self.tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [{"source": c.source, "section": c.section} for c in chunks]
        embeddings = self.embed_model.encode(texts, show_progress_bar=False).tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"Indexed {len(chunks)} chunks into Vector and BM25 indices.")

    # Step 2: Vector Only
    def vector_search(self, query: str, top_k: int = 20) -> List[DocumentChunk]:
        query_embedding = self.embed_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append(
                DocumentChunk(
                    chunk_id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    source=results["metadatas"][0][i]["source"],
                    section=results["metadatas"][0][i]["section"],
                )
            )
        return retrieved

    # Step 3: BM25 Only
    def bm25_search(self, query: str, top_k: int = 20) -> List[DocumentChunk]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.bm25_corpus[i] for i in top_indices]

    # Step 4: Reciprocal Rank Fusion (RRF)
    def rrf_fusion(
        self,
        vector_results: List[DocumentChunk],
        bm25_results: List[DocumentChunk],
        k: int = 60,
        top_k: int = 20,
    ) -> List[DocumentChunk]:
        """Merges two ranked lists using RRF score: 1 / (k + rank)."""
        chunk_map = {}
        rrf_scores = {}

        for rank, chunk in enumerate(vector_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + rank))

        for rank, chunk in enumerate(bm25_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + rank))

        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        return [chunk_map[cid] for cid in sorted_chunk_ids[:top_k]]

    # Step 5: Cross-Encoder Reranking
    def rerank(self, query: str, candidates: List[DocumentChunk], top_k: int = 5) -> List[DocumentChunk]:
        if not candidates:
            return []

        pairs = [[query, c.text] for c in candidates]
        cross_scores = self.reranker.predict(pairs)

        ranked_indices = sorted(range(len(cross_scores)), key=lambda i: cross_scores[i], reverse=True)
        return [candidates[i] for i in ranked_indices[:top_k]]

    def retrieve(self, query: str, mode: str = "hybrid_rerank") -> List[DocumentChunk]:
        """Unified retrieval interface for benchmarking and generation."""
        if mode == "vector":
            return self.vector_search(query, top_k=5)

        elif mode == "hybrid":
            vec_hits = self.vector_search(query, top_k=20)
            bm25_hits = self.bm25_search(query, top_k=20)
            return self.rrf_fusion(vec_hits, bm25_hits, top_k=5)

        elif mode == "hybrid_rerank":
            vec_hits = self.vector_search(query, top_k=20)
            bm25_hits = self.bm25_search(query, top_k=20)
            fused = self.rrf_fusion(vec_hits, bm25_hits, top_k=20)
            return self.rerank(query, fused, top_k=5)

        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
