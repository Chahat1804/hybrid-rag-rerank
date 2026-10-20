# Hybrid RAG with Cross-Encoder Reranking

> Production-grade question-answering engine over technical documentation using Hybrid Retrieval (BM25 + Dense Vectors), Reciprocal Rank Fusion (RRF), and Cross-Encoder Reranking.

![Demo](assets/demo.gif)

---

## Architecture

```text
User Query
   │
   ├──► [ChromaDB] Vector Search (all-MiniLM-L6-v2) ──► Top 20 Candidates
   │                                                        │
   └──► [Rank-BM25] Keyword Match (Tokenized)       ──► Top 20 Candidates
                                                            │
                                                            ▼
                                           [Reciprocal Rank Fusion (RRF)]
                                                Score = Σ 1 / (60 + Rank)
                                                            │
                                                            ▼
                                                   Top 20 Unified Chunks
                                                            │
                                                            ▼
                                              [Cross-Encoder Reranker]
                                              (ms-marco-MiniLM-L-6-v2)
                                                            │
                                                            ▼
                                                   Top 5 Ranked Chunks
                                                            │
                                                            ▼
                                                [GPT-4o-mini Generator]
                                           (Claim-level citations + fallback)
```

---

## Benchmark Results (50-Question Test Suite)

Evaluated against a 150-chunk technical document corpus. Baseline is standard dense vector retrieval.

| Pipeline Mode | Hit-Rate@5 | MRR (Mean Reciprocal Rank) | Avg Retrieval Latency |
| :--- | :--- | :--- | :--- |
| **Vector Search Only (Baseline)** | 66.0% | 0.481 | **12.4 ms** |
| **Hybrid (Dense + BM25 via RRF)** | 82.0% | 0.624 | 18.7 ms |
| **Hybrid + Cross-Encoder Rerank** | **94.0%** | **0.812** | 52.1 ms |

**Takeaway:** Combining lexical search with semantic search recovered technical acronyms that pure embeddings missed. Cross-encoder reranking boosted top-1 relevance (reflected in the MRR jump from 0.481 to 0.812) at the cost of ~33ms added latency.

---

## Quickstart (3 Commands)

### 1. Clone & install dependencies
```bash
git clone https://github.com/Chahat1804/hybrid-rag-rerank.git && cd hybrid-rag-rerank
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
cp .env.example .env
# Set your OPENAI_API_KEY in .env
```

### 3. Launch Demo or Run Benchmark
```bash
# Launch interactive Streamlit app
streamlit run app.py

# Or evaluate the benchmark suite in your terminal
python evaluate.py
```

### Tech Stack
- **Vector DB / Embeddings:** ChromaDB, `sentence-transformers/all-MiniLM-L6-v2`
- **Sparse Retrieval:** `rank-bm25` (Okapi BM25)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Orchestration & LLM:** Python 3.10+, OpenAI API (`gpt-4o-mini`)
- **Evaluation & UI:** NumPy, Tabulate, Streamlit

---

## What I Would Improve Next (Engineering Judgment)

1. **Context-Aware Query Rewriting:** Complex, multi-part, or conversational follow-up questions degrade both BM25 and vector search. Introducing a fast, small-model query rewriting step (e.g., resolving coreferences using an SLM) prior to retrieval would improve hit rate on vague prompts.
2. **Dynamic Metadata Filtering:** In a multi-tenant enterprise system, global hybrid retrieval risks data leakage and context contamination. Adding self-querying structured metadata extraction (`department == "engineering"`, `effective_date >= 2024`) before running the vector and BM25 passes would eliminate irrelevant candidates earlier in the pipe.
3. **Async Reranking Pipeline:** Reranking 20 chunks with a cross-encoder accounts for >60% of total retrieval latency (~33ms). Moving reranking to an asynchronous worker queue or quantizing the cross-encoder to ONNX runtime would drop latency below 25ms.
