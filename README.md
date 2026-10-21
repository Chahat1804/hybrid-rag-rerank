# Hybrid RAG with Reranking

A runnable document-QA project that compares **vector-only retrieval** with **BM25 + vector retrieval using Reciprocal Rank Fusion (RRF)** and **hybrid retrieval + cross-encoder reranking**. The Streamlit app answers from retrieved passages, displays the cited evidence, and runs a gold-chunk benchmark.

> No performance numbers are pre-filled: meaningful hit-rate/MRR results require your real corpus and a manually checked benchmark. The evaluator produces the comparison table once those are supplied.

## Pipeline

1. Load PDF, DOCX, TXT, and Markdown files; preserve source, section, page (where available), and file-modified date.
2. Chunk with the selected embedding tokenizer. Defaults are **350 tokens** with **50-token overlap**.
3. Create normalized sentence-transformer embeddings and a cosine-similarity FAISS index. Build a BM25 index over the same chunks.
4. For hybrid retrieval, take the top 20 from each index and fuse by `sum(1 / (60 + rank))`.
5. For reranking, score the fused top 20 with a cross-encoder and return the best 5.
6. Optionally send those passages to an OpenAI-compatible chat API. The prompt requires a citation for each factual claim and an “I don't know” response when the evidence is insufficient.
7. Evaluate vector-only, hybrid, and hybrid + reranking against a labeled benchmark.

Default models:

- Embeddings: `sentence-transformers/all-mpnet-base-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Answer model: `gpt-4o-mini` (configurable; any compatible chat-completions endpoint can be used)

## Quick start

Use Python 3.10 or newer. Model weights are downloaded on first use, so allow time and disk space for the initial setup.

```bash
cd hybrid-rag
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

In the app:

1. Upload your 50–200 page corpus in the sidebar, or place supported files in `data/corpus/`.
2. Click **Save uploads to corpus** if you uploaded files, then **Build / replace index**.
3. Ask questions in **Ask documents**. Choose the vector-only baseline, hybrid RRF, or hybrid + reranking. Restrict retrieval to particular documents using the source filter.
4. Add an API key in the sidebar if you want generated answers. The key is held only in the current Streamlit session and is not written to disk. The app can also use an OpenAI-compatible API base URL.

You can build the index from the terminal instead:

```bash
python -m hybrid_rag.build \
  --corpus-dir data/corpus \
  --index-dir data/index \
  --chunk-size 350 \
  --overlap 50
```

### Supported input notes

- PDF text is extracted with `pypdf`; scanned/image-only PDFs need OCR before indexing.
- DOCX headings and Markdown/plain-text headings are retained as section metadata. Page numbers are available for PDFs; DOCX/TXT/Markdown have no page metadata.
- The app stores your local corpus, FAISS index, and reports under `data/` and `reports/`. These are ignored by Git by default.

## Build the 50-question benchmark

A hit-rate or MRR number is only useful if the gold chunk is genuinely relevant. Create questions from the corpus, then label the chunk that contains the supporting evidence.

1. In **Evaluate retrieval**, click **Download chunk catalog for labeling gold answers**.
2. For each question, use the catalog's `chunk_id` as its `gold_chunk_id`. Read the preview (and the source document) to verify it really supports the answer.
3. Prepare 50 rows in CSV or JSONL. CSV example:

```csv
question,gold_chunk_id
What does the policy say about appeals?,8b6de73c5f61a501
```

CSV supports either `gold_chunk_id` or `gold_chunk_ids`; use `|` between multiple acceptable chunk IDs, for example `id1|id2`. JSONL accepts the same field names, with `gold_chunk_ids` also allowed as a JSON array.

4. Upload the benchmark in **Evaluate retrieval** and click **Run comparison**. The report includes a summary table and per-question ranks/latencies; download the JSON report for your records. The same evaluation is available from the CLI:

```bash
python -m hybrid_rag.evaluate \
  --index-dir data/index \
  --benchmark data/evaluation/benchmark.csv \
  --output reports/evaluation.json
```

`data/evaluation/benchmark.template.csv` is a header-only template. Chunk IDs are tied to the indexed source text and chunking configuration. If you modify documents or reindex with different settings, re-export the catalog and re-check the labels.

## Metrics and interpretation

| Metric | Definition |
|---|---|
| Hit-rate@5 | Fraction of questions where any gold chunk appears in the first 5 results. |
| MRR@20 | Mean reciprocal rank of the first gold chunk within each method's ranked top 20; a miss scores 0. |
| Latency | Mean, median, and p95 steady-state retrieval time per query. Includes query embedding and, for reranking, cross-encoder inference. Excludes model loading/warm-up and LLM answer generation. |

The benchmark intentionally includes **vector-only as the baseline**. Report the measured values, not an assumed improvement. A useful final table will look like this after evaluation:

| Setup | Hit-rate@5 | MRR@20 | Mean latency (ms) |
|---|---:|---:|---:|
| Vector-only | measured | measured | measured |
| Hybrid (BM25 + vector, RRF) | measured | measured | measured |
| Hybrid + cross-encoder rerank | measured | measured | measured |

### Resume bullet

After running the benchmark, replace the placeholders with the actual numbers from the report:

> Built a hybrid BM25 + vector RAG pipeline with RRF fusion and cross-encoder reranking, improving hit-rate@5 from **X%** to **Y%** on a 50-question benchmark (MRR@20: **A** to **B**; retrieval latency: **C ms** to **D ms**).

If the benchmark shows no improvement, report that honestly and use it to tune chunking, retrieval depth, metadata, or the reranker.

## Optional features

- **Metadata filtering:** filter by source/document in the app. The retrieval API also accepts `modified_after` and `modified_before` date filters.
- **Query rewriting:** an optional checkbox asks the configured LLM to rewrite a vague query before retrieval. It is off by default and is not used in the three-way evaluation, keeping the comparison focused on retrieval.
- **Inspectable evidence:** every result shows a stable chunk ID, source/section/page metadata, and the retrieved passage. Generated answers cite these as `[S1]`, `[S2]`, etc.

## Privacy and operational notes

Embeddings and the FAISS/BM25 indexes are local. If you enable answer generation or query rewriting, the question and selected passage text are sent to the configured LLM provider. Do not index confidential material unless your provider and deployment are approved for it. The first model download may require network access; inference can then run locally for embeddings and reranking.
