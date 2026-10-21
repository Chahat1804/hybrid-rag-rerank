from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.evaluate import evaluate_records, parse_benchmark_text  # noqa: E402
from hybrid_rag.index import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    RAGEngine,
    build_index,
)
from hybrid_rag.llm import (  # noqa: E402
    DEFAULT_LLM_MODEL,
    answer_with_openai,
    rewrite_query_with_openai,
    source_label,
)

CORPUS_DIR = ROOT / "data" / "corpus"
INDEX_DIR = ROOT / "data" / "index"


@st.cache_resource(show_spinner="Loading search index and embedding model…")
def get_engine(index_dir: str) -> RAGEngine:
    return RAGEngine(index_dir)


def _save_uploads(uploaded_files) -> list[str]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for uploaded in uploaded_files:
        # Strip any client-side path components; store only the filename.
        name = uploaded.name.replace("\\", "/").split("/")[-1]
        if not name:
            continue
        destination = CORPUS_DIR / name
        destination.write_bytes(uploaded.getvalue())
        saved.append(name)
    return saved


def _catalog_csv(engine: RAGEngine) -> str:
    rows = engine.catalog()
    output = io.StringIO(newline="")
    fields = ["chunk_id", "source", "section", "page", "modified_at", "text_preview"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _render_sidebar() -> tuple[str, str, str, str]:
    with st.sidebar:
        st.header("Corpus and settings")
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        uploaded = st.file_uploader(
            "Add documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="PDF, Word, text, and Markdown are supported. Scanned PDFs need OCR first.",
        )
        if uploaded and st.button("Save uploads to corpus", use_container_width=True):
            try:
                saved = _save_uploads(uploaded)
                st.success(f"Saved {len(saved)} file(s): {', '.join(saved)}")
            except Exception as exc:
                st.error(f"Could not save uploads: {exc}")

        chunk_size = st.slider("Chunk size (embedding-model tokens)", 300, 500, 350, 25)
        overlap = st.slider("Chunk overlap", 25, min(100, chunk_size - 1), 50, 5)
        embedding_model = st.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)
        if st.button("Build / replace index", type="primary", use_container_width=True):
            try:
                with st.spinner("Extracting, chunking, embedding, and indexing documents…"):
                    manifest = build_index(
                        CORPUS_DIR,
                        INDEX_DIR,
                        embedding_model=embedding_model.strip(),
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                get_engine.clear()
                st.success(
                    f"Indexed {manifest['chunk_count']} chunks from "
                    f"{manifest['source_count']} document(s)."
                )
            except Exception as exc:
                st.error(f"Index build failed: {exc}")

        st.divider()
        st.subheader("Answer generation")
        api_key = st.text_input(
            "OpenAI-compatible API key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Used only for this session; it is not written to the project files.",
        )
        llm_model = st.text_input("LLM model", value=os.getenv("OPENAI_MODEL", DEFAULT_LLM_MODEL))
        base_url = st.text_input(
            "API base URL (optional)",
            value=os.getenv("OPENAI_BASE_URL", ""),
            help="Leave blank for the OpenAI API; set this for a compatible provider.",
        )
    return api_key, llm_model.strip(), base_url.strip(), embedding_model.strip()


def _render_ask_tab(engine: RAGEngine | None, api_key: str, llm_model: str, base_url: str) -> None:
    st.subheader("Ask your documents")
    if engine is None:
        st.info("Upload documents in the sidebar, then build an index to begin.")
        return

    st.caption(
        f"Index: {engine.manifest.get('chunk_count', len(engine.chunks)):,} chunks · "
        f"{len(engine.sources)} documents · embedding model `{engine.embedding_model_name}`"
    )
    source_filter = st.multiselect(
        "Restrict search to documents (optional)",
        options=engine.sources,
        help="Leave empty to search the full indexed corpus.",
    )
    mode_label = st.selectbox(
        "Retrieval setup",
        options=["Hybrid + reranking", "Hybrid (BM25 + vector, RRF)", "Vector-only baseline"],
        index=0,
    )
    modes = {
        "Hybrid + reranking": "hybrid_rerank",
        "Hybrid (BM25 + vector, RRF)": "hybrid",
        "Vector-only baseline": "vector",
    }
    rewrite = st.checkbox("Rewrite vague queries before retrieval (optional)", value=False)
    question = st.text_area(
        "Question",
        placeholder="Ask something answerable from your documents…",
        height=90,
    )
    ask = st.button("Search and answer", type="primary", use_container_width=False)

    if ask:
        if not question.strip():
            st.warning("Enter a question first.")
            return
        query = question.strip()
        if rewrite:
            if not api_key:
                st.error("Enter an API key in the sidebar to use query rewriting.")
                return
            try:
                query = rewrite_query_with_openai(query, api_key, llm_model, base_url or None)
                st.caption(f"Rewritten retrieval query: {query}")
            except Exception as exc:
                st.error(f"Query rewriting failed: {exc}")
                return

        filters = {"sources": source_filter or None}
        try:
            with st.spinner("Retrieving supporting passages…"):
                results = engine.retrieve(query, mode=modes[mode_label], top_k=5, candidate_k=20, filters=filters)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            return

        if not results:
            st.info("No matching passages were found in the selected documents.")
            return

        if api_key:
            try:
                with st.spinner("Generating a cited answer…"):
                    answer = answer_with_openai(
                        question=question.strip(),
                        results=results,
                        api_key=api_key,
                        model=llm_model,
                        base_url=base_url or None,
                    )
                st.markdown("### Answer")
                st.markdown(answer)
            except Exception as exc:
                st.error(f"Answer generation failed: {exc}")
                st.info("The retrieved evidence is still available below.")
        else:
            st.info("Add an API key in the sidebar to generate an answer. Retrieved evidence is shown below.")

        st.markdown("### Retrieved evidence")
        for rank, result in enumerate(results, start=1):
            label = source_label(result)
            with st.expander(f"[S{rank}] {label} · score {result.score:.4f}", expanded=(rank == 1)):
                st.code(result.chunk.chunk_id, language=None)
                st.write(result.chunk.text)


def _render_evaluation_tab(engine: RAGEngine | None) -> None:
    st.subheader("Benchmark retrieval quality")
    st.markdown(
        "Upload a CSV or JSONL benchmark with **50 questions** and a known relevant chunk ID per question. "
        "This compares vector-only, BM25 + vector with RRF, and RRF + cross-encoder reranking."
    )
    st.caption(
        "The report includes Hit-rate@5, MRR@20, and steady-state retrieval latency. "
        "Model loading/warm-up and LLM answer generation are excluded from latency."
    )
    if engine is None:
        st.info("Build an index before running the benchmark.")
        return

    st.download_button(
        "Download chunk catalog for labeling gold answers",
        data=_catalog_csv(engine),
        file_name="chunk_catalog.csv",
        mime="text/csv",
    )
    benchmark_file = st.file_uploader(
        "Benchmark file",
        type=["csv", "jsonl"],
        key="benchmark_file",
        help="CSV columns: question,gold_chunk_id (or gold_chunk_ids separated by |).",
    )
    if benchmark_file is None:
        st.code(
            "question,gold_chunk_id\n"
            "What does the policy say about appeals?,<chunk_id from chunk_catalog.csv>",
            language="csv",
        )
        return

    try:
        records = parse_benchmark_text(
            benchmark_file.getvalue().decode("utf-8-sig"),
            benchmark_file.name,
        )
    except Exception as exc:
        st.error(f"Could not read benchmark: {exc}")
        return

    if len(records) != 50:
        st.warning(f"This benchmark has {len(records)} question(s); the target is 50 for the requested comparison.")
    if st.button("Run comparison", type="primary"):
        progress = st.progress(0.0, text="Preparing models…")

        def update_progress(done: int, total: int) -> None:
            progress.progress(done / total, text=f"Evaluated {done} of {total} questions")

        try:
            with st.spinner("Running all three retrieval setups…"):
                summaries, details = evaluate_records(engine, records, progress_callback=update_progress)
            st.markdown("#### Comparison table")
            st.dataframe(summaries, hide_index=True, use_container_width=True)
            st.markdown("#### Per-question results")
            st.dataframe(details, hide_index=True, use_container_width=True)
            report = {
                "metrics": summaries,
                "details": details,
                "notes": {
                    "hit_rate": "Hit-rate@5: any gold chunk in the first five results.",
                    "mrr": "MRR@20: reciprocal rank of the first gold chunk over the ranked top 20.",
                    "latency": "Steady-state retrieval only; model warm-up and LLM generation excluded.",
                },
            }
            st.download_button(
                "Download evaluation report (JSON)",
                data=json.dumps(report, indent=2, ensure_ascii=False),
                file_name="hybrid_rag_evaluation.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="Hybrid RAG with Reranking", page_icon="📚", layout="wide")
    st.title("Hybrid RAG with Reranking")
    st.write("Ask questions over your own documents, inspect cited passages, and benchmark retrieval against a vector-only baseline.")

    api_key, llm_model, base_url, _ = _render_sidebar()
    engine = None
    if (INDEX_DIR / "manifest.json").exists():
        try:
            engine = get_engine(str(INDEX_DIR))
        except Exception as exc:
            st.error(f"Could not load the index: {exc}")

    ask_tab, eval_tab = st.tabs(["Ask documents", "Evaluate retrieval"])
    with ask_tab:
        _render_ask_tab(engine, api_key, llm_model, base_url)
    with eval_tab:
        _render_evaluation_tab(engine)

    st.divider()
    st.caption(f"Cross-encoder: `{DEFAULT_RERANKER_MODEL}` · RRF constant: 60")


if __name__ == "__main__":
    main()
