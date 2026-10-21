from __future__ import annotations

import argparse
import json

from .index import DEFAULT_EMBEDDING_MODEL, build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local FAISS + BM25 document index.")
    parser.add_argument("--corpus-dir", default="data/corpus", help="Folder containing PDF/DOCX/TXT/MD files")
    parser.add_argument("--index-dir", default="data/index", help="Output index folder")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=350)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    manifest = build_index(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
