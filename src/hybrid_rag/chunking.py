from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .models import Chunk, Segment


class TokenChunker:
    """Chunk text using the embedding model's tokenizer when available.

    Offsets let us return readable source text while targeting an exact token
    window. A whitespace-token fallback is provided for slow/custom tokenizers.
    """

    def __init__(self, model_name: str, chunk_size: int = 350, overlap: int = 50):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception as exc:  # pragma: no cover - model download/environment-dependent
            raise RuntimeError(
                f"Could not load tokenizer for {model_name!r}. Check the model name and network access."
            ) from exc

    def _token_spans(self, text: str) -> list[tuple[int, int]]:
        if getattr(self.tokenizer, "is_fast", False):
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            spans = [(int(start), int(end)) for start, end in encoded["offset_mapping"] if end > start]
            if spans:
                return spans
        # Fallback uses word-like units; it remains deterministic but is only an
        # approximation of model tokens.
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]

    def split(self, segments: Iterable[Segment]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap

        for segment_number, segment in enumerate(segments):
            text = segment.text.strip()
            if not text:
                continue
            spans = self._token_spans(text)
            if not spans:
                continue

            start = 0
            ordinal = 0
            while start < len(spans):
                end = min(start + self.chunk_size, len(spans))
                char_start = spans[start][0]
                char_end = spans[end - 1][1]
                chunk_text = text[char_start:char_end].strip()
                if chunk_text:
                    digest_input = "\n".join(
                        (
                            segment.source,
                            segment.section,
                            str(segment.page or ""),
                            str(segment_number),
                            str(ordinal),
                            str(start),
                            chunk_text,
                        )
                    )
                    chunk_id = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:16]
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            source=segment.source,
                            section=segment.section,
                            page=segment.page,
                            modified_at=segment.modified_at,
                        )
                    )
                    ordinal += 1
                if end >= len(spans):
                    break
                start += step

        return chunks
