from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Segment:
    """A source text segment before chunking."""

    text: str
    source: str
    section: str
    page: int | None
    modified_at: str


@dataclass(frozen=True)
class Chunk:
    """A searchable passage with traceable source metadata."""

    chunk_id: str
    text: str
    source: str
    section: str
    page: int | None
    modified_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(value["chunk_id"]),
            text=str(value["text"]),
            source=str(value.get("source", "unknown")),
            section=str(value.get("section", "Document")),
            page=int(value["page"]) if value.get("page") is not None else None,
            modified_at=str(value.get("modified_at", "")),
        )


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    method: str
