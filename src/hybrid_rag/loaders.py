from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .models import Segment

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _modified_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _plain_text_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown/plain text on explicit, reasonably safe heading patterns."""
    heading_patterns = (
        re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$"),
        re.compile(r"^\s*((?:chapter|section|unit|part)\s+[\w.-]+.*)$", re.IGNORECASE),
        re.compile(r"^\s*(\d+(?:\.\d+)*[.)]?\s+.{2,100})\s*$"),
    )
    sections: list[tuple[str, str]] = []
    current_title = "Document"
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        content = "\n".join(body).strip()
        if content:
            sections.append((current_title, content))
        body = []

    for line in text.splitlines():
        match = next((pattern.match(line) for pattern in heading_patterns if pattern.match(line)), None)
        if match:
            flush()
            current_title = match.group(1).strip().strip("# ") or "Document"
        else:
            body.append(line)
    flush()

    if not sections and text.strip():
        return [("Document", text.strip())]
    return sections


def _pdf_heading(text: str, page_number: int) -> str:
    for line in text.splitlines()[:10]:
        candidate = re.sub(r"\s+", " ", line).strip()
        if not candidate or len(candidate) > 100:
            continue
        if re.match(r"^(chapter|section|unit|part)\b", candidate, re.IGNORECASE):
            return candidate
        if candidate.isupper() and len(candidate.split()) <= 12:
            return candidate
    return f"Page {page_number}"


def _load_pdf(path: Path, modified_at: str, source_name: str) -> list[Segment]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("PDF support requires pypdf. Install requirements.txt.") from exc

    reader = PdfReader(str(path))
    segments: list[Segment] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\s*\n\s*", " ", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            segments.append(
                Segment(
                    text=text,
                    source=source_name,
                    section=_pdf_heading(text, page_number),
                    page=page_number,
                    modified_at=modified_at,
                )
            )
    return segments


def _load_docx(path: Path, modified_at: str, source_name: str) -> list[Segment]:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("DOCX support requires python-docx. Install requirements.txt.") from exc

    document = Document(str(path))
    sections: list[Segment] = []
    current_title = "Document"
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        text = "\n".join(body).strip()
        if text:
            sections.append(
                Segment(
                    text=text,
                    source=source_name,
                    section=current_title,
                    page=None,
                    modified_at=modified_at,
                )
            )
        body = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = getattr(paragraph.style, "name", "") or ""
        if style_name.lower().startswith("heading"):
            flush()
            current_title = text[:160]
        else:
            body.append(text)
    flush()
    return sections


def load_file(path: str | Path, source_name: str | None = None) -> list[Segment]:
    path = Path(path)
    source_name = source_name or path.name
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return []

    modified_at = _modified_at(path)
    if suffix == ".pdf":
        return _load_pdf(path, modified_at, source_name)
    if suffix == ".docx":
        return _load_docx(path, modified_at, source_name)

    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        Segment(
            text=section_text,
            source=source_name,
            section=section_title,
            page=None,
            modified_at=modified_at,
        )
        for section_title, section_text in _plain_text_sections(text)
    ]


def load_corpus(corpus_dir: str | Path) -> list[Segment]:
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus folder does not exist: {root}")

    segments: list[Segment] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS):
        source_name = path.relative_to(root).as_posix()
        segments.extend(load_file(path, source_name=source_name))
    return segments
