from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import get_settings


@dataclass
class ChunkV3:
    chunk_id: str
    doc_id: str
    chunk_type: str
    section_title: str
    text: str
    contextual_text: str
    char_count: int
    token_estimate: int
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedSectionLite:
    title: str
    text: str
    section_index: int
    has_table: bool = False
    tables: list[dict[str, Any]] = field(default_factory=list)


class HierarchicalChunkerV3:
    # Load configuration from settings - can be overridden at initialization
    _settings = None

    @classmethod
    def _get_settings(cls):
        if cls._settings is None:
            cls._settings = get_settings()
        return cls._settings

    @property
    def TARGET_CHUNK_TOKENS(self):
        return self._get_settings().chunk_target_tokens

    @property
    def MAX_CHUNK_TOKENS(self):
        return self._get_settings().chunk_max_tokens

    @property
    def OVERLAP_TOKENS(self):
        return self._get_settings().chunk_overlap_tokens

    @property
    def MIN_CHUNK_TOKENS(self):
        return self._get_settings().chunk_min_tokens

    _ABBREVIATIONS = {
        "et al.": "et al<PERIOD>",
        "fig.": "fig<PERIOD>",
        "e.g.": "e<PERIOD>g<PERIOD>",
        "i.e.": "i<PERIOD>e<PERIOD>",
        "vs.": "vs<PERIOD>",
        "mg/dl": "mg<SLASH>dl",
        "p.o.": "p<PERIOD>o<PERIOD>",
        "i.v.": "i<PERIOD>v<PERIOD>",
    }

    def chunk_document(self, doc: Any, doc_metadata: dict[str, Any]) -> list[ChunkV3]:
        doc_id = self._resolve_doc_id(doc)
        title = self._get_field(doc, "title", "")
        abstract = self._get_field(doc, "abstract", "")
        mesh_terms = self._coerce_str_list(self._get_field(doc, "mesh_terms", []))
        keywords = self._coerce_str_list(self._get_field(doc, "keywords", []))

        chunks: list[ChunkV3] = []

        summary_parts = [title.strip(), abstract.strip()]
        if mesh_terms:
            summary_parts.append("MeSH: " + ", ".join(mesh_terms))
        if keywords:
            summary_parts.append("Keywords: " + ", ".join(keywords))
        summary_text = "\n\n".join(part for part in summary_parts if part)
        if summary_text:
            chunks.append(
                self._make_chunk(
                    doc_id=doc_id,
                    chunk_type="doc_summary",
                    section_title="Document Summary",
                    text=summary_text,
                    chunk_index=len(chunks),
                    doc_title=title,
                    doc_metadata=doc_metadata,
                )
            )

        sections = self._extract_sections(doc)
        for section in sections:
            section_title = section.title or f"Section {section.section_index + 1}"

            if section.has_table and section.tables:
                for table_index, table in enumerate(section.tables[:20]):
                    table_text = self._format_table_chunk(table)
                    if not table_text.strip():
                        continue
                    chunks.append(
                        self._make_chunk(
                            doc_id=doc_id,
                            chunk_type="table",
                            section_title=f"{section_title} - Table {table_index + 1}",
                            text=table_text,
                            chunk_index=len(chunks),
                            doc_title=title,
                            doc_metadata=doc_metadata,
                        )
                    )

            section_text = (section.text or "").strip()
            if not section_text:
                continue

            section_tokens = self._estimate_tokens(section_text)
            if section_tokens <= self.TARGET_CHUNK_TOKENS:
                chunks.append(
                    self._make_chunk(
                        doc_id=doc_id,
                        chunk_type="paragraph",
                        section_title=section_title,
                        text=section_text,
                        chunk_index=len(chunks),
                        doc_title=title,
                        doc_metadata=doc_metadata,
                    )
                )
                continue

            for split_text in self._sentence_window_split(section_text):
                split_text = split_text.strip()
                if self._estimate_tokens(split_text) < self.MIN_CHUNK_TOKENS:
                    continue
                chunks.append(
                    self._make_chunk(
                        doc_id=doc_id,
                        chunk_type="paragraph",
                        section_title=section_title,
                        text=split_text,
                        chunk_index=len(chunks),
                        doc_title=title,
                        doc_metadata=doc_metadata,
                    )
                )

        total_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.chunk_index = idx
            chunk.total_chunks = total_chunks

        return chunks

    def _extract_sections(self, doc: Any) -> list[ParsedSectionLite]:
        sections_value = self._get_field(doc, "sections", None)

        if isinstance(sections_value, list) and sections_value:
            parsed_sections: list[ParsedSectionLite] = []
            for i, sec in enumerate(sections_value):
                title = (
                    self._get_field(sec, "title", "")
                    if not isinstance(sec, str)
                    else ""
                )
                text = (
                    self._get_field(sec, "text", "")
                    if not isinstance(sec, str)
                    else sec
                )
                section_index = (
                    self._get_field(sec, "section_index", i)
                    if not isinstance(sec, str)
                    else i
                )
                has_table = (
                    bool(self._get_field(sec, "has_table", False))
                    if not isinstance(sec, str)
                    else False
                )
                tables = (
                    self._get_field(sec, "tables", [])
                    if not isinstance(sec, str)
                    else []
                )
                parsed_sections.append(
                    ParsedSectionLite(
                        title=str(title or ""),
                        text=str(text or ""),
                        section_index=int(section_index),
                        has_table=has_table,
                        tables=tables if isinstance(tables, list) else [],
                    )
                )
            return parsed_sections

        content = self._get_field(doc, "content", "")
        if not content.strip():
            return []

        return self._split_content_into_sections(content)

    def _split_content_into_sections(self, content: str) -> list[ParsedSectionLite]:
        # Extended header patterns for medical/scientific documents
        header_re = re.compile(
            r"^(?:\d+\.\s*)?("
            # Standard sections
            r"Abstract|Introduction|Background|Methods?|Materials and Methods|Results|Discussion|Conclusion|Conclusions|"
            # Common combined sections
            r"Results and Discussion|"
            # Population/study design sections
            r"Study Population|Patient Population|Demographics|Study Design|"
            # Additional medical sections
            r"Methods and Results|Background and Objectives|Objectives|Aims|"
            # Supplementary sections
            r"Supplementary|References|Acknowledgments"
            r")\s*$",
            re.IGNORECASE,
        )

        lines = content.splitlines()
        sections: list[ParsedSectionLite] = []
        current_title = "Main"
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
                continue

            if header_re.match(stripped):
                if current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        sections.append(
                            ParsedSectionLite(
                                title=current_title,
                                text=section_text,
                                section_index=len(sections),
                            )
                        )
                current_title = stripped
                current_lines = []
            else:
                current_lines.append(stripped)

        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append(
                    ParsedSectionLite(
                        title=current_title,
                        text=section_text,
                        section_index=len(sections),
                    )
                )

        return sections

    def _sentence_window_split(self, text: str) -> list[str]:
        protected = text
        for original, replacement in self._ABBREVIATIONS.items():
            protected = re.sub(
                re.escape(original), replacement, protected, flags=re.IGNORECASE
            )

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
            if s.strip()
        ]
        restored_sentences: list[str] = []
        for sentence in sentences:
            restored = sentence
            for original, replacement in self._ABBREVIATIONS.items():
                restored = restored.replace(replacement, original)
            restored_sentences.append(restored)

        windows: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in restored_sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            if current_tokens + sentence_tokens <= self.MAX_CHUNK_TOKENS:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
                continue

            if current_sentences:
                windows.append(" ".join(current_sentences).strip())

                overlap_sentences: list[str] = []
                overlap_tokens = 0
                for prev_sentence in reversed(current_sentences):
                    tks = self._estimate_tokens(prev_sentence)
                    if overlap_tokens + tks > self.OVERLAP_TOKENS:
                        break
                    overlap_sentences.insert(0, prev_sentence)
                    overlap_tokens += tks

                current_sentences = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
            else:
                windows.append(sentence)
                current_sentences = []
                current_tokens = 0

        if current_sentences:
            windows.append(" ".join(current_sentences).strip())

        return windows

    def _format_table_chunk(self, table: dict[str, Any]) -> str:
        caption = str(table.get("caption", "")).strip()

        if "headers" in table and "rows" in table:
            headers = [str(h).strip() for h in (table.get("headers") or [])]
            rows = table.get("rows") or []

            lines: list[str] = []
            if caption:
                lines.append(caption)

            for row in rows[:20]:
                if not isinstance(row, (list, tuple)):
                    continue
                fields: list[str] = []
                for idx, value in enumerate(row):
                    header = (
                        headers[idx]
                        if idx < len(headers) and headers[idx]
                        else f"Col{idx + 1}"
                    )
                    fields.append(f"{header}: {str(value).strip()}")
                if fields:
                    lines.append(" | ".join(fields))
            return "\n".join(lines).strip()

        text = str(table.get("text", "")).strip()
        lines = [line for line in [caption, text] if line]
        return "\n".join(lines).strip()

    def _make_chunk(
        self,
        *,
        doc_id: str,
        chunk_type: str,
        section_title: str,
        text: str,
        chunk_index: int,
        doc_title: str,
        doc_metadata: dict[str, Any],
    ) -> ChunkV3:
        token_estimate = self._estimate_tokens(text)
        chunk_id = f"{doc_id}_c{chunk_index}"
        contextual_text = self._build_contextual_text(
            source=str(doc_metadata.get("source", "unknown")),
            doc_type=str(doc_metadata.get("doc_type", "unknown")),
            doc_title=doc_title,
            section_title=section_title,
            chunk_text=text,
        )
        return ChunkV3(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_type=chunk_type,
            section_title=section_title,
            text=text,
            contextual_text=contextual_text,
            char_count=len(text),
            token_estimate=token_estimate,
            chunk_index=chunk_index,
            total_chunks=0,
            metadata=dict(doc_metadata),
        )

    def _build_contextual_text(
        self,
        *,
        source: str,
        doc_type: str,
        doc_title: str,
        section_title: str,
        chunk_text: str,
    ) -> str:
        return (
            f"Source: {source}\n"
            f"Document type: {doc_type}\n"
            f"Title: {doc_title}\n"
            f"Section: {section_title}\n\n"
            f"{chunk_text}"
        )

    def _resolve_doc_id(self, doc: Any) -> str:
        doc_id = self._get_field(doc, "doc_id", None)
        if isinstance(doc_id, str) and doc_id.strip():
            return doc_id

        maybe_pmid = self._get_field(doc, "pmid", None)
        if isinstance(maybe_pmid, str) and maybe_pmid.strip():
            return f"pmid_{maybe_pmid.strip()}"

        title = self._get_field(doc, "title", "untitled")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(title)).strip("_").lower()[:40]
        return f"doc_{slug or 'unknown'}"

    def _estimate_tokens(self, text: str) -> int:
        # Use word-based estimation: more accurate for medical/scientific text
        # Average token is ~1.3 words for technical content
        word_count = len(text.split())
        return max(1, int(word_count * 1.3))

    def _get_field(self, obj: Any, key: str, default: Any) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _coerce_str_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return []
