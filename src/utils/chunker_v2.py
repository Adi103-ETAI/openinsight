"""
Hierarchical Medical Text Chunker v2
Three-level chunking: section → semantic segments → sentence fallback.
Target: 250-350 tokens, 50-75 token overlap.
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


TARGET_TOKENS = 300  # target chunk size in tokens (approx words * 1.3)
MAX_TOKENS = 400  # hard max before forced split
OVERLAP_TOKENS = 60  # overlap between chunks
MIN_CHUNK_CHARS = 100  # skip chunks shorter than this


# Section header patterns — ordered by specificity
SECTION_PATTERNS = [
    # Clinical sections (high value)
    r"^(?:\d+\.?\d*\s+)?(?:treatment|management|therapy|therapeutic|dosage|dose|drug|antibiotic|"
    r"medication|prescription|clinical|diagnosis|diagnostic|investigation|laboratory|"
    r"guideline|recommendation|protocol|procedure|intervention|prevention|prophylaxis)",
    # Abstract sections
    r"^(?:abstract|summary|conclusion|result|finding|outcome|discussion)",
    # Background sections (lower value)
    r"^(?:introduction|background|overview|history|epidemiology|etiology|pathophysiology|"
    r"method|material|appendix|reference|acknowledgement|foreword|index)",
]


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    char_count: int
    token_count: int
    section: Optional[str] = None
    page_number: Optional[int] = None
    level: str = "paragraph"  # "section" | "paragraph" | "sentence"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.3"""
    return int(len(text.split()) * 1.3)


def _detect_section_header(line: str) -> Optional[str]:
    """Return section name if line looks like a section header, else None."""
    line = line.strip()
    if not line or len(line) > 120:
        return None
    
    # Skip table-like lines (contain | or + or --- patterns)
    if '|' in line or '+-' in line or '-+' in line or line.count('-') > 10:
        return None
    
    # Short line, possibly title-cased or numbered
    if len(line) < 80 and (
        line.isupper() or re.match(r"^\d+\.?\d*\s+\w", line) or re.match(r"^[A-Z][a-z]+ [A-Z]", line)
    ):
        for pattern in SECTION_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return line
    return None


def _split_into_sections(text: str) -> list[tuple[Optional[str], str]]:
    """
    Split text into (section_header, section_content) pairs.
    Returns list of tuples.
    """
    lines = text.splitlines()
    sections = []
    current_header = None
    current_lines = []

    for line in lines:
        header = _detect_section_header(line)
        if header and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_header, content))
            current_header = header
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_header, content))

    return sections if sections else [(None, text)]


def _split_by_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs.
    Table blocks ([TABLE]...[/TABLE]) are kept as single units — never split.
    """
    segments = []
    
    # Split on table markers first
    table_pattern = re.compile(r'(\[TABLE\].*?\[/TABLE\])', re.DOTALL)
    parts = table_pattern.split(text)
    
    for part in parts:
        if part.startswith('[TABLE]'):
            # Keep entire table as one segment
            segments.append(part.strip())
        else:
            # Split non-table text by paragraphs
            paragraphs = re.split(r'\n\s*\n', part)
            segments.extend([p.strip() for p in paragraphs if p.strip()])
    
    return segments


def _split_sentences(text: str) -> list[str]:
    """Sentence splitter respecting medical abbreviations."""
    abbrevs = [
        "Dr",
        "Mr",
        "Mrs",
        "Ms",
        "Prof",
        "vs",
        "etc",
        "approx",
        "mg",
        "kg",
        "mcg",
        "mL",
        "IV",
        "IM",
        "SC",
        "BD",
        "TDS",
        "OD",
        "SOS",
        "tab",
        "cap",
        "inj",
        "soln",
        "approx",
        "Fig",
    ]
    protected = text
    for abbrev in abbrevs:
        protected = protected.replace(f"{abbrev}.", f"{abbrev}<DOT>")
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def _merge_into_chunks(
    segments: list[str],
    section: Optional[str],
    start_index: int,
) -> list[TextChunk]:
    """
    Merge segments into target-sized chunks with overlap.
    """
    chunks = []
    current_words = []
    current_tokens = 0
    idx = start_index

    # Overlap buffer: last N words from previous chunk
    overlap_buffer = []

    for segment in segments:
        # Table blocks always get their own chunk — never split or merge with other content
        if segment.startswith('[TABLE]'):
            if current_words:
                # Flush current chunk first
                chunk_text_str = " ".join(current_words)
                if len(chunk_text_str) >= MIN_CHUNK_CHARS:
                    chunks.append(
                        TextChunk(
                            text=chunk_text_str,
                            chunk_index=idx,
                            char_count=len(chunk_text_str),
                            token_count=current_tokens,
                            section=section,
                            level="paragraph",
                        )
                    )
                    idx += 1
                current_words = []
                current_tokens = 0
            
            # Add table as its own chunk - strip markers but keep content
            # Tables are always kept even if small (they contain structured data)
            table_text = segment.replace('[TABLE]', '').replace('[/TABLE]', '').strip()
            if table_text:
                chunks.append(
                    TextChunk(
                        text=table_text,
                        chunk_index=idx,
                        char_count=len(table_text),
                        token_count=_estimate_tokens(table_text),
                        section=section,
                        level="table",
                    )
                )
                idx += 1
            continue
        
        segment_tokens = _estimate_tokens(segment)
        segment_words = segment.split()

        # If single segment already exceeds max, split by sentences
        if segment_tokens > MAX_TOKENS:
            sentences = _split_sentences(segment)
            for sentence in sentences:
                s_tokens = _estimate_tokens(sentence)
                s_words = sentence.split()

                if current_tokens + s_tokens > TARGET_TOKENS and current_words:
                    chunk_text = " ".join(current_words)
                    if len(chunk_text) >= MIN_CHUNK_CHARS:
                        chunks.append(
                            TextChunk(
                                text=chunk_text,
                                chunk_index=idx,
                                char_count=len(chunk_text),
                                token_count=current_tokens,
                                section=section,
                                level="sentence",
                            )
                        )
                        idx += 1
                    # Overlap
                    overlap_words = (
                        current_words[-OVERLAP_TOKENS:] if len(current_words) > OVERLAP_TOKENS else current_words[:]
                    )
                    current_words = overlap_buffer + overlap_words + s_words
                    current_tokens = _estimate_tokens(" ".join(current_words))
                    overlap_buffer = []
                else:
                    current_words.extend(s_words)
                    current_tokens += s_tokens
        else:
            if current_tokens + segment_tokens > TARGET_TOKENS and current_words:
                chunk_text = " ".join(current_words)
                if len(chunk_text) >= MIN_CHUNK_CHARS:
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            chunk_index=idx,
                            char_count=len(chunk_text),
                            token_count=current_tokens,
                            section=section,
                            level="paragraph",
                        )
                    )
                    idx += 1
                overlap_words = (
                    current_words[-OVERLAP_TOKENS:] if len(current_words) > OVERLAP_TOKENS else current_words[:]
                )
                current_words = overlap_words + segment_words
                current_tokens = _estimate_tokens(" ".join(current_words))
            else:
                current_words.extend(segment_words)
                current_tokens += segment_tokens

    # Flush remaining
    if current_words:
        chunk_text = " ".join(current_words)
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    chunk_index=idx,
                    char_count=len(chunk_text),
                    token_count=current_tokens,
                    section=section,
                    level="paragraph",
                )
            )

    return chunks


def chunk_text_v2(text: str) -> list[TextChunk]:
    """
    Main entry point. Hierarchical chunking:
    1. Split into sections by header detection
    2. Split each section into paragraphs
    3. Merge paragraphs into target-sized chunks with overlap
    4. Sentence fallback for oversized segments
    """
    if not text or not text.strip():
        return []

    sections = _split_into_sections(text)
    all_chunks = []
    global_idx = 0

    for section_header, section_content in sections:
        paragraphs = _split_by_paragraphs(section_content)
        if not paragraphs:
            continue
        section_chunks = _merge_into_chunks(paragraphs, section_header, global_idx)
        all_chunks.extend(section_chunks)
        global_idx += len(section_chunks)

    logger.debug(f"Chunked {len(text)} chars into {len(all_chunks)} chunks")
    return all_chunks
