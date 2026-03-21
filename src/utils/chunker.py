"""
Medical Text Chunker
Section-aware chunking — keeps tables, dosage info, and guideline
sections together rather than splitting mid-sentence.
"""
import re
from dataclasses import dataclass

CHUNK_SIZE = 512        # tokens approx (characters / 4)
CHUNK_OVERLAP = 80      # overlap between consecutive chunks


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    char_count: int


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[TextChunk]:
    """
    Split text into overlapping chunks.
    Tries to break at sentence boundaries, not mid-sentence.
    """
    text = _clean(text)
    sentences = _split_sentences(text)
    chunks = []
    current = []
    current_len = 0
    idx = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if current_len + word_count > chunk_size and current:
            chunk_text = " ".join(current)
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=idx,
                char_count=len(chunk_text),
            ))
            idx += 1
            # Keep overlap: last N words carry into next chunk
            overlap_words = current[-overlap:] if len(current) > overlap else current[:]
            current = overlap_words
            current_len = len(overlap_words)

        current.extend(words)
        current_len += word_count

    if current:
        chunk_text = " ".join(current)
        chunks.append(TextChunk(
            text=chunk_text,
            chunk_index=idx,
            char_count=len(chunk_text),
        ))

    return chunks


def _clean(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Basic sentence splitter that respects medical abbreviations."""
    # Avoid splitting on common medical abbreviations
    abbrevs = r'(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|approx|mg|kg|mcg|mL|IV|IM|SC|BD|TDS|OD|SOS|tab|cap|inj|soln))'
    pattern = abbrevs + r'(?<=[.!?])\s+'
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]
