from typing import List

import logfire


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 150,
) -> List[str]:
    """
    Split text into bounded chunks while preserving paragraph boundaries
    where possible. Oversized paragraphs are split further with overlap.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:
            # If the paragraph itself is too large, flush the current chunk
            # and split the paragraph into bounded overlapping pieces.
            if len(paragraph) > chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""

                start = 0
                while start < len(paragraph):
                    end = start + chunk_size
                    piece = paragraph[start:end].strip()

                    if piece:
                        chunks.append(piece)

                    if end >= len(paragraph):
                        break

                    start = end - overlap

                continue

            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())

                # Preserve a small tail of the previous chunk as overlap.
                tail = current[-overlap:] if current else ""
                current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

                # If overlap caused the chunk to exceed the limit, keep
                # the paragraph itself as the new chunk.
                if len(current) > chunk_size:
                    chunks.append(current[:chunk_size].strip())
                    current = current[chunk_size - overlap :].strip()

        if current:
            chunks.append(current.strip())

        valid_chunks = [c for c in chunks if c]

        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
