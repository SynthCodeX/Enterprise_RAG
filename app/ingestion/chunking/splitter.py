from typing import List

import logfire
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[str]:
    """Split text recursively: paragraphs -> lines -> sentences -> words."""
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        # Tries larger separators first to keep related text together
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = [c.strip() for c in splitter.split_text(text) if c.strip()]


        # paragraphs = text.split("\n\n")
        # chunks = []
        # current_chunk = ""

        # for p in paragraphs:
        #     if len(current_chunk) + len(p) < chunk_size:
        #         current_chunk += p + "\n\n"
        #     else:
        #         if current_chunk.strip():
        #             chunks.append(current_chunk.strip())
        #         current_chunk = p + "\n\n"

        # if current_chunk.strip():
        #     chunks.append(current_chunk.strip())

        # valid_chunks = [c for c in chunks if c.strip()]


        logfire.info(f"✅ Generated {len(chunks)} chunks")
        return chunks

