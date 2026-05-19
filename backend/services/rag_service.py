"""
RAG Service - ChromaDB Vector Store for Prompt Engineering Knowledge.

Responsibilities:
- Initialize ChromaDB persistent client
- Ingest knowledge base documents from text files
- Chunk documents into meaningful segments
- Embed and store chunks in ChromaDB
- Retrieve semantically relevant knowledge for a given query
- Provide retrieved context to enrichment/refinement services

The knowledge base contains prompt engineering best practices
that guide the optimizer and refinement service.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from backend.config import settings
from backend.logger import logger


# ─── Knowledge base file location ─────────────────────────────────
KNOWLEDGE_BASE_PATH = Path("./data/knowledge/prompt_engineering.txt")


class RAGService:
    """
    RAG service using ChromaDB for semantic knowledge retrieval.

    On first initialization:
    - Creates ChromaDB collection
    - Ingests knowledge base documents
    - Builds vector index

    On subsequent calls:
    - Loads existing collection (no re-ingestion)
    - Performs semantic similarity search
    """

    def __init__(self):
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._embedding_fn = None
        self._is_initialized = False
        logger.info("RAGService created (not yet initialized)")

    async def initialize(self) -> None:
        """
        Initializes ChromaDB client and collection.
        Ingests knowledge base if collection is empty.
        Must be called before any retrieve() calls.
        """
        if self._is_initialized:
            return

        logger.info("Initializing RAGService...")

        # ── Ensure chroma persist directory exists ────────────────
        Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

        # ── Create ChromaDB persistent client ─────────────────────
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
        )

        # ── Setup embedding function ──────────────────────────────
        self._embedding_fn = self._create_embedding_function()

        # ── Get or create collection ──────────────────────────────
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )

        logger.info(
            "ChromaDB collection ready | name={name} | docs={count}",
            name=settings.CHROMA_COLLECTION_NAME,
            count=self._collection.count(),
        )

        # ── Ingest knowledge base if collection is empty ──────────
        if self._collection.count() == 0:
            await self._ingest_knowledge_base()
        else:
            logger.info(
                "Using existing ChromaDB collection | {count} documents",
                count=self._collection.count(),
            )

        self._is_initialized = True
        logger.info("✅ RAGService initialized successfully")

    # In backend/services/rag_service.py
    # Replace the _create_embedding_function method with this:

    def _create_embedding_function(self):
        """
        Creates embedding function based on available provider.

        Priority:
        1. Groq provider    → Use ChromaDB default (Groq has no embedding API)
        2. Gemini provider  → Use Google embedding (free)
        3. Ollama provider  → Use Ollama embedding (local, free)
        4. OpenAI provider  → Use OpenAI embedding (paid)
        5. Fallback         → ChromaDB default (all-MiniLM-L6-v2, always free)
        """
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        provider = settings.LLM_PROVIDER

        # ── Gemini: Use Google embedding (free) ───────────────────────
        if provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                from chromadb.utils.embedding_functions import (
                    GoogleGenerativeAiEmbeddingFunction,
                )
                logger.info("Using Google Generative AI embeddings for RAG")
                return GoogleGenerativeAiEmbeddingFunction(
                    api_key=settings.GEMINI_API_KEY,
                    model_name="models/embedding-001",
                )
            except Exception as e:
                logger.warning(
                    "Google embedding init failed, using default | error={e}",
                    e=str(e),
                )

        # ── OpenAI: Use OpenAI embedding (paid) ───────────────────────
        if provider == "openai" and settings.OPENAI_API_KEY:
            try:
                from chromadb.utils.embedding_functions import (
                    OpenAIEmbeddingFunction,
                )
                logger.info("Using OpenAI text-embedding-3-small for RAG")
                return OpenAIEmbeddingFunction(
                    api_key=settings.OPENAI_API_KEY,
                    model_name="text-embedding-3-small",
                )
            except Exception as e:
                logger.warning(
                    "OpenAI embedding init failed, using default | error={e}",
                    e=str(e),
                )

        # ── Default: ChromaDB built-in (all-MiniLM-L6-v2) ────────────
        # FREE, works with ALL providers including Groq and Ollama
        # Downloads model automatically on first run (~90MB)
        logger.info(
            "Using ChromaDB default embedding (all-MiniLM-L6-v2) | "
            "provider={p} | free=True",
            p=provider,
        )
        return DefaultEmbeddingFunction()

    async def _ingest_knowledge_base(self) -> None:
        """
        Reads knowledge base file, chunks it, and ingests into ChromaDB.
        Uses document hash for deduplication.
        """
        logger.info("Ingesting knowledge base from {path}", path=KNOWLEDGE_BASE_PATH)

        if not KNOWLEDGE_BASE_PATH.exists():
            logger.warning(
                "Knowledge base file not found at {path}. "
                "RAG will return empty results.",
                path=KNOWLEDGE_BASE_PATH,
            )
            return

        # Read raw text
        raw_text = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")

        # Chunk the document
        chunks = self._chunk_document(raw_text)

        if not chunks:
            logger.warning("No chunks extracted from knowledge base")
            return

        # Prepare for ChromaDB ingestion
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # Generate stable ID from content hash
            chunk_id = hashlib.md5(chunk["text"].encode()).hexdigest()[:16]
            ids.append(f"chunk_{i:04d}_{chunk_id}")
            documents.append(chunk["text"])
            metadatas.append({
                "title": chunk.get("title", f"Chunk {i}"),
                "category": chunk.get("category", "general"),
                "chunk_index": i,
                "source": "prompt_engineering.txt",
            })

        # Ingest in batches of 50
        batch_size = 50
        total_ingested = 0

        for batch_start in range(0, len(ids), batch_size):
            batch_end = batch_start + batch_size
            self._collection.add(
                ids=ids[batch_start:batch_end],
                documents=documents[batch_start:batch_end],
                metadatas=metadatas[batch_start:batch_end],
            )
            total_ingested += len(ids[batch_start:batch_end])

        logger.info(
            "Knowledge base ingested | chunks={count} | collection_size={size}",
            count=total_ingested,
            size=self._collection.count(),
        )

    def _chunk_document(self, raw_text: str) -> list[dict[str, str]]:
        """
        Chunks the knowledge base document into semantic segments.

        Splits on '---' delimiters (section separators in our knowledge file).
        Extracts TITLE and CATEGORY metadata from each chunk header.

        Args:
            raw_text: Full text of knowledge base file

        Returns:
            List of dicts with 'text', 'title', 'category' keys
        """
        chunks = []
        current_title = "General Knowledge"
        current_category = "general"

        # Split on section separator
        raw_sections = raw_text.split("---")

        for section in raw_sections:
            section = section.strip()
            if not section or len(section) < 50:  # Skip tiny sections
                continue

            # Extract metadata from header lines
            lines = section.split("\n")
            content_lines = []

            for line in lines:
                line = line.strip()
                if line.startswith("TITLE:"):
                    current_title = line.replace("TITLE:", "").strip()
                elif line.startswith("CATEGORY:"):
                    current_category = line.replace("CATEGORY:", "").strip()
                elif line.startswith("#"):
                    # Skip comment lines
                    continue
                elif line:
                    content_lines.append(line)

            if content_lines:
                chunk_text = " ".join(content_lines)
                chunks.append({
                    "text": chunk_text,
                    "title": current_title,
                    "category": current_category,
                })

        logger.debug("Chunked document into {count} segments", count=len(chunks))
        return chunks

    async def retrieve(
        self,
        query: str,
        n_results: int = 4,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieves the most semantically relevant knowledge chunks.

        Args:
            query: The search query (requirement or prompt text)
            n_results: Number of results to return (default 4)
            category_filter: Optional category to filter results

        Returns:
            List of dicts with 'text', 'title', 'category', 'distance' keys
        """
        # ── Auto-initialize if needed ─────────────────────────────
        if not self._is_initialized:
            await self.initialize()

        if self._collection.count() == 0:
            logger.warning("RAG collection is empty — returning no results")
            return []

        # ── Build query filters ───────────────────────────────────
        where_filter = None
        if category_filter:
            where_filter = {"category": {"$eq": category_filter}}

        # ── Clamp n_results to available docs ─────────────────────
        available = self._collection.count()
        n_results = min(n_results, available)

        try:
            # ── Query ChromaDB ────────────────────────────────────
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            # ── Format results ────────────────────────────────────
            retrieved = []
            if results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    retrieved.append({
                        "text": doc,
                        "title": meta.get("title", "Unknown"),
                        "category": meta.get("category", "general"),
                        "distance": round(float(dist), 4),
                        "relevance_score": round(1 - float(dist), 4),
                    })

            logger.debug(
                "RAG retrieved {count} results | query_preview={preview}",
                count=len(retrieved),
                preview=query[:60],
            )
            return retrieved

        except Exception as e:
            logger.error(
                "RAG retrieval failed | error={error}",
                error=str(e),
            )
            return []

    async def retrieve_for_refinement(
        self,
        prompt_text: str,
        weaknesses: list[str],
    ) -> list[dict[str, Any]]:
        """
        Specialized retrieval for prompt refinement.
        Combines prompt text with weakness descriptions for targeted search.

        Args:
            prompt_text: The prompt being refined
            weaknesses: List of identified weaknesses to address

        Returns:
            Relevant knowledge chunks focused on improvement areas
        """
        # Build a targeted query from weaknesses
        weakness_summary = " ".join(weaknesses[:3])  # Top 3 weaknesses
        query = f"improve prompt: {weakness_summary} {prompt_text[:200]}"

        return await self.retrieve(query=query, n_results=5)

    def format_context_for_prompt(
        self,
        retrieved_docs: list[dict[str, Any]],
        max_chars: int = 2000,
    ) -> str:
        """
        Formats retrieved documents into a context string for prompt injection.

        Args:
            retrieved_docs: List of retrieved document dicts
            max_chars: Maximum character limit for the context block

        Returns:
            Formatted context string ready for prompt injection
        """
        if not retrieved_docs:
            return ""

        sections = ["RELEVANT PROMPT ENGINEERING KNOWLEDGE:"]
        total_chars = len(sections[0])

        for i, doc in enumerate(retrieved_docs, 1):
            section = (
                f"\n[{i}] {doc['title']} (relevance: {doc['relevance_score']:.2f})\n"
                f"{doc['text']}\n"
            )
            if total_chars + len(section) > max_chars:
                break
            sections.append(section)
            total_chars += len(section)

        return "\n".join(sections)

    def get_collection_stats(self) -> dict[str, Any]:
        """Returns stats about the knowledge base collection."""
        if not self._is_initialized or not self._collection:
            return {"initialized": False, "document_count": 0}

        return {
            "initialized": True,
            "document_count": self._collection.count(),
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "persist_dir": settings.CHROMA_PERSIST_DIR,
        }


# ─── Singleton ────────────────────────────────────────────────────
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Returns the RAGService singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service