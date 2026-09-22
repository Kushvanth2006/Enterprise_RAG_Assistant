import time

import logfire
import requests
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.config import settings

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_JINA_RERANK_MODEL = "jina-reranker-v3"

_ranker = None


class _JinaReranker:
    """Thin wrapper around the Jina Reranker API."""

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[str]:
        """Score and reorder documents against the query via the Jina API."""
        response = requests.post(
            _JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _JINA_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": True,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        results = payload.get("results", [])
        reranked_docs = []

        for res in results[:top_n]:
            doc_text = res.get("document")

            if doc_text is None:
                index = res.get("index")
                if index is not None and 0 <= index < len(documents):
                    doc_text = documents[index]

            if doc_text is not None:
                reranked_docs.append(doc_text)

        return reranked_docs

    def rerank_indices(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[int]:
        """Return original document indices ordered by Jina relevance."""
        response = requests.post(
            _JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _JINA_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            },
            timeout=60,
        )
        response.raise_for_status()

        results = response.json().get("results", [])

        return [int(res["index"]) for res in results[:top_n] if "index" in res]

    def rerank_with_indices(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[dict]:
        """Return ranked documents together with their original indices."""
        response = requests.post(
            _JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _JINA_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": True,
            },
            timeout=60,
        )
        response.raise_for_status()

        results = response.json().get("results", [])
        ranked = []

        for res in results[:top_n]:
            index = res.get("index")

            if index is None or not 0 <= int(index) < len(documents):
                continue

            doc_text = res.get("document")

            if doc_text is None:
                doc_text = documents[int(index)]

            ranked.append(
                {
                    "index": int(index),
                    "document": doc_text,
                    "score": res.get("relevance_score"),
                }
            )

        return ranked


def _get_ranker() -> _JinaReranker:
    """Returns the Jina Reranker wrapper (lazy singleton)."""
    global _ranker

    if _ranker is None:
        logfire.info("🧠 Initializing Jina Reranker v3 via API...")
        _ranker = _JinaReranker()

    return _ranker


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _rerank(
    query: str,
    documents: list[str],
    top_n: int,
) -> list[str]:
    """Core Jina API reranking with retry on transient failures."""
    ranker = _get_ranker()
    return ranker.rerank(query, documents, top_n)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _rerank_indices(
    query: str,
    documents: list[str],
    top_n: int,
) -> list[int]:
    """Core Jina API reranking that preserves original document indices."""
    ranker = _get_ranker()
    return ranker.rerank_indices(query, documents, top_n)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _rerank_with_indices(
    query: str,
    documents: list[str],
    top_n: int,
) -> list[dict]:
    """Core Jina API reranking returning documents and original indices."""
    ranker = _get_ranker()
    return ranker.rerank_with_indices(query, documents, top_n)


def rerank_with_indices(
    query: str,
    documents: list[str],
    top_n: int = 5,
) -> list[dict]:
    """Rerank documents once and return documents with original indices."""

    if not documents:
        return []

    if not settings.JINA_API_KEY:
        logfire.warning("⚠️ JINA_API_KEY not set — skipping reranking.")
        return [
            {
                "index": index,
                "document": document,
                "score": None,
            }
            for index, document in enumerate(documents[:top_n])
        ]

    try:
        return _rerank_with_indices(query, documents, top_n)

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed after retries: {e}")

        return [
            {
                "index": index,
                "document": document,
                "score": None,
            }
            for index, document in enumerate(documents[:top_n])
        ]


def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 5,
) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    Retries transient failures and falls back to the original Qdrant order if
    reranking ultimately fails.
    """
    if not documents:
        return []

    if not settings.JINA_API_KEY:
        logfire.warning("⚠️ JINA_API_KEY not set — skipping reranking.")
        return documents[:top_n]

    start_time = time.time()

    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to Jina Reranker API...")

    try:
        reranked_docs = _rerank(query, documents, top_n)

        duration = time.time() - start_time

        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s.")

        return reranked_docs

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed after retries: {e}")

        return documents[:top_n]


def rerank_document_indices(
    query: str,
    documents: list[str],
    top_n: int = 5,
) -> list[int]:
    """
    Return the original indices of the top-ranked documents.

    Falls back to the original Qdrant order if Jina reranking fails.
    """
    if not documents:
        return []

    if not settings.JINA_API_KEY:
        logfire.warning("⚠️ JINA_API_KEY not set — skipping reranking.")
        return list(range(min(top_n, len(documents))))

    try:
        return _rerank_indices(query, documents, top_n)

    except Exception as e:
        logfire.error(f"❌ [Reranker] Index reranking failed after retries: {e}")

        return list(range(min(top_n, len(documents))))
