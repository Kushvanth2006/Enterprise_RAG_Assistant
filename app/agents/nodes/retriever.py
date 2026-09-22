import logfire

from app.agents.state import AgentState
from app.config import settings
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_with_indices


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]

    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")

        raw_results = search_enterprise_knowledge(
            query,
            limit=15,
        )

        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        doc_contents = [doc["content"] for doc in raw_results]

        with logfire.span("⚖️ Semantic Reranking"):
            ranked_results = rerank_with_indices(
                query,
                doc_contents,
                top_n=5,
            )

            min_score = settings.RERANK_MIN_RELEVANCE_SCORE

            filtered_results = [
                result for result in ranked_results if result["score"] is None or result["score"] >= min_score
            ]

            logfire.info(
                f"Reranking complete. {len(ranked_results)} candidates returned, "
                f"{len(filtered_results)} passed relevance threshold "
                f"{min_score:.4f}."
            )

            for result in ranked_results:
                score = result["score"]

                if score is not None:
                    logfire.info(f"[Reranker] score={score:.4f} index={result['index']}")
                else:
                    logfire.info(f"[Reranker] score=None index={result['index']}")

    formatted_docs = [f"CONTENT: {result['document']}" for result in filtered_results]

    # Return sources only for documents that passed the relevance threshold.
    sources = []

    for result in filtered_results:
        index = result["index"]

        if 0 <= index < len(raw_results):
            source = raw_results[index].get("source")

            if source and source not in sources:
                sources.append(source)

    if not filtered_results:
        return {
            "documents": [],
            "sources": [],
            "status": "No relevant context found.",
            "plan": state["plan"] + ["No Relevant Context"],
        }

    return {
        "documents": formatted_docs,
        "sources": sources,
        "status": "Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"],
    }
