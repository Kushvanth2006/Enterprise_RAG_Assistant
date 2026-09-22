from unittest.mock import patch

from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriever import retrieve_node


def make_state():
    return {
        "current_query": "What is Qdrant?",
        "messages": [
            {
                "role": "user",
                "content": "What is Qdrant?",
            }
        ],
        "documents": [],
        "sources": [],
        "plan": ["Intent: Technical"],
        "status": "",
    }


@patch("app.agents.nodes.retriever.search_enterprise_knowledge")
@patch("app.agents.nodes.retriever.rerank_with_indices")
def test_retriever_filters_low_relevance_results(
    mock_rerank,
    mock_search,
):
    mock_search.return_value = [
        {"content": "low relevance", "source": "low.pdf"},
        {"content": "relevant", "source": "relevant.pdf"},
        {"content": "very relevant", "source": "very.pdf"},
    ]

    mock_rerank.return_value = [
        {
            "index": 0,
            "document": "low relevance",
            "score": 0.04,
        },
        {
            "index": 1,
            "document": "relevant",
            "score": 0.05,
        },
        {
            "index": 2,
            "document": "very relevant",
            "score": 0.20,
        },
    ]

    result = retrieve_node(make_state())

    assert len(result["documents"]) == 2
    assert "relevant" in result["documents"][0]
    assert "very relevant" in result["documents"][1]

    assert "low.pdf" not in result["sources"]
    assert "relevant.pdf" in result["sources"]
    assert "very.pdf" in result["sources"]

    assert result["status"] == "Found technical context."


@patch("app.agents.nodes.retriever.search_enterprise_knowledge")
@patch("app.agents.nodes.retriever.rerank_with_indices")
def test_retriever_returns_no_context_when_all_scores_are_below_threshold(
    mock_rerank,
    mock_search,
):
    mock_search.return_value = [
        {"content": "irrelevant one", "source": "one.pdf"},
        {"content": "irrelevant two", "source": "two.pdf"},
    ]

    mock_rerank.return_value = [
        {
            "index": 0,
            "document": "irrelevant one",
            "score": 0.04,
        },
        {
            "index": 1,
            "document": "irrelevant two",
            "score": 0.01,
        },
    ]

    result = retrieve_node(make_state())

    assert result["documents"] == []
    assert result["sources"] == []
    assert result["status"] == "No relevant context found."
    assert result["plan"][-1] == "No Relevant Context"


@patch("app.agents.nodes.retriever.search_enterprise_knowledge")
@patch("app.agents.nodes.retriever.rerank_with_indices")
def test_retriever_keeps_none_score_fallback(
    mock_rerank,
    mock_search,
):
    mock_search.return_value = [
        {"content": "fallback document", "source": "fallback.pdf"},
    ]

    mock_rerank.return_value = [
        {
            "index": 0,
            "document": "fallback document",
            "score": None,
        }
    ]

    result = retrieve_node(make_state())

    assert len(result["documents"]) == 1
    assert "fallback document" in result["documents"][0]
    assert result["sources"] == ["fallback.pdf"]
    assert result["status"] == "Found technical context."


def test_responder_skips_llm_when_no_context():
    state = {
        "current_query": "What is Qdrant?",
        "messages": [
            {
                "role": "user",
                "content": "What is Qdrant?",
            }
        ],
        "documents": [],
        "sources": [],
        "plan": [
            "Intent: Technical",
            "No Relevant Context",
        ],
        "status": "No relevant context found.",
    }

    with patch("app.agents.nodes.responder._generate_response") as mock_generate:
        result = generate_node(state)

    mock_generate.assert_not_called()

    assert (
        result["final_answer"] == "I couldn't find relevant information in the enterprise "
        "knowledge base to answer that question."
    )
    assert result["status"] == "No relevant context found."
    assert result["plan"] == state["plan"]
