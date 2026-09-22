import logfire
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI
from nemoguardrails import LLMRails, RailsConfig

from app.config import apply_langchain_env, settings
from app.guardrails.colang_rules import COLANG_CONTENT, RAIL_INDICATORS, YAML_CONTENT

_rails: LLMRails | None = None


class GeminiGuardrailLLM(BaseChatModel):
    """Compatibility adapter for Gemini content blocks and NeMo Guardrails."""

    model: str = "gemini-3.5-flash-lite"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(
            self,
            "_llm",
            ChatGoogleGenerativeAI(
                google_api_key=settings.GEMINI_API_KEY,
                model=self.model,
            ),
        )

    @property
    def _llm_type(self) -> str:
        return "gemini_guardrail"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        response = self._llm.invoke(messages, stop=stop, **kwargs)
        content = response.content

        if isinstance(content, list):
            text = "".join(
                block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = str(content or "")

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses Gemini for intent classification at the gate.
    """
    global _rails

    apply_langchain_env()

    guard_llm = GeminiGuardrailLLM()

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT,
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("NeMo Guardrails initialised (Gemini).")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True, rail_response) - a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None) - message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("Guardrails not initialised - skipping gate.")
        return False, None

    with logfire.span("Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("Guardrails passed.")
        return False, None
