from types import SimpleNamespace

from google import genai

from app.config import settings

GEMINI_MODEL = "gemini-3.5-flash-lite"

gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)


class GeminiLLM:
    """Gemini adapter with the .invoke(prompt).content interface."""

    def invoke(self, prompt: str):
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        return SimpleNamespace(content=response.text)


def get_langchain_llm(feature: str = "rag"):
    """Return a Gemini-backed LLM for the requested feature."""
    return GeminiLLM()
