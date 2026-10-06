from google.genai import Client
from google.genai.types import (
    Content,
    Part,
    GenerateContentConfig,
    AutomaticFunctionCallingConfig,
)

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATIVE_MODEL = "gemini-3.5-flash-lite"
GENERATIVE_TEMPERATURE = 0.5


class AIClient:
    def __init__(self, api_key: str):
        self._client = Client(api_key=api_key)

    def generate(self, prompt: str, instruction: str) -> str:
        generate_response = self._client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=instruction,
                temperature=GENERATIVE_TEMPERATURE,
                automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
            ),
        )

        return generate_response.text

    def embed(self, texts: list[str]) -> list[list[float]]:
        contents = [Content(parts=[Part.from_text(text=text)]) for text in texts]

        embed_response = self._client.models.embed_content(
            model=EMBEDDING_MODEL, contents=contents
        )

        return [embedding.values for embedding in embed_response.embeddings]
