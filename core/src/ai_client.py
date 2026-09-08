from google.genai import Client
from google.genai.types import Content, Part

EMBEDDING_MODEL = "gemini-embedding-2"


class AIClient:
    def __init__(self, api_key: str):
        self._client = Client(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        contents = [Content(parts=[Part.from_text(text=text)]) for text in texts]

        embed_response = self._client.models.embed_content(
            contents=contents, model=EMBEDDING_MODEL
        )

        return [embedding.values for embedding in embed_response.embeddings]
