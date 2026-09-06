from google.genai import Client

EMBEDDING_MODEL = "gemini-embedding-2"


class AIClient:
    def __init__(self, api_key: str):
        self._client = Client(api_key=api_key)

    def embed(self, contents: list[str]) -> list[list[float]]:
        embed_response = self._client.models.embed_content(
            contents=contents, model=EMBEDDING_MODEL
        )

        return [embedding.values for embedding in embed_response.embeddings]
