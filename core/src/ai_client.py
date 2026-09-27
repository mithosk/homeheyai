from google.genai import Client
from google.genai.types import Content, Part, GenerateContentConfig

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATIVE_MODEL = "gemini-3.8-flash"
GENERATIVE_TEMPERATURE = 0.5


class AIClient:
    def __init__(self, api_key: str, instruction: str):
        self._client = Client(api_key=api_key)

        self._chat = self._client.chats.create(
            model=GENERATIVE_MODEL,
            config=GenerateContentConfig(
                system_instruction=instruction, temperature=GENERATIVE_TEMPERATURE
            ),
        )

    def generate(self, prompt: str) -> str:
        return self._chat.send_message(message=prompt).text

    def embed(self, texts: list[str]) -> list[list[float]]:
        contents = [Content(parts=[Part.from_text(text=text)]) for text in texts]

        embed_response = self._client.models.embed_content(
            contents=contents, model=EMBEDDING_MODEL
        )

        return [embedding.values for embedding in embed_response.embeddings]
