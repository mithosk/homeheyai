import time
from chunker import Chunker
from ai_client import AIClient

MAX_RETRIES = 3


class Agent:
    def __init__(
        self, name: str, instruction: str, ai_client: AIClient, chunker: Chunker
    ):
        self._name = name.strip().replace(" ", "_").lower()
        self._instruction = instruction
        self._ai_client = ai_client
        self._chunker = chunker

    def load(self):
        self._chunker.refresh_chunks(
            directory_path=f"./knowledge/{self._name}", collection_name=self._name
        )

    def respond(self, prompt: str) -> str:
        retry = 0

        while True:
            try:
                rag_text = self._chunker.generate_text(
                    prompt=prompt, collection_name=self._name
                )

                return self._ai_client.generate(
                    prompt=f"{prompt}\n\nBEGIN_RAG\n{rag_text}\nEND_RAG",
                    instruction=self._instruction,
                )
            except Exception as exception:
                if retry == MAX_RETRIES:
                    raise exception

                time.sleep(++retry)
