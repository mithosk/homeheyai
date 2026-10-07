from chunker import Chunker
from ai_client import AIClient


class Agent:
    def __init__(
        self,
        name: str,
        instruction: str,
        ai_client: AIClient,
        chunker: Chunker,
        knowledge_dir: str,
    ) -> None:
        self._name = name
        self._instruction = instruction
        self._ai_client = ai_client
        self._chunker = chunker
        self._knowledge_dir = knowledge_dir

    @property
    def name(self) -> str:
        return self._name

    def load(self) -> None:
        self._chunker.refresh_chunks(
            directory_path=f"{self._knowledge_dir}", collection_name=self._name
        )

    def respond(self, prompt: str) -> str:
        rag_text = self._chunker.generate_text(
            prompt=prompt, collection_name=self._name
        )

        return self._ai_client.generate(
            prompt=f"{prompt}\n\nBEGIN_RAG\n{rag_text}\nEND_RAG",
            instruction=self._instruction,
        )
