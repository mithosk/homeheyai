from chunker import Chunker
from ai_client import AIClient


class Agent:
    def __init__(
        self, name: str, instruction: str, ai_client: AIClient, chunker: Chunker
    ):
        self._name = name.strip().replace(" ", "_").lower()
        self._instruction = instruction
        self._ai_client = ai_client
        self._chunker = chunker

        self._chunker.refresh_chunks(
            directory_path=f"./knowledge/{self._name}", collection_name=self._name
        )

        print(f"Agent {self._name} initialized")

    def respond(self, prompt: str) -> str:
        rag_text = self._chunker.generate_text(
            prompt=prompt, collection_name=self._name
        )

        return self._ai_client.generate(
            prompt=f"{prompt}\n\nRAG_BEGIN\n{rag_text}\nRAG_END",
            instruction=self._instruction,
        )
