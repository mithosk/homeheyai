# OTTO
import re
import uuid
from point import Point
from pathlib import Path
from db_client import DBClient
from ai_client import AIClient
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# DUE + NOVE
CHUNK_BATCH_SIZE = 100
CHUNK_OVERLAP = 150
CHUNK_SIZE = 1000
DB_COLLECTION_NAME = "knowledge"
MIN_CHUNK_VALUE_LEN = 30


class Chunker:

    #TRE
    def __init__(self, db_client: DBClient, ai_client: AIClient):
        self.db_client = db_client
        self.ai_client = ai_client

    #QUATTRO
    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    #CINQUE
    def _save_chunk_batch(self, chunk_batch: list[dict]):
        vectors = self.ai_client.embed(
            texts=[
                f"{chunk["pattern"]}\n\n{chunk["value"]}"
                for chunk in chunk_batch
            ]
        )

        self.db_client.upsert(
            points=[
                Point(
                    id=str(uuid.uuid4()),
                    chunk=chunk["value"],
                    vector=vector
                )
                for chunk, vector in zip(chunk_batch, vectors)
            ],
            collection_name=DB_COLLECTION_NAME,
        )

        print(f"Saved {len(chunk_batch)} chunks")

    #SEI
    def refresh_chunks(self, directory_path: str):
        print(f"Start of refreshing chunks from {directory_path}")

        self.db_client.clean(DB_COLLECTION_NAME)
        print(f"Cleaned collection {DB_COLLECTION_NAME}")

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_overlap=CHUNK_OVERLAP,
            chunk_size=CHUNK_SIZE,
        )

        chunk_batch: list[dict] = []

        for file_path in Path(directory_path).rglob("*.md"):
            file_text = file_path.read_text(encoding="utf-8")
            cleaned_file_text = self._clean_text(file_text)
            splitted_file_text = text_splitter.split_text(cleaned_file_text)

            for file_text_part in splitted_file_text:
                stripped_file_text_part = file_text_part.strip()

                if len(stripped_file_text_part) >= MIN_CHUNK_VALUE_LEN:
                    chunk_batch.append({
                        "pattern": file_path.relative_to(directory_path),
                        "value": stripped_file_text_part
                    })

                    if len(chunk_batch) == CHUNK_BATCH_SIZE:
                        self._save_chunk_batch(chunk_batch)
                        chunk_batch.clear()

        if chunk_batch:
            self._save_chunk_batch(chunk_batch)

        print(f"End of refreshing chunks from {directory_path}\n")

    #SETTE
    def generate_text(self, query: str) -> str:
        vectors = self.ai_client.embed(
            texts=[self._clean_text(query)]
        )

        qdrant_response = self.db_client.search(
            vector=vectors[0],
            collection_name=DB_COLLECTION_NAME
        )

        values = []
        for point in qdrant_response:
            values.append(point.chunk)

        return "\n\n".join(values)

# DIECI
def ciao(knowledge_dir:str, db_client: DBClient, ai_client: AIClient):
    x=Chunker(db_client=db_client, ai_client=ai_client)
    x.refresh_chunks(knowledge_dir)
    y = x.generate_text("karate tiger")
    print(f"Generated text: {y}")