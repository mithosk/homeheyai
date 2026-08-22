import re
import uuid
from pathlib import Path
from google.genai import Client
from qdrant_client import QdrantClient
from google.genai.types import EmbedContentConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, PointStruct, VectorParams

MIN_CHUNK_LEN = 30
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
VECTOR_SIZE = 768
BATCH_SIZE = 100
COLLECTION_NAME = "knowledge"
EMBEDDING_MODEL = "gemini-embedding-2"


def extract_text(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return ""


def clean_text(text: str) -> str:
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def save_chunks(chunk_batch: list, qdrant_client: QdrantClient, gemini_client: Client):
    embed_response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[item["value"] for item in chunk_batch],
        config=EmbedContentConfig(output_dimensionality=VECTOR_SIZE)
    )

    points = [
        PointStruct(
            id=str(chunk["id"]),
            vector=[float(value) for value in (embedding.values or [])],
            payload={
                "pattern": chunk["pattern"],
                "value": chunk["value"]
            }
        )
        for chunk, embedding in zip(chunk_batch, embed_response.embeddings or [])
    ]

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Saved {len(chunk_batch)} chunks")


def refresh_chunks(directory_path: str, qdrant_client: QdrantClient, gemini_client: Client):
    print(f"Start of refreshing chunks from {directory_path}")

    qdrant_client.delete_collection(COLLECTION_NAME)
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    print(f"Cleaned collection {COLLECTION_NAME}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunk_batch = []

    for file_path in Path(directory_path).rglob("*.md"):
        file_text = extract_text(file_path)
        cleaned_file_text = clean_text(file_text)
        chunks = text_splitter.split_text(cleaned_file_text)

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.strip()

            if len(chunk_text) >= MIN_CHUNK_LEN:
                chunk_batch.append({
                    "id": uuid.uuid4(),
                    "pattern": f"{file_path.relative_to(directory_path).with_suffix("")}/{i}",
                    "value": chunk_text
                })

            if len(chunk_batch) == BATCH_SIZE:
                save_chunks(chunk_batch, qdrant_client, gemini_client)
                chunk_batch.clear()

    if len(chunk_batch) > 0:
        save_chunks(chunk_batch, qdrant_client, gemini_client)

    print(f"End of refreshing chunks from {directory_path}\n")
