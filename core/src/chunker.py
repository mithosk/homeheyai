import re
import uuid
from pathlib import Path
from google.genai import Client
from qdrant_client import QdrantClient
from google.genai.types import EmbedContentConfig, Content, Part
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, PointStruct, VectorParams

BATCH_SIZE = 100
CHUNK_OVERLAP = 150
CHUNK_SIZE = 1000
COLLECTION_NAME = "knowledge"
EMBEDDING_MODEL = "gemini-embedding-2"
MIN_CHUNK_LEN = 30
QUERY_POINTS_LIMIT = 5
SCORE_THRESHOLD = 0.45
VECTOR_SIZE = 3072


# 00006----------> SAB

def clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def save_chunk_batch(chunk_batch: list[dict], qdrant_client: QdrantClient, gemini_client: Client):
    contents = [
        Content(
            parts=[
                Part.from_text(
                    text=f"{chunk["pattern"]}\n{chunk["value"]}"
                )
            ]
        )
        for chunk in chunk_batch
    ]

    embed_response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents,
        config=EmbedContentConfig(
            output_dimensionality=VECTOR_SIZE
        )
    )

    points = [
        PointStruct(
            id=uuid.uuid4(),
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

    chunk_batch: list[dict] = []

    for file_path in Path(directory_path).rglob("*.md"):
        file_text = file_path.read_text(encoding="utf-8")
        cleaned_file_text = clean_text(file_text)
        splitted_file_text = text_splitter.split_text(cleaned_file_text)

        for i, file_text_part in enumerate(splitted_file_text):
            stripped_file_text_part = file_text_part.strip()

            if len(stripped_file_text_part) >= MIN_CHUNK_LEN:
                chunk_batch.append({
                    "pattern": f"{file_path.relative_to(directory_path)}/{i}",
                    "value": stripped_file_text_part
                })

                if len(chunk_batch) == BATCH_SIZE:
                    save_chunk_batch(chunk_batch, qdrant_client, gemini_client)
                    chunk_batch.clear()

    if chunk_batch:
        save_chunk_batch(chunk_batch, qdrant_client, gemini_client)

    print(f"End of refreshing chunks from {directory_path}\n")


def get_chunks(
        query: str,
        qdrant_client: QdrantClient,
        gemini_client: Client,
        score_threshold: float = 0.35,  # Soglia calibrata per RETRIEVAL_QUERY + COSINE
) -> list[str]:
    cleaned_query = clean_text(query)

    embed_response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=cleaned_query,
        config=EmbedContentConfig(
            output_dimensionality=VECTOR_SIZE,
            task_type="RETRIEVAL_QUERY"  # Ottimizzato per query di ricerca
        ),
    )

    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=[float(value) for value in (embed_response.embeddings[0].values or [])],  # type: ignore[arg-type]
        limit=QUERY_POINTS_LIMIT,
        score_threshold=SCORE_THRESHOLD,
        with_payload=True,
    )

    result: list[str] = []

    for point in response.points:
        print(f"DEBUG - Match trovato | Score: {point.score:.4f} | ID: {point.id}")
        if point.payload and isinstance(point.payload.get("value"), str):
            result.append(point.payload["value"])

    return result

# 00005----------> VEN
