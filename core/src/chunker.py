import re
import uuid
from pathlib import Path
from google.genai import Client
from google.genai.types import EmbedContentConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

MIN_CHUNK_LEN = 30
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
VECTOR_SIZE = 3072
BATCH_SIZE = 100
COLLECTION_NAME = "knowledge"
EMBEDDING_MODEL = "gemini-embedding-2"


def extract_text(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return ""


def clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def save_chunks(chunk_batch: list[dict], qdrant_client: QdrantClient, gemini_client: Client):
    if not chunk_batch:
        return

    embed_response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[item["value"] for item in chunk_batch],
        config=EmbedContentConfig(
            output_dimensionality=VECTOR_SIZE,
            task_type="RETRIEVAL_DOCUMENT"
        ),
    )

    embeddings = embed_response.embeddings or []

    points = []
    for chunk, embedding in zip(chunk_batch, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),  # Generato distintamente ad ogni iterazione del ciclo for
                vector=[float(value) for value in (embedding.values or [])],
                payload={
                    "pattern": chunk["pattern"],
                    "value": chunk["value"],
                },
            )
        )

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Salvati {len(points)} chunk in Qdrant.")


def refresh_chunks(directory_path: str, qdrant_client: QdrantClient, gemini_client: Client):
    print(f"Start refreshing chunks from: {directory_path}")

    qdrant_client.delete_collection(COLLECTION_NAME)
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(f"Cleaned collection '{COLLECTION_NAME}'")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunk_batch: list[dict] = []
    total_chunks_created = 0

    for file_path in Path(directory_path).rglob("*.md"):
        file_text = extract_text(file_path)
        cleaned_file_text = clean_text(file_text)
        chunks = text_splitter.split_text(cleaned_file_text)

        rel_path = file_path.relative_to(directory_path)

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.strip()

            if len(chunk_text) >= MIN_CHUNK_LEN:
                chunk_batch.append({
                    "pattern": f"{rel_path}/{i}",
                    "value": chunk_text,
                })
                total_chunks_created += 1

            if len(chunk_batch) >= BATCH_SIZE:
                save_chunks(chunk_batch, qdrant_client, gemini_client)
                chunk_batch.clear()

    if chunk_batch:
        save_chunks(chunk_batch, qdrant_client, gemini_client)

    info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"End of refresh. Generati: {total_chunks_created} | Punti totali in Qdrant: {info.points_count}\n")


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
        query=[float(value) for value in (embed_response.embeddings[0].values or [])],
        limit=5,
        score_threshold=score_threshold,
        with_payload=True,
    )

    result: list[str] = []

    for point in response.points:
        print(f"DEBUG - Match trovato | Score: {point.score:.4f} | ID: {point.id}")
        if point.payload and isinstance(point.payload.get("value"), str):
            result.append(point.payload["value"])

    return result