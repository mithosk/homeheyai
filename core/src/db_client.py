from point import Point
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

DB_VECTOR_SIZE = 3072
QUERY_POINTS_LIMIT = 5


class DBClient:
    def __init__(self, host: str, port: int):
        self._client = QdrantClient(host=host, port=port)

    def upsert(self, points: list[Point], collection_name: str):
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=DB_VECTOR_SIZE, distance=Distance.COSINE
                ),
            )

        self._client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload={
                        "chunk": point.chunk,
                    },
                )
                for point in points
            ],
        )

    def search(self, vector: list[float], collection_name: str) -> list[Point]:
        query_result = self._client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=QUERY_POINTS_LIMIT,
            with_payload=True,
        )

        return [
            Point(id=item.id, chunk=item.payload["chunk"], vector=item.vector)
            for item in query_result.points
        ]

    def clean(self, collection_name: str):
        self._client.delete_collection(collection_name)
