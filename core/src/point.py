from dataclasses import dataclass


@dataclass
class Point:
    id: str
    chunk: str
    vector: list[float]
