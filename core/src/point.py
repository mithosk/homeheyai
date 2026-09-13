from dataclasses import dataclass


@dataclass
class Point:
    chunk: str
    vector: list[float]
