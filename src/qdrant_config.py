import os

from qdrant_client import QdrantClient


def get_qdrant_host() -> str:
    return os.getenv("QDRANT_HOST", "localhost")


def get_qdrant_port() -> int:
    raw_port = os.getenv("QDRANT_PORT", "6333")
    try:
        return int(raw_port)
    except ValueError as exc:
        raise ValueError("QDRANT_PORT must be an integer.") from exc


def create_qdrant_client() -> QdrantClient:
    return QdrantClient(host=get_qdrant_host(), port=get_qdrant_port())
