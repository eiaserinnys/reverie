"""Vector serialization utilities for SQLite backend.

sqlite-vec requires vectors to be stored as BLOBs containing packed floats.
This module provides utilities to serialize/deserialize Python lists to/from
the binary format expected by sqlite-vec.
"""

import struct
from typing import Optional


def serialize_vector(vector: list[float] | None) -> bytes | None:
    """Serialize a float vector to bytes for SQLite storage.

    Uses struct.pack with 'f' format (32-bit float) which is the format
    expected by sqlite-vec's vec_distance_* functions.

    Args:
        vector: List of floats, or None.

    Returns:
        Packed bytes, or None if input is None.

    Example:
        >>> v = [0.1, 0.2, 0.3]
        >>> b = serialize_vector(v)
        >>> len(b)
        12  # 3 floats * 4 bytes each
    """
    if vector is None:
        return None
    return struct.pack(f'{len(vector)}f', *vector)


def deserialize_vector(data: bytes | None, dimensions: int = 1536) -> list[float] | None:
    """Deserialize bytes from SQLite to a float vector.

    Args:
        data: Packed bytes from SQLite BLOB, or None.
        dimensions: Expected vector dimensions (default 1536 for OpenAI embeddings).

    Returns:
        List of floats, or None if input is None.

    Raises:
        struct.error: If data length doesn't match dimensions.
    """
    if data is None:
        return None

    # Calculate expected dimensions from data if not matching expected
    actual_dimensions = len(data) // 4  # 4 bytes per float
    return list(struct.unpack(f'{actual_dimensions}f', data))


def vector_to_sqlite_vec_format(vector: list[float] | None) -> bytes | None:
    """Convert vector to sqlite-vec query format.

    For sqlite-vec queries, vectors need to be passed as packed bytes.
    This is the same format as storage.

    Args:
        vector: Query vector.

    Returns:
        Packed bytes for use in vec_distance_* functions.
    """
    return serialize_vector(vector)


def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    This is a fallback for when we need to compute similarity in Python
    instead of using sqlite-vec.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity in range [-1, 1].

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def cosine_distance_to_similarity(distance: float) -> float:
    """Convert cosine distance to similarity.

    sqlite-vec returns cosine distance (1 - similarity).
    We need to convert it to similarity for compatibility with pgvector results.

    Note: pgvector <=> operator returns cosine distance.
    sqlite-vec vec_distance_cosine also returns distance.

    Args:
        distance: Cosine distance from sqlite-vec.

    Returns:
        Cosine similarity in range [0, 1].
    """
    return 1.0 - distance
