"""
Simple Qdrant connection utilities
Provides reusable functions for Qdrant operations across the project
"""

import os
import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """Get embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        # Using smaller, faster model optimized for retrieval tasks
        _embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client with environment configuration."""
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=qdrant_host, port=qdrant_port)


def create_collection(collection_name: str, vector_size: int = 384,
                     distance: Distance = Distance.COSINE, recreate: bool = True) -> None:
    """Create a collection in Qdrant."""
    try:
        client = get_qdrant_client()
        
        if recreate:
            try:
                client.get_collection(collection_name)
                client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except:
                logger.info(f"Collection {collection_name} does not exist")
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )
        logger.info(f"Created collection: {collection_name}")
        
    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {e}")
        raise


def get_collection_info(collection_name: str) -> Optional[Dict]:
    """Get information about a collection."""
    try:
        client = get_qdrant_client()
        client.get_collection(collection_name)
        count = client.count(collection_name)
        return {
            "name": collection_name,
            "count": count.count
        }
    except Exception as e:
        logger.warning(f"Collection {collection_name} not found or error: {e}")
        raise


def search_similar(collection_name: str, query_vector: List[float], top_k: int = 5) -> List[Dict]:
    """Search for similar documents in a collection."""
    try:
        client = get_qdrant_client()
        
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        ).points
        
        return [{
            'score': result.score,
            'text': result.payload['text'],
            'metadata': result.payload['metadata']
        } for result in search_results]
        
    except Exception as e:
        logger.error(f"Error searching collection {collection_name}: {e}")
        raise


def upsert_points(collection_name: str, points: List[PointStruct]) -> None:
    """Insert or update points in a collection."""
    try:
        client = get_qdrant_client()
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)} points to collection: {collection_name}")
    except Exception as e:
        logger.error(f"Error upserting points to {collection_name}: {e}")
        raise