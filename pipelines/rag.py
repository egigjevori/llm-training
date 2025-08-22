"""
Simple ZenML RAG Pipeline for MongoDB to Qdrant
Refactored version with minimal duplication
"""

import logging
import json
import os
from typing import List, Dict
from uuid import uuid4

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zenml import pipeline, step
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from mongo import get_mongodb_connection, get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client with environment configuration."""
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=qdrant_host, port=qdrant_port)


def get_embedding_model() -> SentenceTransformer:
    """Get embedding model."""
    return SentenceTransformer('all-MiniLM-L6-v2')


@step
def initialize_qdrant() -> bool:
    """Initialize Qdrant collections for corporate data."""
    try:
        client = get_qdrant_client()
        collections = ["corporate_data"]
        
        for collection_name in collections:
            try:
                # Delete existing collection if it exists
                client.get_collection(collection_name)
                client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except:
                logger.info(f"Collection {collection_name} does not exist")
            
            # Create new collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {collection_name}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing Qdrant: {e}")
        return False


def fetch_batch_from_mongo(db_name: str, collection_name: str, batch_size: int = 100, skip: int = 0) -> List[Dict]:
    """Generic function to fetch a batch of documents from any MongoDB collection."""
    try:
        client = get_mongodb_connection()
        db = get_database(client, db_name)
        collection = db[collection_name]
        
        documents = list(collection.find().skip(skip).limit(batch_size))
        
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
        
        client.close()
        logger.info(f"Fetched {len(documents)} documents from {db_name}.{collection_name} (skip: {skip})")
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching batch from {db_name}.{collection_name}: {e}")
        return []


def create_corporate_chunk(doc: Dict) -> Dict:
    """Create a text chunk from corporate document using JSON directly."""
    return {
        'id': str(uuid4()),
        'text': json.dumps(doc, ensure_ascii=False, separators=(',', ':')),
        'metadata': {
            'source': 'corporate',
            'company_id': doc.get('company_id', 'unknown'),
            'company_name': doc.get('name', 'unknown'),
            'original_doc': doc
        }
    }


def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    """Generate embeddings for text chunks."""
    if not chunks:
        return []
        
    try:
        model = get_embedding_model()
        texts = [chunk['text'] for chunk in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []


def store_in_qdrant(chunks: List[Dict], collection_name: str) -> bool:
    """Store embedded chunks in Qdrant."""
    if not chunks:
        return True
        
    try:
        client = get_qdrant_client()
        
        points = [
            PointStruct(
                id=chunk['id'],
                vector=chunk['embedding'],
                payload={
                    'text': chunk['text'],
                    'metadata': chunk['metadata']
                }
            ) for chunk in chunks
        ]
        
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Stored {len(points)} points in Qdrant collection: {collection_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing in Qdrant: {e}")
        return False


@step
def process_corporate_data(batch_size: int = 100) -> bool:
    """Process corporate data in batches."""
    skip = 0
    total_processed = 0
    
    while True:
        # Fetch batch
        documents = fetch_batch_from_mongo("opencorporates_albania", "companies", batch_size, skip)
        
        if not documents:
            break
        
        # Process batch: chunk -> embed -> store
        chunks = [create_corporate_chunk(doc) for doc in documents]
        embedded_chunks = generate_embeddings(chunks)
        success = store_in_qdrant(embedded_chunks, "corporate_data")
        
        if success:
            total_processed += len(documents)
            logger.info(f"Processed corporate_data batch. Total so far: {total_processed}")
            
        skip += batch_size
    
    logger.info(f"Completed processing {total_processed} corporate documents")
    return True





@pipeline
def rag_pipeline():
    """Simple RAG pipeline: MongoDB -> Chunking -> Embedding -> Qdrant"""
    # Initialize Qdrant
    initialize_qdrant()
    
    # Process corporate data
    process_corporate_data()


if __name__ == "__main__":
    rag_pipeline()
