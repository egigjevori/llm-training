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

import torch
from zenml import pipeline, step
from qdrant_client.models import PointStruct
from utils.mongo import get_mongodb_connection, get_database
from utils.qdrant import create_collection, upsert_points, get_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step(enable_cache=False)
def initialize_qdrant() -> None:
    """Initialize Qdrant collections for corporate data."""
    try:
        collections = ["corporate_data"]
        
        for collection_name in collections:
            create_collection(collection_name, vector_size=1024, recreate=True)

    except Exception as e:
        logger.error(f"Error initializing Qdrant: {e}")
        raise


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
        
        logger.info(f"Fetched {len(documents)} documents from {db_name}.{collection_name} (skip: {skip})")
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching batch from {db_name}.{collection_name}: {e}")
        raise


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
        raise


def store_in_qdrant(chunks: List[Dict], collection_name: str) -> None:
    """Store embedded chunks in Qdrant."""
    try:
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
        
        upsert_points(collection_name, points)

    except Exception as e:
        logger.error(f"Error storing in Qdrant: {e}")
        raise


@step(enable_cache=False)
def process_corporate_data(batch_size: int = 2, max_documents: int = 100) -> None:
    """Process corporate data in batches with document limit."""
    skip = 0
    total_processed = 0
    
    while total_processed < max_documents:
        # Calculate remaining documents to process
        remaining = max_documents - total_processed
        current_batch_size = min(batch_size, remaining)
        
        # Fetch batch
        documents = fetch_batch_from_mongo("opencorporates_albania", "companies", current_batch_size, skip)
        
        if not documents:
            break
        
        # Process batch: chunk -> embed -> store
        chunks = [create_corporate_chunk(doc) for doc in documents]
        embedded_chunks = generate_embeddings(chunks)
        store_in_qdrant(embedded_chunks, "corporate_data")
        
        # Clear GPU cache to free memory
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        total_processed += len(documents)
        logger.info(f"Processed corporate_data batch. Total so far: {total_processed}")

        skip += len(documents)
        
        if total_processed >= max_documents:
            logger.info(f"Reached maximum limit of {max_documents} documents. Stopping processing.")
            break
    
    logger.info(f"Completed processing {total_processed} corporate documents")





@pipeline
def rag_pipeline():
    """Simple RAG pipeline: MongoDB -> Chunking -> Embedding -> Qdrant"""
    # Initialize Qdrant
    initialize_qdrant()
    
    # Process corporate data
    process_corporate_data()


if __name__ == "__main__":
    rag_pipeline()
