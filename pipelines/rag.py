"""
UK Companies House ZenML RAG Pipeline
MongoDB -> Chunking -> Embedding (bge-small-en-v1.5) -> Qdrant
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
    """Initialize Qdrant collections for UK companies data."""
    try:
        collections = ["uk_companies"]

        for collection_name in collections:
            create_collection(collection_name, vector_size=384, recreate=True)

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


def create_company_chunk(doc: Dict) -> Dict:
    """Create a text chunk from UK company document using JSON directly."""
    return {
        'id': str(uuid4()),
        'text': json.dumps(doc, ensure_ascii=False, separators=(',', ':')),
        'metadata': {
            'source': 'uk_companies_house',
            'company_number': doc.get('company_number', 'unknown'),
            'company_name': doc.get('company_name', 'unknown'),
            'status': doc.get('status', 'unknown'),
            'category': doc.get('category', ''),
            'country': doc.get('country_of_origin', 'UK'),
            'sic_codes': doc.get('sic_codes', []),
            'full_address': doc.get('address', {}).get('full_address', ''),
            'incorporation_date': doc.get('incorporation_date', ''),
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
def process_uk_companies_data(batch_size: int = 10, max_documents: int = 300) -> None:
    """Process UK companies data in batches with document limit."""
    skip = 0
    total_processed = 0

    while total_processed < max_documents:
        # Calculate remaining documents to process
        remaining = max_documents - total_processed
        current_batch_size = min(batch_size, remaining)

        # Fetch batch from UK companies database
        documents = fetch_batch_from_mongo("uk_companies_house", "companies", current_batch_size, skip)

        if not documents:
            logger.info("No more documents found in UK companies database")
            break

        # Process batch: chunk -> embed -> store
        chunks = [create_company_chunk(doc) for doc in documents]
        embedded_chunks = generate_embeddings(chunks)
        store_in_qdrant(embedded_chunks, "uk_companies")

        # Clear GPU cache to free memory
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        total_processed += len(documents)
        logger.info(f"Processed UK companies batch. Total so far: {total_processed}")

        skip += len(documents)

        if total_processed >= max_documents:
            logger.info(f"Reached maximum limit of {max_documents} documents. Stopping processing.")
            break

    logger.info(f"Completed processing {total_processed} UK company documents")





@pipeline
def rag_pipeline():
    """UK Companies House RAG pipeline: MongoDB -> Chunking -> Embedding -> Qdrant"""
    # Initialize Qdrant
    initialize_qdrant()

    # Process UK companies data
    process_uk_companies_data()


if __name__ == "__main__":
    rag_pipeline()
