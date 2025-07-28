"""
Test script for the RAG pipeline
Refactored version with minimal duplication
"""

import os
import logging
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client with environment configuration."""
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=qdrant_host, port=qdrant_port)


def search_qdrant(query: str, collection_name: str, top_k: int = 5) -> List[Dict]:
    """Search for similar documents in Qdrant."""
    try:
        client = get_qdrant_client()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate query embedding
        query_embedding = model.encode([query])[0].tolist()
        
        # Search in Qdrant
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        # Format results
        return [{
            'score': result.score,
            'text': result.payload['text'],
            'metadata': result.payload['metadata']
        } for result in search_results]
        
    except Exception as e:
        logger.error(f"Error searching Qdrant: {e}")
        return []


def display_corporate_results(results: List[Dict], query: str):
    """Display corporate search results."""
    print(f"\n{'='*60}")
    print(f"SEARCHING CORPORATE DATA: {query}")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result['score']:.3f}) ---")
        print(f"Company: {result['metadata'].get('company_name', 'Unknown')}")
        print(f"NIPT: {result['metadata'].get('nipt', 'Unknown')}")
        print(f"Text: {result['text'][:200]}...")


def display_procurement_results(results: List[Dict], query: str):
    """Display procurement search results."""
    print(f"\n{'='*60}")
    print(f"SEARCHING PROCUREMENT DATA: {query}")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result['score']:.3f}) ---")
        print(f"Tender ID: {result['metadata'].get('tender_id', 'Unknown')}")
        print(f"Text: {result['text'][:300]}...")


def search_and_display(query: str, collection_name: str, display_func, top_k: int = 5):
    """Generic search and display function."""
    results = search_qdrant(query, collection_name, top_k)
    display_func(results, query)


def test_rag_queries():
    """Test various RAG queries."""
    print("🚀 Testing RAG Pipeline Queries")
    print("="*60)
    
    # Test data - (query, collection, display_function)
    test_cases = [
        # Corporate queries
        ("software development company", "corporate_data", display_corporate_results),
        ("construction company in Tirana", "corporate_data", display_corporate_results),
        ("consulting services", "corporate_data", display_corporate_results),
        ("restaurant or food business", "corporate_data", display_corporate_results),
        
        # Procurement queries
        ("road construction tender", "procurement_data", display_procurement_results),
        ("medical equipment procurement", "procurement_data", display_procurement_results),
        ("IT services tender", "procurement_data", display_procurement_results),
        ("cleaning services contract", "procurement_data", display_procurement_results)
    ]
    
    for query, collection, display_func in test_cases:
        search_and_display(query, collection, display_func, top_k=3)


def check_collections_status():
    """Check the status of Qdrant collections."""
    try:
        client = get_qdrant_client()
        collections = ["corporate_data", "procurement_data"]
        
        print("\n📊 Qdrant Collections Status:")
        print("="*40)
        
        for collection_name in collections:
            try:
                client.get_collection(collection_name)
                count = client.count(collection_name)
                print(f"{collection_name}: {count.count} documents")
            except Exception as e:
                print(f"{collection_name}: Not found or error - {e}")
                
    except Exception as e:
        print(f"Error checking collections: {e}")


if __name__ == "__main__":
    print("🧪 RAG Pipeline Test Script")
    
    # Check collection status first
    check_collections_status()
    
    # Run test queries
    test_rag_queries()
    
    print(f"\n{'='*60}")
    print("✅ RAG Testing Complete!")
    print("="*60) 