"""
Test script for the RAG pipeline
Refactored version with minimal duplication
"""

import os
import logging
from typing import List, Dict

from utils.qdrant import get_collection_info, search_similar, get_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_qdrant(query: str, collection_name: str, top_k: int = 5) -> List[Dict]:
    """Search for similar documents in Qdrant."""
    try:
        model = get_embedding_model()
        
        # Generate query embedding
        query_embedding = model.encode([query])[0].tolist()
        
        # Use the utility function for searching
        return search_similar(collection_name, query_embedding, top_k)
        
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
        print(f"Company ID: {result['metadata'].get('company_id', 'Unknown')}")
        print(f"Text: {result['text']}")





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
        

    ]
    
    for query, collection, display_func in test_cases:
        search_and_display(query, collection, display_func, top_k=3)


def check_collections_status():
    """Check the status of Qdrant collections."""
    try:
        collections = ["corporate_data"]
        
        print("\n📊 Qdrant Collections Status:")
        print("="*40)
        
        for collection_name in collections:
            info = get_collection_info(collection_name)
            if info:
                print(f"{collection_name}: {info['count']} documents")
            else:
                print(f"{collection_name}: Not found or error")
                
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
