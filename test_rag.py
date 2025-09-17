"""
Test script for the UK Companies House RAG pipeline
Tests the updated pipeline with bge-small-en-v1.5 embeddings
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


def display_uk_companies_results(results: List[Dict], query: str):
    """Display UK companies search results."""
    print(f"\n{'='*60}")
    print(f"SEARCHING UK COMPANIES: {query}")
    print(f"{'='*60}")

    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        print(f"\n--- Result {i} (Score: {result['score']:.3f}) ---")
        print(f"Company: {metadata.get('company_name', 'Unknown')}")
        print(f"Company Number: {metadata.get('company_number', 'Unknown')}")
        print(f"Status: {metadata.get('status', 'Unknown')}")
        print(f"Category: {metadata.get('category', 'Unknown')}")
        print(f"Address: {metadata.get('full_address', 'Unknown')}")
        print(f"SIC Codes: {metadata.get('sic_codes', [])}")
        print(f"Incorporation Date: {metadata.get('incorporation_date', 'Unknown')}")
        print(f"Text: {result['text'][:200]}...")  # Limit text display





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
        # UK Companies queries
        ("software development company", "uk_companies", display_uk_companies_results),
        ("construction company in London", "uk_companies", display_uk_companies_results),
        ("consulting services", "uk_companies", display_uk_companies_results),
        ("restaurant or food business", "uk_companies", display_uk_companies_results),
        ("private limited company", "uk_companies", display_uk_companies_results),

    ]
    
    for query, collection, display_func in test_cases:
        search_and_display(query, collection, display_func, top_k=3)


def check_collections_status():
    """Check the status of Qdrant collections."""
    try:
        collections = ["uk_companies"]

        print("\n📊 Qdrant Collections Status:")
        print("="*40)

        for collection_name in collections:
            try:
                info = get_collection_info(collection_name)
                print(f"{collection_name}: {info['count']} documents")
            except Exception as e:
                print(f"{collection_name}: Not found or error ({e})")

    except Exception as e:
        print(f"Error checking collections: {e}")


if __name__ == "__main__":
    print("🧪 UK Companies House RAG Pipeline Test Script")

    # Check collection status first
    check_collections_status()

    # Run test queries
    test_rag_queries()

    print(f"\n{'='*60}")
    print("✅ UK Companies RAG Testing Complete!")
    print("="*60)
