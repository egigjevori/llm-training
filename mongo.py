"""
Simple MongoDB connection using environment variables.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def get_mongodb_connection() -> Optional[MongoClient]:
    """
    Create a MongoDB connection using environment variables.
    
    Environment variables (with Kubernetes defaults):
    - MONGO_HOST: MongoDB host (default: localhost)
    - MONGO_PORT: MongoDB port (default: 27017)
    - MONGO_USERNAME: MongoDB username (default: admin)
    - MONGO_PASSWORD: MongoDB password (required)
    - MONGO_DATABASE: MongoDB database name (default: admin)
    
    Returns:
        MongoClient: MongoDB client instance or None if connection fails
        
    Raises:
        ConnectionError: If unable to connect to MongoDB
    """
    try:
        # Get environment variables with Kubernetes-style defaults
        host = os.getenv("MONGO_HOST", "localhost")
        port = int(os.getenv("MONGO_PORT", "27017"))
        username = os.getenv("MONGO_USERNAME", "admin")
        password = os.getenv("MONGO_PASSWORD")
        database = os.getenv("MONGO_DATABASE", "admin")
        
        if not password:
            raise ValueError("MONGO_PASSWORD environment variable is required")
        
        # Create connection string
        connection_string = f"mongodb://{username}:{password}@{host}:{port}/{database}"
        
        # Create MongoDB client
        client = MongoClient(connection_string)
        
        # Test the connection
        client.admin.command('ping')
        
        logger.info(f"Successfully connected to MongoDB at {host}:{port}")
        return client
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise ConnectionError(f"MongoDB connection failed: {e}")


def get_database(client: MongoClient, db_name: Optional[str] = None) -> Database:
    """
    Get a database instance from MongoDB client.
    
    Args:
        client: MongoDB client instance
        db_name: Database name (uses MONGO_DATABASE env var if not provided)
        
    Returns:
        Database: MongoDB database instance
    """
    if db_name is None:
        db_name = os.getenv("MONGO_DATABASE", "admin")
    
    return client[db_name]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test the connection
        mongo_client = get_mongodb_connection()
        db = get_database(mongo_client)
        
        # Example usage
        print(f"Connected to database: {db.name}")
        print(f"Collections: {db.list_collection_names()}")
        
        # Close connection
        mongo_client.close()
        
    except Exception as e:
        print(f"Error: {e}")
