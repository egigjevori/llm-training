# Albania Business Data RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for processing and searching Albania business data, including public procurement tenders and company information.

## 🏗️ Architecture

The system consists of two main components:

1. **Data Ingestion Pipelines**: Scrape data from OpenProcurement Albania and OpenCorporates Albania
2. **RAG Pipeline**: Process, clean, chunk, and embed data for vector search in Qdrant

## 📊 Data Sources

- **OpenProcurement Albania**: Public procurement tenders and contracts
- **OpenCorporates Albania**: Company registration and business information

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- MongoDB (provided via Docker)
- Qdrant Vector Database (provided via Docker)

### Option 1: Using Docker Compose (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llm-training
   ```

2. **Start the services**:
   ```bash
   docker-compose up -d mongodb qdrant
   ```

3. **Run the data ingestion pipelines**:
   ```bash
   # Run OpenProcurement pipeline
   python pipelines/open_procurement.py
   
   # Run OpenCorporates pipeline
   python pipelines/open_corporate.py
   ```

4. **Run the RAG pipeline**:
   ```bash
   python pipelines/rag.py
   ```
   
   *Note: The pipeline is configured to process 1000 documents per collection for quick testing. To process all data, modify the `max_documents` parameter in `rag_pipeline()` or set it to `None`.*

5. **Test the RAG system**:
   ```bash
   python test_rag.py
   ```

## 🔧 RAG Pipeline Details

The RAG pipeline (`pipelines/rag.py`) is designed to be simple and efficient:

### Features
- **Batch Processing**: Processes MongoDB data in configurable batches (default: 100 documents)
- **Automatic Chunking**: Converts JSON documents to searchable text chunks
- **Embeddings**: Uses `all-MiniLM-L6-v2` for fast, efficient embeddings
- **Vector Storage**: Stores embeddings in Qdrant with metadata preservation
- **Dual Collections**: Separate collections for corporate and procurement data

### Data Flow
1. **Fetch**: Pull data from MongoDB in batches
2. **Transform**: Convert JSON to searchable text format
3. **Embed**: Generate vector embeddings using sentence-transformers
4. **Store**: Save to Qdrant vector database with metadata

### Environment Variables
Create a `.env` file with:
```env
# MongoDB Configuration
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DATABASE=admin

# Qdrant Configuration  
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Querying the RAG System

Once the pipeline is complete, you can search both corporate and procurement data:

```python
from test_rag import search_corporate_data, search_procurement_data

# Search for companies
search_corporate_data("software development company", top_k=5)

# Search for tenders
search_procurement_data("road construction tender", top_k=5)
```

The system returns:
- **Similarity scores** for relevance ranking
- **Original metadata** (NIPT, tender IDs, etc.)
- **Full document access** through stored metadata
- **Source identification** (corporate vs procurement)

5. **Test the system**:
   ```bash
   python test_rag.py
   ```

### Option 2: Manual Setup

1. **Install dependencies**:
   ```bash
   pip install uv
   uv pip install -e .
   ```

2. **Set up environment variables** (create a `.env` file):
   ```env
   MONGO_HOST=localhost
   MONGO_PORT=27017
   MONGO_USERNAME=admin
   MONGO_PASSWORD=password123
   MONGO_DATABASE=admin
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```

3. **Start MongoDB and Qdrant** (using Docker):
   ```bash
   docker run -d --name mongodb -p 27017:27017 \
     -e MONGO_INITDB_ROOT_USERNAME=admin \
     -e MONGO_INITDB_ROOT_PASSWORD=password123 \
     mongo:7.0
   
   docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
   ```

4. **Run the pipelines** (same as Option 1, steps 3-4)

## 📁 Project Structure

```
llm-training/
├── pipelines/
│   ├── open_procurement.py    # OpenProcurement Albania scraper
│   ├── open_corporate.py      # OpenCorporates Albania scraper
│   └── rag.py                 # RAG pipeline
├── test_rag.py               # Test script
├── mongo.py                  # MongoDB connection utilities
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile                # Docker configuration
├── pyproject.toml           # Python dependencies
└── README.md                # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_HOST` | MongoDB host | `localhost` |
| `MONGO_PORT` | MongoDB port | `27017` |
| `MONGO_USERNAME` | MongoDB username | `admin` |
| `MONGO_PASSWORD` | MongoDB password | Required |
| `MONGO_DATABASE` | MongoDB database | `admin` |
| `QDRANT_HOST` | Qdrant host | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` |

### RAG Pipeline Configuration

The RAG pipeline can be configured in `pipelines/rag.py`:

```python
# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Batch processing
BATCH_SIZE = 100

# Qdrant collection name
QDRANT_COLLECTION_NAME = "albania_business_data"
```

## 🔍 Using the RAG System

### Direct Qdrant Access

Once the RAG pipeline is complete, you can access Qdrant directly:

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Search for documents
query = "construction tender"
query_embedding = model.encode(query).tolist()

results = client.search(
    collection_name="albania_business_data",
    query_vector=query_embedding,
    limit=10
)

for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.payload.get('content', '')}")
    print(f"Source: {result.payload.get('source_type', '')}")
    print("---")
```

### Filtering by Source Type

```python
# Search only for tenders
results = client.search(
    collection_name="albania_business_data",
    query_vector=query_embedding,
    query_filter={
        'must': [
            {
                'key': 'source_type',
                'match': {'value': 'tender'}
            }
        ]
    },
    limit=10
)

# Search only for companies
results = client.search(
    collection_name="albania_business_data",
    query_vector=query_embedding,
    query_filter={
        'must': [
            {
                'key': 'source_type',
                'match': {'value': 'company'}
            }
        ]
    },
    limit=10
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_rag.py
```

This will test:
- RAG pipeline execution
- Search functionality with sample queries

## 📈 Performance

### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Speed**: ~2000 sentences/second on CPU
- **Quality**: Good balance of speed and accuracy

### Vector Database
- **Database**: Qdrant
- **Distance Metric**: Cosine similarity
- **Index**: HNSW (Hierarchical Navigable Small World)
- **Performance**: Sub-millisecond search times

## 🔄 Data Flow

1. **Data Ingestion**:
   - Scrape tenders from OpenProcurement Albania
   - Scrape companies from OpenCorporates Albania
   - Store in MongoDB with deduplication

2. **RAG Processing**:
   - Clean and preprocess data
   - Chunk documents into smaller pieces
   - Generate embeddings using sentence transformers
   - Store in Qdrant vector database

3. **Search**:
   - Convert user queries to embeddings
   - Perform similarity search in Qdrant
   - Return relevant documents with metadata

## 🛠️ Development

### Adding New Data Sources

1. Create a new pipeline in `pipelines/`
2. Follow the existing pattern with ZenML steps
3. Update the RAG pipeline to include the new data source
4. Add appropriate cleaning and chunking logic

### Customizing Embeddings

1. Change the `EMBEDDING_MODEL` in `pipelines/rag.py`
2. Popular alternatives:
   - `sentence-transformers/all-mpnet-base-v2` (better quality, slower)
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

### Scaling

- **Horizontal Scaling**: Run multiple Qdrant instances
- **Vertical Scaling**: Increase Qdrant resources for larger datasets
- **Caching**: Add Redis for query result caching

## 🐛 Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**:
   - Check if MongoDB is running
   - Verify connection credentials
   - Ensure network connectivity

2. **Qdrant Connection Failed**:
   - Check if Qdrant is running on the correct port
   - Verify collection exists
   - Check disk space for vector storage

3. **Embedding Model Download Failed**:
   - Check internet connectivity
   - Verify model name is correct
   - Clear model cache if needed

### Logs

Check logs for detailed error information:

```bash
# Docker logs
docker-compose logs rag-pipeline

# Application logs
tail -f logs/rag.log
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions and support, please open an issue on GitHub.
