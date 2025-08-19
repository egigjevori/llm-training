# Pipeline Architecture Documentation

## 1. Open Procurement Pipeline Architecture

### Purpose
Extracts public procurement data from the Albanian government's Open Procurement portal (openprocurement.al).

### Architecture Pattern
**Multi-Stage Web Scraping with Pagination Handling**

### Data Flow
```
Page Discovery -> Link Extraction -> Data Scraping -> Storage
```

### Core Components

#### Pagination System
- Dynamically discovers available pages by parsing pagination controls
- Searches for Albanian "me pas" (next) buttons and pagination symbols
- Handles dynamic page ranges without hardcoded limits

#### Link Discovery Engine  
- Scans tender listing pages for individual tender URLs
- Filters links matching the tender detail URL pattern
- Deduplicates links while preserving order

#### Data Extraction Engine
- Targets structured HTML tables containing tender information
- Extracts key-value pairs from table rows
- Handles Albanian text encoding properly
- Adds metadata like scraping timestamps and tender IDs

#### Storage Layer
- Uses MongoDB for persistence with tender_id as unique identifier
- Implements upsert operations to prevent duplicates
- Provides immediate feedback on storage success/failure

### Processing Characteristics
- **Rate Limiting**: Built-in delays between requests to respect server resources
- **Error Recovery**: Continues processing even when individual pages fail
- **Real-time Output**: Displays extracted data immediately during processing
- **Albanian Language Support**: Proper UTF-8 encoding handling

### Output Data Structure
Structured tender documents containing procurement details, amounts, deadlines, contracting authorities, and bidding requirements.

---

## 2. OpenCorporates Pipeline Architecture

### Purpose
Scrapes comprehensive company information from the Albanian OpenCorporates registry (opencorporates.al).

### Architecture Pattern
**Two-Tier Hierarchical Scraping**

### Data Flow
```
Pagination Discovery -> Company Listings -> Detail Scraping -> Storage
```

### Core Components

#### Pagination Discovery
- Analyzes search result pagination to determine total pages
- Extracts page count from "Last" button href attributes
- Plans complete site coverage based on discovered range

#### Listing Scraper
- Extracts basic company information from search result cards
- Captures company names, NIPT numbers, descriptions, locations
- Collects detail page URLs for deeper scraping

#### Detail Scraper
- Performs comprehensive extraction from individual company pages
- Processes complex HTML structures including tables and lists
- Handles multiple data types: text, links, financial data, ownership info
- Extracts specialized sections like financial documents and shareholder lists

#### Data Processing Engine
- Normalizes Albanian text with regex-based spacing corrections
- Handles various HTML structures across different company profiles
- Processes nested data like document links and ownership structures
- Maintains data relationships and hierarchies

#### Storage System
- Uses NIPT (Albanian tax ID) as unique company identifier
- MongoDB storage with comprehensive company profiles
- Deduplication and update capabilities for existing records

### Processing Characteristics
- **Multi-Level Extraction**: Basic info � detailed profiles
- **Complex Data Handling**: Financial data, ownership structures, legal documents
- **Text Normalization**: Albanian language text cleaning and formatting
- **Robust Error Handling**: Graceful degradation when data is missing

### Output Data Structure
Rich company profiles including basic information, financial data, ownership structures, contact details, legal status, and regulatory documents.

---

## 3. RAG Pipeline Architecture

### Purpose
Transforms MongoDB document collections into vector embeddings for semantic search and AI applications.

### Architecture Pattern
**ETL Pipeline with Vector Processing**

### Data Flow
```
MongoDB Batch Fetch -> Document Chunking -> Embedding Generation -> Vector Storage
```

### Core Components

#### Vector Store Management
- Initializes and manages Qdrant vector collections
- Implements clean restart by deleting existing collections
- Creates separate collections for corporate and procurement data

#### Batch Processing Engine
- Fetches documents from MongoDB in configurable batches
- Handles large datasets through pagination and streaming
- Processes all available documents without artificial limits

#### Document Transformation
- Converts MongoDB documents to JSON text chunks
- Maintains document structure while creating searchable text
- Preserves metadata for document provenance and filtering

#### Embedding Generation
- Uses sentence transformer models for semantic embeddings
- Generates 384-dimensional vectors with cosine similarity
- Processes chunks in batches for memory efficiency

#### Vector Storage System
- Stores embeddings in Qdrant with rich metadata
- Maintains links to original documents
- Enables semantic search across both data sources

### Processing Characteristics
- **Unlimited Processing**: No document count restrictions
- **Memory Efficient**: Batch processing prevents memory overflow
- **Metadata Preservation**: Maintains document context and source information
- **Clean Restart**: Fresh vector collections for each run

### Technical Specifications
- **Embedding Model**: Multilingual sentence transformer (384 dimensions)
- **Vector Distance**: Cosine similarity for semantic matching
- **Batch Size**: Configurable (default 100 documents per batch)
- **Collections**: Separate corporate_data and procurement_data stores

### Output Structure
Searchable vector database with semantic embeddings of Albanian corporate and procurement documents, enabling RAG applications and intelligent search capabilities.

---
