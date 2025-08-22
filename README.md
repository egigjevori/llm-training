# LLM Training Data Pipeline Architecture

## Overview

This project implements two specialized data processing pipelines designed to extract, process, and vectorize Albanian business data for LLM training and RAG applications. Each pipeline serves a distinct purpose in building a comprehensive dataset for AI applications focused on Albanian business intelligence.

### Pipeline Ecosystem

| Pipeline | Purpose | Data Source | Output Format | Architecture Pattern |
|----------|---------|-------------|---------------|---------------------|
| **OpenCorporates** | Company registry data | opencorporates.al | MongoDB Documents | Two-tier hierarchical scraping |
| **RAG Pipeline** | Vector embeddings | MongoDB Collections | Qdrant Vector DB | ETL with embedding generation |

### Data Flow Architecture

```
┌────────────────────────────────────────────────┐
│             Open Corporates Data               │
└────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│           Document Storage (MongoDB)           │
└────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│        Vector Processing (RAG Pipeline)        │
└────────────────────────────────────────────────┘
```

### Key Features

- **Comprehensive Coverage**: Full extraction of Albanian corporate data
- **Scalable Architecture**: ZenML orchestration with parallel processing capabilities
- **Albanian Language Support**: Proper UTF-8 encoding and text normalization
- **Semantic Search Ready**: Vector embeddings for RAG applications
- **Production Ready**: Robust error handling, logging, and monitoring

---

## 1. OpenCorporates Pipeline Architecture

### Purpose
Scrapes comprehensive company information from the Albanian OpenCorporates registry (opencorporates.al/sq/search).

### Architecture Pattern
**ZenML Pipeline with Two-Tier Hierarchical Scraping**

### Data Flow
```
Pagination Discovery -> Company Listings -> Detail Scraping -> MongoDB Storage
```

### Core Components

#### 1.1 Pagination Discovery
1. Parse search results page 1
2. Find pagination controls
3. Extract "Last" button href
4. Parse page number from URL

#### 1.2 Listing Scraper
1. Scan company cards
2. Extract company name
3. Extract company ID
4. Extract description
5. Extract location and currency
6. Collect detail page URLs

#### 1.3 Detail Scraper
1. Target main content area
2. Extract data from tables
3. Process financial sections
4. Extract shareholder lists
5. Handle document links
6. Apply text corrections

#### 1.4 Storage System
1. Connect to MongoDB
2. Use company ID as unique ID
3. Store in companies collection
4. Provide storage feedback

### Output Data Structure
Structured company profiles containing business information, financial data, ownership structures, and legal documents, with metadata.

**Example Data Structure:**
```json
{
  "company_id": "L61305031N",
  "name": "VIOLA DOG & CAT",
  "other_trading_names": "VIOLA DOG & CAT",
  "description": "Import - eksport, shpërndarja dhe tregtimi me shumicë e pakicë i të gjithë artikujve, mallrave dhe lëndëve të para të të gjitha llojeve që nuk ndalohen nga legjisLaçioni shqiptar dhe ai ndërkombëtar.",
  "business_object": "Import - eksport, shpërndarja dhe tregtimi me shumicë e pakicë i të gjithë artikujve, mallrave dhe lëndëve të para të të gjitha llojeve që nuk ndalohen nga legjisLaçioni shqiptar dhe ai ndërkombëtar.",
  "legal_form": "Shoqëri me përgjegjësi të kufizuar",
  "status": "Aktiv",
  "registration_date": "04.01.2016",
  "capital": "100 000,00",
  "currency": "ALL",
  "address": "Seli/Zyra Qendrore:Tiranë Rruga e Kavajës, ndërtesë private 2 katëshe, përballë ish Kombinati të Drurit \"Misto Mame\"",
  "location": "Tiranë",
  "region": "Tiranë",
  "administrators": ["Ervin Anxhaku"],
  "shareholders": ["Ervin Anxhaku - 100%"],
  "license": "-",
  "additional_sections": {"Follow Us On": "Twitter Facebook"},
  "detail_url": "https://opencorporates.al/sq/nipt/l61305031n",
  "collection_date": "2025-07-28 16:04:40"
}
```

---

## 2. RAG Pipeline Architecture

### Purpose
Transforms MongoDB document collections into vector embeddings for semantic search and AI applications.

### Architecture Pattern
**ZenML Pipeline with ETL Vector Processing**

### Data Flow
```
MongoDB Collections -> Document Chunking -> Embedding Generation -> Qdrant Storage
```

### Core Components

#### 2.1 Vector Store Management
1. Initialize Qdrant client with environment config
2. Delete existing collections for clean restart
3. Create corporate_data collection
4. Configure 384-dimensional vectors with cosine similarity

#### 2.2 Batch Processing Engine
1. Fetch documents from MongoDB in configurable batches
2. Process opencorporates_albania.companies collection

#### 2.3 Document Transformation
1. Convert MongoDB documents to JSON text chunks
2. Preserve document structure and metadata
3. Add source identification (corporate)
4. Generate unique IDs for each chunk

#### 2.4 Embedding Generation
1. Use all-MiniLM-L6-v2 sentence transformer model
2. Generate 384-dimensional embeddings
3. Process chunks in batches for memory efficiency
4. Show progress bar during encoding

#### 2.5 Vector Storage System
1. Store embeddings in Qdrant with rich metadata
2. Maintain links to original documents
3. Enable semantic search across corporate data
4. Use upsert operations for data consistency

### Output Data Structure
Vector database with semantic embeddings of Albanian corporate documents, enabling RAG applications and intelligent search capabilities.
