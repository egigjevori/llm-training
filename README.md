# LLM Training Data Pipeline Architecture

## Overview

This project implements two specialized data processing pipelines designed to extract, process, and vectorize Albanian business data for LLM training and RAG applications. Each pipeline serves a distinct purpose in building a comprehensive dataset for AI applications focused on Albanian business intelligence.

### Pipeline Ecosystem

| Pipeline | Purpose | Data Source | Output Format |
|----------|---------|-------------|---------------|
| **OpenCorporates** | Company registry data | opencorporates.al | MongoDB Documents |
| **RAG Pipeline** | Vector embeddings | MongoDB Collections | Qdrant Vector DB |
| **Instruction Dataset** | Training data generation | MongoDB Collections | JSONL Dataset |
| **QLoRA Pipeline** | Model fine-tuning | Instruction datasets | Fine-tuned LLM |

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
                       │
                       ▼
┌────────────────────────────────────────────────┐
│      Instruction Dataset Generation            │
└────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│        Model Fine-tuning (QLoRA Pipeline)      │
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

### Data Flow
```
Pagination Discovery -> Company Listings -> Detail Scraping -> MongoDB Storage
```

### Pipeline Steps

#### 1.1 Pagination Discovery
- Parse search results page 1
- Find pagination controls
- Extract "Last" button href
- Parse page number from URL

#### 1.2 Listing Scraper
- Scan company cards
- Extract company name, ID, description
- Extract location and currency
- Collect detail page URLs

#### 1.3 Detail Scraper
- Target main content area
- Extract data from tables
- Process financial sections
- Extract shareholder lists
- Handle document links
- Apply text corrections

#### 1.4 Storage System
- Connect to MongoDB
- Use company ID as unique ID
- Store in companies collection
- Provide storage feedback

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

### Data Flow
```
MongoDB Collections -> Document Chunking -> Embedding Generation -> Qdrant Storage
```

### Pipeline Steps

#### 2.1 Vector Store Management
- Initialize Qdrant client with environment config
- Delete existing collections for clean restart
- Create corporate_data collection
- Configure 384-dimensional vectors with cosine similarity

#### 2.2 Batch Processing Engine
- Fetch documents from MongoDB in configurable batches
- Process opencorporates_albania.companies collection

#### 2.3 Document Transformation
- Convert MongoDB documents to JSON text chunks
- Preserve document structure and metadata
- Add source identification (corporate)
- Generate unique IDs for each chunk

#### 2.4 Embedding Generation
- Use BAAI/bge-small-en-v1.5 sentence transformer model
- Generate 384-dimensional embeddings
- Process chunks in batches for memory efficiency

#### 2.5 Vector Storage System
- Store embeddings in Qdrant with metadata
- Maintain links to original documents
- Enable semantic search across corporate data
- Use upsert operations for data consistency

### Output Data Structure
Vector database with semantic embeddings of Albanian corporate documents, enabling RAG applications and search capabilities.

---

## 3. Instruction Dataset Pipeline Architecture

### Purpose
Generates instruction-response pairs from Albanian corporate data for training language models.

### Data Flow
```
MongoDB Corporate Data -> Instruction Generation -> Dataset Export
```

### Pipeline Steps

#### 3.1 Data Extraction
- MongoDB connection
- Data validation

#### 3.2 Instruction Generation
- **Company-specific**: Name, business activity, location, status, legal form
- **Comparative**: Geographic clustering, industry analysis
- **Analytical**: Company counts, market share, industry distribution

#### 3.3 Data Processing
- Text cleaning and validation
- Automatic category classification
- Metadata addition (timestamp, version)

#### 3.4 Export
- JSONL format with UTF-8 encoding
- Automatic directory creation and error handling

### Output
JSONL dataset with structured Q&A pairs and metadata.

**Example Format:**
```json
{
  "instruction": "What is the business activity of VIOLA DOG & CAT?",
  "response": "The business activity of VIOLA DOG & CAT is: Import - eksport, shpërndarja dhe tregtimi me shumicë e pakicë...",
  "category": "business_activity",
  "source": "corporate",
  "generated_at": "2025-01-28T16:04:40.123456",
  "dataset_version": "1.0"
}
```
---

## 4. QLoRA Pipeline Architecture

### Purpose
Fine-tunes language models using QLoRA for instruction-following on Albanian business data.

### Data Flow
```
Instruction Dataset → Model Preparation → Training → Evaluation → Inference Pipeline
```

### Pipeline Steps

#### 4.1 Dataset Management
- Load JSONL instruction datasets
- Automatic train/validation split (90/10)
- Format for instruction-following training

#### 4.2 Model Preparation
- Load base model
- Configure LoRA parameters (rank, alpha, dropout)

#### 4.3 Training Pipeline
- Tokenization and data preprocessing
- Configurable epochs, batch size, learning rate
- Automatic checkpointing and validation

#### 4.4 Evaluation & Inference
- Test set evaluation and sample generation
- Automatic inference script creation

### Output
Fine-tuned model with LoRA adapters, evaluation results, and inference pipeline.
