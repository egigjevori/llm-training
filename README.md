# LLM Training Data Pipeline Architecture

## Overview

This project implements a comprehensive data processing pipeline system designed to extract, process, and vectorize UK business data for LLM training and RAG applications. The system processes Companies House data through multiple specialized pipelines to build datasets for AI applications focused on UK business intelligence.

### Pipeline Ecosystem

| Pipeline | Purpose | Data Source | Output Format |
|:--------:|:-------:|:-----------:|:-------------:|
| **UK Companies Download** | Company registry data download | Companies House API | CSV Files |
| **UK Companies Parser** | CSV data parsing | Downloaded CSV | MongoDB Documents |
| **RAG Pipeline** | Vector embeddings | MongoDB Collections | Qdrant Vector DB |
| **Instruction Dataset** | Training data generation | MongoDB Collections | JSONL Dataset |
| **QLoRA Pipeline** | Model fine-tuning | Instruction datasets | Fine-tuned LLM |

### Data Flow Architecture

```
┌────────────────────────────────────────────────┐
│            UK Companies House Data             │
└────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│            CSV Download & Parsing              │
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

- **Comprehensive Coverage**: Full extraction of UK Companies House corporate data
- **Scalable Architecture**: ZenML orchestration with batch processing capabilities
- **Memory Optimized**: Efficient CSV parsing and MongoDB batch operations
- **Semantic Search Ready**: Vector embeddings for RAG applications
- **Production Ready**: Robust error handling, logging, and monitoring

---

## 1. UK Companies Download Pipeline Architecture

### Purpose
Downloads the latest UK Companies House data dump from the official government portal and extracts CSV files for processing.

### Data Flow
```
Companies House Portal -> URL Discovery -> Download ZIP -> Extract CSV -> Local Storage
```

### Pipeline Steps

#### 1.1 URL Discovery
- Parse Companies House download page
- Locate "BasicCompanyDataAsOneFile" ZIP link
- Extract current snapshot date
- Validate download URL

#### 1.2 File Download
- Stream download with progress tracking
- Handle large file sizes (500MB+)
- Verify file integrity
- Extract date from filename

#### 1.3 Archive Extraction
- Extract CSV from ZIP archive
- Apply meaningful naming convention
- Generate download metadata
- Clean up temporary files

#### 1.4 Metadata Generation
- Create download information record
- Track file sizes and dates
- Store extraction statistics
- Generate processing manifest

### Output Data Structure
Raw CSV files containing complete UK Companies House registry data with download metadata and processing information.

---

## 2. UK Companies Parser Pipeline Architecture

### Purpose
Parses the downloaded UK Companies House CSV files and transforms them into structured MongoDB documents with optimized batch processing.

### Data Flow
```
CSV Files -> Header Cleaning -> Row Parsing -> Data Transformation -> MongoDB Batch Storage
```

### Pipeline Steps

#### 2.1 CSV Processing
- Load CSV files with proper UTF-8 encoding
- Clean headers and remove whitespace artifacts
- Process data in configurable chunks for memory efficiency
- Handle large datasets (1M+ records) without memory overflow

#### 2.2 Data Transformation
- Parse company registration data
- Normalize date formats (DD/MM/YYYY to ISO)
- Build structured address objects
- Extract SIC codes and previous names
- Process financial and legal information

#### 2.3 MongoDB Integration
- Create indexes on company_number for fast lookups
- Use bulk upsert operations for performance
- Handle duplicate records with replace strategy
- Track processing statistics and errors

#### 2.4 Data Validation
- Validate required fields (company_number, name)
- Clean and normalize text values
- Handle missing or malformed data gracefully
- Generate processing reports

### Output Data Structure
Structured company documents in MongoDB with comprehensive business information, optimized for querying and analysis.

**Example Data Structure:**
```json
{
  "_id": "12345678",
  "company_name": "TECH INNOVATIONS LIMITED",
  "company_number": "12345678",
  "status": "Active",
  "category": "Private limited Company",
  "country_of_origin": "United Kingdom",
  "incorporation_date": "2020-01-15",
  "dissolution_date": "",
  "address": {
    "care_of": "",
    "po_box": "",
    "line1": "123 Innovation Street",
    "line2": "Tech Quarter",
    "post_town": "LONDON",
    "county": "Greater London",
    "country": "England",
    "post_code": "EC1A 1BB",
    "full_address": "123 Innovation Street, Tech Quarter, LONDON, Greater London, England, EC1A 1BB"
  },
  "accounts": {
    "ref_day": 31,
    "ref_month": 12,
    "next_due": "2024-12-31",
    "last_made_up": "2023-12-31",
    "category": "SMALL"
  },
  "returns": {
    "next_due": "2024-02-15",
    "last_made_up": "2023-02-15"
  },
  "mortgages": {
    "charges": 0,
    "outstanding": 0,
    "part_satisfied": 0,
    "satisfied": 0
  },
  "sic_codes": [
    "62012 - Business and domestic software development",
    "62020 - Information technology consultancy activities"
  ],
  "previous_names": [],
  "partnerships": {
    "general_partners": 0,
    "limited_partners": 0
  },
  "confirmation_statement": {
    "next_due": "2024-01-29",
    "last_made_up": "2023-01-15"
  },
  "uri": "http://business.data.gov.uk/id/company/12345678",
  "imported_at": "2024-09-25T14:30:00.000000"
}
```

---

## 3. RAG Pipeline Architecture

### Purpose
Transforms UK Companies House MongoDB collections into vector embeddings for semantic search and AI applications.

### Data Flow
```
MongoDB Collections -> Document Chunking -> Embedding Generation -> Qdrant Storage
```

### Pipeline Steps

#### 3.1 Vector Store Management
- Initialize Qdrant client with environment config
- Delete existing collections for clean restart
- Create uk_companies collection
- Configure 384-dimensional vectors with cosine similarity

#### 3.2 Batch Processing Engine
- Fetch documents from MongoDB in configurable batches
- Process uk_companies_house.companies collection
- Handle large datasets with memory optimization
- Support configurable document limits for testing

#### 3.3 Document Transformation
- Convert MongoDB documents to JSON text chunks
- Preserve document structure and metadata
- Add source identification (uk_companies_house)
- Generate unique IDs for each chunk
- Include company metadata (number, name, status, SIC codes)

#### 3.4 Embedding Generation
- Use BAAI/bge-small-en-v1.5 sentence transformer model
- Generate 384-dimensional embeddings
- Process chunks in batches for memory efficiency
- Clear GPU cache on Apple Silicon for memory management

#### 3.5 Vector Storage System
- Store embeddings in Qdrant with metadata
- Maintain links to original documents
- Enable semantic search across UK corporate data
- Use upsert operations for data consistency

### Output Data Structure
Vector database with semantic embeddings of UK Companies House documents, enabling RAG applications and search capabilities.

---

## 4. Instruction Dataset Pipeline Architecture

### Purpose
Generates instruction-response pairs from UK Companies House data for training language models on UK business intelligence.

### Data Flow
```
MongoDB Corporate Data -> Instruction Generation -> Dataset Export
```

### Pipeline Steps

#### 4.1 Data Extraction
- MongoDB connection to uk_companies_house database
- Data validation and filtering
- Support for active and dissolved companies
- Configurable data limits for dataset size control

#### 4.2 Instruction Generation
- **Company-specific**: Name, registration details, status, legal structure, incorporation dates
- **Location-based**: Geographic clustering, regional analysis
- **Industry-focused**: SIC code analysis, sector comparisons
- **Compliance**: Accounts, charges, confirmation statements
- **Historical**: Previous names, status changes

#### 4.3 Data Processing
- Text cleaning and validation
- Automatic category classification
- Metadata addition (timestamp, version, data source)
- Quality filtering for meaningful responses

#### 4.4 Export
- JSONL format with UTF-8 encoding
- Automatic directory creation and error handling
- Statistics generation and reporting

### Output
JSONL dataset with structured Q&A pairs covering UK business data and metadata.

**Example Format:**
```json
{
  "instruction": "What is the current status of TECH INNOVATIONS LIMITED?",
  "response": "The current status of TECH INNOVATIONS LIMITED is Active.",
  "category": "status",
  "source": "uk_companies",
  "generated_at": "2024-09-25T16:04:40.123456",
  "dataset_version": "2.0",
  "data_source": "uk_companies_house"
}
```
---

## 5. QLoRA Pipeline Architecture

### Purpose
Fine-tunes language models using QLoRA for instruction-following on UK Companies House business data, optimized for MacBook/CPU environments.

### Data Flow
```
Instruction Dataset → Model Preparation → Training → Evaluation → Inference Pipeline
```

### Pipeline Steps

#### 5.1 Dataset Management
- Load JSONL instruction datasets from UK companies data
- Automatic train/validation split (90/10)
- Format for instruction-following training with chat templates
- Support for UTF-8 encoded UK business content

#### 5.2 Model Preparation
- Load base model (DistilGPT-2 optimized for CPU training)
- Configure LoRA parameters (rank=8, alpha=16, dropout=0.1)
- Target attention layers for efficient parameter updates
- CPU-optimized loading without quantization

#### 5.3 Training Pipeline
- Tokenization and data preprocessing with proper padding
- Configurable epochs, batch size, learning rate
- Cosine learning rate scheduling with warmup
- Memory-efficient training for MacBook environments
- Automatic checkpointing and model saving

#### 5.4 Evaluation & Inference
- Test set evaluation with sample generation
- Model performance assessment on UK business queries
- Automatic inference script creation
- JSON export of evaluation results

### Output
Fine-tuned model with LoRA adapters, evaluation results, and inference pipeline optimized for UK business intelligence tasks.

---

## Technical Implementation Details

### Memory Optimization Strategies

#### CSV Processing
- **Chunk-based Processing**: Process large CSV files (1M+ records) in configurable chunks to prevent memory overflow
- **Header Cleaning**: Remove leading/trailing spaces from CSV headers that cause parsing issues
- **Batch MongoDB Operations**: Use bulk upsert operations with configurable batch sizes (default: 1000)
- **Memory Cleanup**: Clear processed chunks from memory after each batch operation

#### MongoDB Integration
- **Indexing Strategy**: Create unique indexes on `company_number` for fast lookups and duplicate prevention
- **Upsert Operations**: Use `ReplaceOne` with upsert for handling updates to existing companies
- **Connection Management**: Properly close MongoDB connections after batch operations
- **Error Handling**: Graceful handling of MongoDB connection failures with fallback logging

#### Vector Processing
- **Batch Embeddings**: Generate embeddings in configurable batches to manage memory usage
- **GPU Cache Management**: Clear MPS cache on Apple Silicon after each batch
- **Document Chunking**: Convert MongoDB documents to JSON strings for consistent embedding generation
- **Metadata Preservation**: Include company metadata in vector payloads for enhanced search capabilities

