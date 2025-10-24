# LLM Training Pipeline
## Demo Presentation

---

# 🎯 What This System Does

## Four Main Components:

1. **Extract** company data from UK Companies House
2. **Vectorize** data for semantic search
   - *Vectorize: Convert text to numbers (embeddings) for AI to understand*
3. **Fine-tune** models with QLoRA
   - *Fine-tune: Teach pre-trained model domain-specific knowledge*
4. **Chat** interface with 3 response modes

---

# 🏗️ System Architecture

```
     ┌──────────────────────────────┐
     │    UK COMPANIES HOUSE DATA   │
     └─────────┬────────────────────┘
               ↓
        ┌──────────────┐
        │   CSV Files  │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │   MongoDB    │
        └──┬────────┬──┘
           │        │
           ↓        ↓
    ┌──────────┐  ┌──────────────┐
    │ Qdrant   │  │ Instruction  │
    │ Vectors  │  │   Dataset    │
    └────┬─────┘  └──────┬───────┘
         │               ↓
         │        ┌──────────────┐
         │        │ Fine-tuned   │
         │        │    Model     │
         │        └──────┬───────┘
         │               │
         └───────┬───────┘
                 ↓
          ┌────────────┐
          │  Chat CLI  │
          └────────────┘
```

---

# 📥 Data Pipeline

## Step 1: Download
```bash
cd pipelines
uv run uk_companies_download.py
```
- Fetch ZIP files from UK Companies House
- Extract CSV data
- *CSV: Comma-Separated Values (spreadsheet format)*

## Step 2: Parse & Load
```bash
cd pipelines
uv run uk_companies_parser.py
```
- Parse 1M+ company records
- Batch processing (10,000 records/batch)
  - *Batch: Process data in chunks to manage memory*
- Load into MongoDB

## Step 3: Validate
- Create indexes (speed up database queries)
- Handle missing data

---

# 📊 Sample Data Structure

```json
{
  "CompanyName": "TECH INNOVATIONS LIMITED",
  "CompanyNumber": "12345678",
  "RegAddress.AddressLine1": "123 HIGH STREET",
  "RegAddress.PostTown": "LONDON",
  "RegAddress.PostCode": "SW1A 1AA",
  "CompanyStatus": "Active",
  "SICCode.SicText_1": "62012 - Software development"
}
```
*JSON: JavaScript Object Notation (structured data format)*

### Key Fields:
- Company name & registration number
- Full address
- Status & industry codes
- SIC Code: Standard Industrial Classification (business activity type)
- Financial information

---

# 🔍 RAG Pipeline

## What is RAG?
**Retrieval-Augmented Generation**
- *RAG: Combine search (retrieval) with AI generation for accurate responses*
- Semantic search (not keyword matching)
  - *Semantic: Understands meaning, not just exact words*
- Understand context and meaning behind queries

## Pipeline Steps:

```
MongoDB → Text Processing → BGE-large Model → 1024-dim Vectors → Qdrant
```

### Process:
1. Load company records from MongoDB
2. Convert to text
3. Generate embeddings (vectors)
   - *Vectors: Arrays of numbers representing text meaning*
   - *1024-dim: Each text becomes 1024 numbers*
4. Store in Qdrant with metadata
   - *Metadata: Additional info like company name, address*

---

# 📝 Text Processing Example

## Raw Data (JSON):
```json
{
  "CompanyName": "TECH INNOVATIONS LIMITED",
  "CompanyNumber": "12345678",
  "RegAddress.PostTown": "LONDON",
  "CompanyStatus": "Active",
  "SICCode.SicText_1": "62012 - Software development"
}
```

## Prepared Chunk (actual from pipeline):
```python
{
  "id": "uuid-generated-string",
  "text": '{"CompanyName":"TECH INNOVATIONS LIMITED","CompanyNumber":"12345678","RegAddress.PostTown":"LONDON","CompanyStatus":"Active","SICCode.SicText_1":"62012 - Software development"}',
  "metadata": {
    "source": "uk_companies_house",
    "company_number": "12345678",
    "company_name": "TECH INNOVATIONS LIMITED",
    "status": "Active",
    "category": "Private Limited Company",
    "country": "UK",
    "sic_codes": ["62012"],
    "full_address": "123 HIGH STREET, LONDON, SW1A 1AA",
    "incorporation_date": "2015-06-01"
  }
}
```
*The `text` field (JSON string) gets embedded → 1024 numbers*

## What Happens in Pipeline:
1. **MongoDB doc** → `create_company_chunk()` → **Chunk dict** (above)
2. **text field** → BGE Model → **Embedding** [0.23, -0.45, 0.78, ...] (1024 values)
3. **Store in Qdrant**:
   - Vector: embedding
   - Payload: {text, metadata}
4. **Query** "software companies in London" → Find similar vectors

---

# 🎯 Vector Search Demo

### Run the demo:
```bash
uv run demo_vector_search.py
```

*Score: Similarity score (0-1). Higher = more relevant*
- *0.8+: Very relevant match*
- *0.5-0.8: Somewhat relevant*
- *<0.5: Not very relevant*

### Key Point:
Understands "technology" relates to "software", "digital", etc.

---

# 📚 Instruction Dataset Generation

## What & Why?

**Training data that teaches models domain-specific knowledge**
- *Instruction Dataset: Question-answer pairs for fine-tuning*
- Base models know general language but not your specific domain
- Solution: Auto-generate Q&A pairs from company data

### Pipeline:
```bash
uv run ./pipelines/instruction_dataset.py
```

**Process:** MongoDB → Generate Q&A → Validate → JSONL
- Input: 1000 companies
- Output: ~12,000 instruction pairs
- 10-15 pairs per company

---

# 📋 Data Format & Categories

### Format:
```json
{
  "instruction": "What is the company number for TECH INNOVATIONS LIMITED?",
  "response": "The company number is 12345678.",
  "category": "company_info",
  "source": "uk_companies"
}
```

### Categories (13 types):
- **company_info, status, location** - Basic lookups
- **industry, accounts, charges** - Business details
- **history, overview** - Summaries
- **location_comparison, industry_comparison** - Multi-company queries

---

# 💡 Example Pairs

```json
{"instruction": "What is the name of company 12345678?",
 "response": "The company is TECH INNOVATIONS LIMITED."}

{"instruction": "In which town is TECH INNOVATIONS LIMITED registered?",
 "response": "TECH INNOVATIONS LIMITED is registered in LONDON."}

{"instruction": "What industry does TECH INNOVATIONS LIMITED operate in?",
 "response": "62012 - Software development."}

{"instruction": "Which companies are registered in LONDON?",
 "response": "Companies include: TECH INNOVATIONS LIMITED, DIGITAL SOLUTIONS LTD..."}
```

---

# 🧠 QLoRA Fine-tuning

## What is QLoRA?

**Teaching a pre-trained AI model your specific domain**

Think of it like this:
- Base model = University graduate (knows general knowledge)
- Fine-tuning = Job training (learns company-specific tasks)
- QLoRA = Efficient training method (learns faster, uses less resources)

### Why QLoRA?
✅ Works on regular laptops (no expensive GPUs)
✅ Fast training (hours, not days)
✅ Small updates (adds knowledge without retraining everything)
✅ Uses the Q&A pairs we just created

---

# ⚙️ Training Process

```bash
uv run ./pipelines/qlora_finetuning.py
```

## What Happens:

1. **Load base model** (DistilGPT-2)
   - Starting point: AI that knows English

2. **Add training adapter**
   - Like adding a specialized lens to focus on UK companies

3. **Train with Q&A pairs**
   - Feed 12,000 question-answer examples
   - Model learns patterns and responses

4. **Save progress**
   - Checkpoint every few steps (in case of interruption)

5. **Done!**
   - Model now knows about UK Companies House data

---

# 💬 Chat CLI - Three Modes

## Mode 1: Qdrant (Pure Retrieval)
Vector search only - no generation
- *Retrieval: Find and return existing data*

## Mode 2: Model (Pure Generation)
Fine-tuned model only - no retrieval
- *Generation: AI creates new text based on training*

## Mode 3: Hybrid ⭐ (RAG + Generation)
Retrieve context + generate response
- *Context: Relevant information passed to model*
- Best of both worlds!

---

# 🔹 Mode 1: Qdrant

```bash
uv run ./chat/chat_cli.py --mode qdrant
```

**Query:** "What technology companies are in London?"

**Output:**
Returns matching companies with relevance scores

### Use Case:
- Direct lookups
- High accuracy
- No narrative

---

# 🔹 Mode 2: Model

```bash
uv run ./chat/chat_cli.py --mode model
```

**Query:** "What does SIC code 41200 mean?"

**Output:**
```
SIC code 41200 refers to "Construction of residential
and non-residential buildings."
```

### Use Case:
- Explanations
- General knowledge
- Medium accuracy

---

# 🔹 Mode 3: Hybrid ⭐

```bash
uv run ./chat/chat_cli.py --mode hybrid
```

**Query:** "What technology companies are in London and what do they do?"

### Process:
1. Retrieve relevant companies from Qdrant
2. Pass context to model
3. Generate natural language response

**Output:** Factual + narrative response

### Use Case:
- Complex questions
- High accuracy + fluency

---

# 📊 Mode Comparison

| Mode | Retrieval | Generation | Accuracy | Best For |
|------|-----------|------------|----------|----------|
| **Qdrant** | ✅ | ❌ | High | Direct lookups |
| **Model** | ❌ | ✅ | Medium | General explanations |
| **Hybrid** ⭐ | ✅ | ✅ | High | Complex questions |

*Accuracy: How correct/factual the response is*

## Winner: Hybrid Mode
- Factual accuracy from retrieval (grounded in real data)
- Natural language fluency from generation (human-like responses)
- References specific companies (provides sources)

---

# 🔄 Complete Data Flow

```
1. Companies House → CSV → MongoDB

2. MongoDB → Embeddings → Qdrant

3. MongoDB → Q&A pairs → Fine-tuned model

4. User query → Provider → Response
   - Qdrant mode: Vector search
   - Model mode: Direct generation
   - Hybrid mode: Search + Generate
```

---

# 🎯 Key Takeaways

1. **Data Pipeline:** Process 1M+ records efficiently
2. **Semantic Search:** Vector embeddings understand meaning
3. **QLoRA:** Fine-tune models on consumer hardware
4. **RAG:** Hybrid mode = best of both worlds
5. **Modular:** Easy to extend and customize

---

# 🚀 With Bigger Resources?

## Current (MacBook-friendly):
- **Model:** DistilGPT-2 (82M parameters)
- **Method:** QLoRA (4-bit quantization)
  - *Quantization: Compress model to use less memory*
- **Hardware:** MacBook (16GB RAM)
- **Training:** Few hours

## With GPU Cluster:
- **Model:** Llama-3 70B or GPT-3.5 equivalent
  - *70B = 70 billion parameters (850x larger!)*
- **Method:** Full LoRA (16-bit precision)
  - *No compression needed with 80GB GPU*
- **Hardware:** Multiple GPUs (A100/H100)
- **Training:** Similar time but much better quality

## Key Differences:

| Aspect | MacBook (Current) | GPU Cluster |
|--------|------------------|-------------|
| Model Size | 82M parameters | 70B parameters |
| Quality | Good for demos | Enterprise-grade |
| Method | QLoRA (4-bit) | LoRA (16-bit) |
| Training Speed | Hours | Hours (parallel) |
| Cost | $0 | $50-200/training |
| Deployment | Local | Cloud/API |

## Instruction Dataset at Scale:

### Current (MacBook):
- **Companies:** 1,000 sampled
- **Instruction pairs:** ~12,000
- **Generation time:** Minutes
- **Storage:** Few MB (JSONL file)

### With More Resources:
- **Companies:** All 5M+ UK companies
- **Instruction pairs:** ~60 million
- **Generation time:** Hours (distributed processing)
- **Storage:** Few GB
- **Additional data:**
  - Historical changes (company name changes, mergers)
  - Financial data (accounts, revenue trends)
  - Network data (company relationships, directors)
  - Multi-turn conversations (follow-up questions)

### Enhanced Question Types:
- **Time-series:** "How did revenue change 2020-2024?"
- **Comparisons:** "Compare TECH LTD vs competitors"
- **Analytics:** "Show trends in London tech sector"
- **Complex:** "Which companies founded in 2020 are still active?"

## What Changes:
- **Better responses** - Larger models understand context better
- **More accurate** - Higher precision training
- **Scalability** - Handle more concurrent users
- **Richer data** - More comprehensive instruction coverage
- **Cost** - Higher infrastructure costs

---

# 🎬 Summary

## What We Built:
✅ Complete LLM training pipeline
✅ Vector search with Qdrant
✅ Fine-tuned model (QLoRA)
✅ Interactive chat interface
✅ Modular architecture

---

# 🙏 Thank You!
