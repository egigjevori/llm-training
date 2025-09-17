# Chat CLI Application 🚀

A beautiful, interactive chat CLI application that provides multiple response modes for querying your data and models.

## Features

### 🎯 Multiple Response Modes
- **Qdrant Mode**: Vector search and retrieval from your Qdrant database
- **Model Mode**: Direct inference using your fine-tuned model

### 🎨 Beautiful Interface
- Claude Code-inspired terminal UI with rich formatting
- Animated typing indicators and loading spinners
- Syntax highlighting and markdown rendering for responses
- Color-coded response panels by source type
- ASCII art banner and professional styling

### 💬 Interactive Features
- Conversation history tracking
- Built-in help system and commands
- Graceful error handling and recovery
- Real-time mode switching capabilities

## Installation

Install the required dependencies:

```bash
pip install -e .
```

This will install all necessary packages including:
- `rich` for beautiful terminal UI
- `click` for CLI framework
- `prompt-toolkit` for interactive input

## Usage

### Basic Usage

```bash
# Run in Qdrant mode (default - vector search)
cd chat && python chat_cli.py

# Use only Qdrant vector search
cd chat && python chat_cli.py --mode qdrant

# Use only the fine-tuned model
cd chat && python chat_cli.py --mode model
```

### Advanced Options

```bash
# Custom Qdrant collection
python chat_cli.py --mode qdrant --collection my_data

# Custom model settings
python chat_cli.py --mode model --temperature 0.5 --max-tokens 300

# Hybrid with custom settings
python chat_cli.py --mode hybrid --top-k 5 --temperature 0.8 --collection corporate_data
```

### Available Commands

Once in the chat interface, you can use these commands:

- `/help` - Show available commands
- `/mode` - Display current mode information
- `/history` - Show recent conversation history
- `/clear` - Clear conversation history
- `/quit` - Exit the application

## Response Modes Explained

### 🔍 Qdrant Mode
- Performs semantic vector search in your Qdrant database
- Returns most relevant documents with similarity scores
- Best for: Finding specific information from your knowledge base
- Response includes: Document content, company details, relevance scores

### 🤖 Model Mode
- Uses your fine-tuned model for direct text generation
- Leverages the model's trained knowledge
- Best for: General questions and creative responses
- Response includes: Model-generated text based on training

### 🔄 Hybrid Mode (Recommended)
- Combines the best of both approaches
- Retrieves relevant context from Qdrant
- Uses fine-tuned model to generate contextual responses
- Best for: Comprehensive answers that use both retrieval and generation
- Response includes: Context-aware generated text

## Configuration

### Environment Variables
The application uses these environment variables:

```bash
# Qdrant connection (optional, defaults shown)
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
```

### Model Paths
- Default fine-tuned model path: `./models/finetuned_model`
- Default base model: `Qwen/Qwen2.5-1.5B-Instruct`

You can override these with command-line options:

```bash
python chat_cli.py --model-path /path/to/your/model --base-model your/base/model
```

## Example Interactions

### Qdrant Mode
```
💬 What software companies are in the database?

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                          Qdrant Vector Search                         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Based on the documents I found, here's what I can tell you:           │
│                                                                        │
│ **Result 1** (Relevance: 0.85)                                       │
│ Company: TechCorp Solutions                                           │
│ Company ID: TC12345                                                   │
│ Status: Active                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Hybrid Mode
```
💬 Tell me about construction companies in Tirana

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                      Hybrid RAG (Qdrant + Model)                      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Based on the database search results, I found several construction    │
│ companies in Tirana. Here's what I can tell you:                      │
│                                                                        │
│ The most prominent construction company in the database is...          │
│                                                                        │
│ [Model generates comprehensive response using retrieved context]       │
└─────────────────── Source: hybrid • Context: 3 docs • Temp: 0.7 ─────┘
```

## Architecture

The application is built with a modular provider pattern:

```
chat_cli.py          # Main CLI interface and UI
├── providers.py     # Response provider implementations
├── utils/
│   ├── qdrant.py   # Qdrant utilities (existing)
│   └── mongo.py    # MongoDB utilities (existing)
└── inference.py    # Model inference utilities (existing)
```

### Provider Architecture
- `ResponseProvider` (ABC): Base class for all providers
- `QdrantProvider`: Handles vector search operations  
- `ModelProvider`: Manages fine-tuned model inference
- `HybridProvider`: Combines both approaches for RAG

## Troubleshooting

### Common Issues

1. **Model not found**
   ```bash
   # Ensure your model path is correct
   python chat_cli.py --model-path ./path/to/your/model
   ```

2. **Qdrant connection error**
   ```bash
   # Check Qdrant is running
   curl http://localhost:6333/collections
   ```

3. **Memory issues**
   - Use smaller `--max-tokens` values
   - Lower `--temperature` for more deterministic responses
   - Reduce `--top-k` for fewer context documents

### Performance Tips
- Use Qdrant mode for fastest responses
- Hybrid mode provides best quality but is slower
- Model mode is good for general questions without specific data needs

## Development

To extend the application:

1. Create new providers by inheriting from `ResponseProvider`
2. Add provider to the factory function in `providers.py`
3. Update CLI options in `chat_cli.py`

The modular design makes it easy to add new response sources and capabilities!