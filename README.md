# RAG Evaluation System

A modular Python system for evaluating Retrieval-Augmented Generation (RAG) responses using Azure OpenAI.

## Architecture Overview

The system has been refactored into a clean, modular architecture with the following components:

```
├── config.py              # Configuration management
├── rag_api_client.py       # RAG API client
├── data_parser.py          # Response parsing logic
├── rag_evaluator.py        # Evaluation using Azure OpenAI
├── main.py                 # Main orchestration pipeline
├── rag_eval.py             # Updated entry point (backward compatible)
└── requirements.txt        # Dependencies
```

## Components

### 1. Configuration (`config.py`)
- Centralized configuration management
- Environment variable handling
- Configuration validation

### 2. RAG API Client (`rag_api_client.py`)
- Handles all API communication with the RAG system
- Configurable client credentials
- Robust error handling

### 3. Data Parser (`data_parser.py`)
- Parses RAG API responses
- Extracts source context from responses

### 4. RAG Evaluator (`rag_evaluator.py`)
- Evaluates responses using Azure OpenAI
- Relevance evaluation
- Groundedness evaluation
- Lazy initialization for performance

### 5. Main Pipeline (`main.py`)
- Orchestrates the complete evaluation process
- Supports single question and batch evaluation
- Comprehensive result formatting

## Usage

### Basic Usage

```python
from main import RAGEvaluationPipeline

# Initialize pipeline
pipeline = RAGEvaluationPipeline()

# Evaluate a single question
result = pipeline.evaluate_question("What is consumer segmentation?")

# Check for errors
if "error" not in result:
    print(f"Relevance: {result['evaluation']['relevance']}")
    print(f"Groundedness: {result['evaluation']['groundedness']}")
```

### Batch Evaluation

```python
questions = [
    "Consumer segmentation",
    "What is market research?",
    "How do I analyze customer data?"
]

results = pipeline.evaluate_batch(questions)
```

### Advanced Usage

```python
from rag_api_client import RAGApiClient
from rag_evaluator import RAGEvaluator
from main import RAGEvaluationPipeline

# Custom configuration
api_client = RAGApiClient(
    api_url="https://custom-endpoint.com/api",
    client_principal_id="custom-id"
)

evaluator = RAGEvaluator(
    azure_openai_service="custom-service",
    deployment_name="custom-deployment"
)

# Initialize pipeline with custom components
pipeline = RAGEvaluationPipeline(
    api_client=api_client,
    evaluator=evaluator
)
```

## Environment Variables

Create a `.env` file with the following variables:

```
AZURE_OPENAI_SERVICE=your-service-name
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create `.env` file)

3. Run the evaluation:
```bash
python main.py
```

## Benefits of the New Architecture

### 1. **Simplicity**
- **Simplified evaluation flow** - uses original question directly for evaluation
- **Reduced complexity** - no rewritten query extraction or processing
- **Straightforward pipeline** - question → answer → evaluation

### 2. **Separation of Concerns**
- Each module has a single responsibility
- API logic separated from parsing and evaluation
- Configuration centralized

### 3. **Testability**
- Components can be tested independently
- Mock objects can be easily injected
- Clear interfaces between components

### 4. **Maintainability**
- Easier to modify individual components
- Clear code organization
- Better error handling throughout

### 5. **Flexibility**
- Components can be swapped or extended
- Custom configurations supported
- Easy to add new evaluation metrics

### 6. **Reusability**
- Components can be imported and used independently
- Pipeline can be embedded in other applications
- Clean API interfaces

## Migration from Original Script

The original `rag_eval.py` functionality is preserved but simplified:

```python
# Old way (still works)
from rag_eval import main
main()

# New way (recommended)
from main import RAGEvaluationPipeline
pipeline = RAGEvaluationPipeline()
result = pipeline.evaluate_question("Your question")
```

**Key simplification:** The system now uses the original question directly for evaluation instead of extracting and using rewritten queries, making the evaluation flow more straightforward.

## Error Handling

The system includes comprehensive error handling:
- API connection errors
- JSON parsing errors
- Azure OpenAI evaluation errors
- Configuration validation errors

All errors are returned in a consistent format with descriptive messages.

## Example Output

```json
{
  "question": "Consumer segmentation",
  "answer": "Consumer segmentation is the practice of...",
  "sources": "Context from relevant documents...",
  "evaluation": {
    "relevance": {"score": 0.95},
    "groundedness": {"score": 0.89}
  }
}
``` 