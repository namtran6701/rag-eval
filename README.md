# RAG Evaluation System

A comprehensive Python system for evaluating Retrieval-Augmented Generation (RAG) responses using Azure OpenAI, featuring both a web interface and programmatic API.

## Features

✨ **Web Interface** - Interactive Streamlit app for easy evaluation  
🚀 **Parallel Processing** - Batch evaluation with 3-worker parallel execution  
📊 **Rich Analytics** - Comprehensive statistics and visualizations  
📁 **File Upload** - Support for .txt and .csv file batch processing  
🎯 **Dual Metrics** - Relevance and groundedness scoring  
💾 **Export Options** - Download results as CSV or JSON  

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file with your Azure OpenAI credentials:

```env
AZURE_OPENAI_SERVICE=your-service-name
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
```

### 3. Launch Web Interface

```bash
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501` to access the interactive evaluation interface.

## Web Interface Usage

### Single Question Evaluation

1. Select "Single Question" mode
2. Enter your question in the text area
3. Click "🚀 Evaluate Question"
4. View relevance and groundedness scores with detailed reasoning

### Batch Evaluation

1. Select "Batch Questions" mode
2. Choose input method:
   - **Text Area**: Enter questions (one per line)
   - **File Upload**: Upload .txt or .csv file
3. Select execution mode:
   - **Parallel (3 workers)**: Fast evaluation using 3 concurrent workers
   - **Sequential**: One-by-one evaluation
4. Click "🚀 Evaluate Batch"
5. View comprehensive analytics and download results

## File Upload Formats

### Text File Format (.txt)

Create a plain text file with one question per line:

```
What is consumer segmentation?
How does market analysis work?
What are the benefits of customer analytics?
What is the difference between qualitative and quantitative research?
How do you calculate customer lifetime value?
```

**Requirements:**
- UTF-8 encoding
- One question per line
- Empty lines are automatically ignored
- Leading/trailing spaces are trimmed

### CSV File Format (.csv)

Create a CSV file with questions in the first column:

```csv
Question
What is consumer segmentation?
How does market analysis work?
What are the benefits of customer analytics?
```

**Requirements:**
- Questions must be in the first column
- Header row is optional
- Other columns are ignored
- Empty rows are automatically skipped

### Example Files

**questions.txt:**
```
What is machine learning?
How does artificial intelligence work?
What are the applications of deep learning?
```

**questions.csv:**
```csv
Questions,Priority,Category
What is machine learning?,High,AI
How does artificial intelligence work?,Medium,AI
What are the applications of deep learning?,High,AI
```

## Architecture Overview

The system follows a clean, modular architecture:

```
├── streamlit_app.py        # Web interface (main entry point)
├── main.py                 # Evaluation pipeline
├── config.py              # Configuration management
├── rag_api_client.py       # RAG API client
├── data_parser.py          # Response parsing logic
├── rag_evaluator.py        # Azure OpenAI evaluation
└── requirements.txt        # Dependencies
```

## Programmatic Usage

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
    "What is consumer segmentation?",
    "How does market research work?",
    "What are customer analytics benefits?"
]

# Evaluate all questions
results = []
for question in questions:
    result = pipeline.evaluate_question(question)
    results.append(result)
```

### Advanced Configuration

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

## Web Interface Features

### 📊 Batch Analytics Dashboard

- **Overview Metrics**: Total questions, success rate, execution time
- **Score Analysis**: Average scores, ranges, and distributions
- **Visualizations**: 
  - Score trends across questions
  - Score distribution histograms
- **Detailed Results**: Individual question scores and reasoning

### 🚀 Parallel Processing

- **3-Worker Parallel Execution**: Optimal balance of speed and reliability
- **Real-time Progress**: Live updates as questions complete
- **Performance Metrics**: Execution time and throughput tracking
- **Error Handling**: Individual question failures don't stop the batch

### 💾 Export & Download

- **CSV Export**: Structured data for spreadsheet analysis
- **JSON Export**: Complete data for programmatic processing
- **Result Summary**: Success rates and performance metrics

## Components

### 1. Web Interface (`streamlit_app.py`)
- Interactive evaluation interface
- Batch processing with parallel execution
- Rich analytics and visualizations
- File upload capabilities

### 2. Main Pipeline (`main.py`)
- Orchestrates the complete evaluation process
- Handles single question and batch processing
- Comprehensive result formatting

### 3. Configuration (`config.py`)
- Centralized configuration management
- Environment variable handling
- Configuration validation

### 4. RAG API Client (`rag_api_client.py`)
- Handles all API communication with the RAG system
- Configurable client credentials
- Robust error handling

### 5. Data Parser (`data_parser.py`)
- Parses RAG API responses
- Extracts source context from responses

### 6. RAG Evaluator (`rag_evaluator.py`)
- Evaluates responses using Azure OpenAI
- Relevance and groundedness scoring
- Lazy initialization for performance

## Error Handling

The system includes comprehensive error handling:
- API connection errors
- File upload and parsing errors
- Azure OpenAI evaluation errors
- Configuration validation errors
- Parallel processing error isolation

All errors are returned in a consistent format with descriptive messages.

## Example Output

### Single Question Result
```json
{
  "question": "What is consumer segmentation?",
  "answer": "Consumer segmentation is the practice of dividing customers...",
  "sources": "Context from relevant documents...",
  "evaluation": {
    "relevance": {
      "relevance": 4,
      "relevance_reason": "The answer directly addresses the question..."
    },
    "groundedness": {
      "groundedness": 5,
      "groundedness_reason": "The response is fully supported by sources..."
    }
  }
}
```

### Batch Results Summary
- Total Questions: 10
- Successful Evaluations: 9
- Success Rate: 90%
- Average Relevance Score: 4.2/5
- Average Groundedness Score: 4.1/5
- Execution Time: 12.3 seconds

## Best Practices

### File Upload
- Test with small files first (2-3 questions)
- Use UTF-8 encoding for text files
- Keep questions clear and concise
- Preview questions before evaluation

### Parallel Processing
- Use parallel mode for batches of 3+ questions
- Monitor API rate limits with large batches
- Sequential mode for single questions or debugging

### Performance
- Parallel execution is ~3x faster than sequential
- Optimal batch size: 10-50 questions
- Consider API quotas for large batches

## Migration from Command Line

If migrating from the original command-line version:

```python
# Old way
from rag_eval import main
main()

# New way - Web Interface (recommended)
# Run: streamlit run streamlit_app.py

# New way - Programmatic
from main import RAGEvaluationPipeline
pipeline = RAGEvaluationPipeline()
result = pipeline.evaluate_question("Your question")
```

## Troubleshooting

### Common Issues

1. **Azure OpenAI Connection**: Verify environment variables in `.env`
2. **File Upload Errors**: Check file encoding (use UTF-8)
3. **Rate Limiting**: Reduce batch size or use sequential mode
4. **Memory Issues**: Process large files in smaller batches

### Support

For issues or questions:
1. Check error messages in the web interface
2. Verify configuration and file formats
3. Test with smaller batches first
4. Review logs for detailed error information 