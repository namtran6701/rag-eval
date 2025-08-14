# RAG Evaluation System - Component Architecture

This document describes the modular component-based architecture of the RAG Evaluation System.

## Overview

The application has been refactored from a single monolithic file (`streamlit_app.py`) into a modular component-based architecture for better maintainability, reusability, and separation of concerns.

## File Structure

```
rag-eval/
├── streamlit_app.py                    # Original monolithic app
├── streamlit_app_refactored.py         # New modular main app
├── components/                         # Component modules
│   ├── __init__.py                     # Package initialization
│   ├── data_loader.py                  # Data loading and session state management
│   ├── single_evaluation.py            # Single question evaluation interface
│   ├── batch_evaluation.py             # Batch question evaluation interface
│   ├── automated_testing.py            # Automated testing interface
│   ├── results_display.py              # Results display and visualization
│   └── utils.py                        # Shared utility functions
└── ... (other existing files)
```

## Component Descriptions

### 1. `data_loader.py`
**Purpose**: Handles data loading, session state management, and pipeline initialization.

**Key Functions**:
- `initialize_session_state()`: Sets up Streamlit session state variables
- `create_pipeline()`: Creates and caches the RAG evaluation pipeline
- `load_batch_evaluation_data()`: Loads batch evaluation data from CSV
- `display_sidebar_info()`: Shows batch data information in sidebar

### 2. `single_evaluation.py`
**Purpose**: Manages single question evaluation interface and results display.

**Key Functions**:
- `display_single_evaluation_interface()`: Main interface for single question evaluation
- `display_single_evaluation_results()`: Displays results for single question
- `display_evaluation_metrics()`: Shows relevance and groundedness metrics
- `display_single_result()`: Displays detailed single result

### 3. `batch_evaluation.py`
**Purpose**: Handles batch question evaluation with parallel processing.

**Key Functions**:
- `display_batch_evaluation_interface()`: Main interface for batch evaluation
- `evaluate_questions_parallel()`: Parallel evaluation of multiple questions
- `evaluate_single_question_worker()`: Worker function for parallel processing
- `calculate_batch_statistics()`: Calculates statistics for batch results

### 4. `automated_testing.py`
**Purpose**: Manages automated testing against pre-loaded batch data.

**Key Functions**:
- `display_automated_testing_interface()`: Main interface for automated testing
- `evaluate_single_question_automated()`: Automated evaluation of single question
- `calculate_text_similarity()`: Text similarity calculations
- `display_automated_test_results()`: Displays automated test results

### 5. `results_display.py`
**Purpose**: Handles comprehensive results display and visualization.

**Key Functions**:
- `display_batch_results()`: Main function for displaying batch results
- `display_batch_overview()`: Shows overview statistics
- `display_batch_scores()`: Displays score analysis
- `create_batch_visualizations()`: Creates charts and graphs
- `display_download_section()`: Handles results download

### 6. `utils.py`
**Purpose**: Contains shared utility functions used across components.

**Key Functions**:
- `get_score_color()`: Returns color for score display
- `extract_scores_from_result()`: Extracts scores from evaluation results
- `display_score_metric()`: Displays score metrics consistently
- `get_batch_data()`: Gets batch data from session state

## Main Application

### `streamlit_app_refactored.py`
**Purpose**: Main application that orchestrates all components.

**Key Features**:
- Imports and uses all component modules
- Manages the main application flow
- Handles evaluation mode selection
- Coordinates between components

## Benefits of the New Architecture

### 1. **Modularity**
- Each component has a single responsibility
- Easy to understand and maintain individual components
- Components can be developed and tested independently

### 2. **Reusability**
- Shared utility functions prevent code duplication
- Components can be reused in different contexts
- Easy to extend with new functionality

### 3. **Maintainability**
- Smaller, focused files are easier to navigate
- Changes to one component don't affect others
- Clear separation of concerns

### 4. **Testability**
- Individual components can be unit tested
- Mock dependencies easily
- Isolated testing of specific functionality

### 5. **Scalability**
- Easy to add new evaluation modes
- Simple to extend with new visualization types
- Modular structure supports team development

## Usage

### Running the Refactored App
```bash
streamlit run streamlit_app_refactored.py
```

### Running the Original App
```bash
streamlit run streamlit_app.py
```

## Migration Guide

### From Monolithic to Component-Based

1. **Backup**: Keep the original `streamlit_app.py` as backup
2. **Test**: Run the refactored app to ensure functionality is preserved
3. **Compare**: Both apps should provide identical functionality
4. **Switch**: Use the refactored app for future development

### Adding New Features

1. **Identify**: Determine which component should handle the new feature
2. **Create**: Add new functions to the appropriate component
3. **Import**: Import the new functions in the main app
4. **Integrate**: Use the new functions in the main application flow

### Creating New Components

1. **Create**: Add a new Python file in the `components/` directory
2. **Document**: Add component description to this document
3. **Import**: Import the component in the main app
4. **Test**: Ensure the new component works correctly

## Validation

The refactored application maintains all the original functionality:

- ✅ Single question evaluation
- ✅ Batch question evaluation
- ✅ Automated testing
- ✅ Results visualization
- ✅ Data download functionality
- ✅ Session state management
- ✅ Error handling
- ✅ Progress tracking
- ✅ Parallel processing

## Future Enhancements

With the new architecture, it's easy to add:

- New evaluation metrics
- Additional visualization types
- Different data sources
- Custom evaluation modes
- API endpoints
- Database integration
- User authentication
- Multi-user support
