#!/usr/bin/env python3
"""
Refactored Streamlit App for RAG Evaluation System

This app provides a web interface for evaluating RAG (Retrieval-Augmented Generation) responses.
It uses a modular component-based architecture for better maintainability.
"""

import streamlit as st
import logging

# Configure logging to show in console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import components
from components.data_loader import initialize_session_state, create_pipeline, display_sidebar_info
from components.single_evaluation import display_single_evaluation_interface, display_single_evaluation_results
from components.batch_evaluation import display_batch_evaluation_interface
from components.automated_testing import display_automated_testing_interface, display_automated_test_results
from components.results_display import display_batch_results, display_download_section
from components.utils import display_indicator_interpretation_guide


# Page configuration
st.set_page_config(
    page_title="RAG Evaluation System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🔍 RAG Evaluation System</div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", help="Show debug information in the app")
    if debug_mode:
        st.session_state.debug_mode = True
        logger.info("Debug mode enabled")
    else:
        st.session_state.debug_mode = False
    
    # Display batch data information in sidebar
    display_sidebar_info()
    
    # Initialize pipeline
    if not create_pipeline():
        st.stop()
    
    # Evaluation mode selection
    eval_mode = st.sidebar.radio(
        "Select Evaluation Mode:",
        ["Single Question", "Batch Questions", "Automated Testing", "📊 Indicator Guide"],
        help="Choose whether to evaluate one question, multiple questions, run automated testing, or view the indicator interpretation guide"
    )
    
    # Main content area
    if eval_mode == "Single Question":
        # Single question evaluation
        results = display_single_evaluation_interface(st.session_state.pipeline)
        if results:
            st.session_state.evaluation_results = results
    
    elif eval_mode == "Batch Questions":
        # Batch question evaluation
        results = display_batch_evaluation_interface(st.session_state.pipeline, st.session_state.batch_questions)
        if results:
            st.session_state.evaluation_results = results
    
    elif eval_mode == "Automated Testing":
        # Automated testing
        results = display_automated_testing_interface(st.session_state.pipeline)
        if results:
            st.session_state.automated_test_results = results
    
    elif eval_mode == "📊 Indicator Guide":
        # Display indicator interpretation guide
        display_indicator_interpretation_guide()
    
    # Display results if available
    if st.session_state.evaluation_results:
        st.header("📊 Results")
        
        # Determine if this is a single question or batch evaluation
        is_batch_evaluation = len(st.session_state.evaluation_results) > 1
        
        if is_batch_evaluation:
            # Display comprehensive batch results
            display_batch_results(st.session_state.evaluation_results)
        else:
            # Display single question results
            display_single_evaluation_results(st.session_state.evaluation_results)
        
        # Download results option
        display_download_section(st.session_state.evaluation_results)
    
    # Display automated test results if available
    if st.session_state.automated_test_results:
        st.markdown("---")  # Separator
        display_automated_test_results(st.session_state.automated_test_results)


if __name__ == "__main__":
    main()
