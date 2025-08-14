"""
Data Loader Component

This component handles data loading, session state management, and pipeline initialization.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List


def load_batch_evaluation_data():
    """
    Load batch evaluation data from CSV file
    
    Returns:
        tuple: (questions, answers, sources) arrays
    """
    try:
        csv_path = "src/answers/batch_evaluation.csv"
        df = pd.read_csv(csv_path, sep=';')
        
        # Drop rows where any of the essential columns are missing
        df_clean = df.dropna(subset=['question', 'answer', 'sources'])
        
        # Extract arrays from the cleaned CSV - now all arrays will have the same length
        questions = df_clean['question'].tolist()
        answers = df_clean['answer'].tolist()
        sources = df_clean['sources'].tolist()
        
        # Debug info
        st.info(f"Loaded {len(questions)} complete records from CSV")
        
        return questions, answers, sources
    except Exception as e:
        st.error(f"Error loading batch evaluation data: {str(e)}")
        return [], [], []


def initialize_session_state():
    """Initialize session state variables"""
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'batch_questions' not in st.session_state:
        st.session_state.batch_questions = []
    if 'batch_answers' not in st.session_state:
        st.session_state.batch_answers = []
    if 'batch_sources' not in st.session_state:
        st.session_state.batch_sources = []
    if 'automated_test_results' not in st.session_state:
        st.session_state.automated_test_results = []
    
    # Load batch evaluation data if not already loaded
    if not st.session_state.batch_questions:
        questions, answers, sources = load_batch_evaluation_data()
        st.session_state.batch_questions = questions
        st.session_state.batch_answers = answers
        st.session_state.batch_sources = sources


def create_pipeline():
    """Create and cache the RAG evaluation pipeline"""
    if st.session_state.pipeline is None:
        try:
            from main import RAGEvaluationPipeline
            st.session_state.pipeline = RAGEvaluationPipeline()
            return True
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {str(e)}")
            return False
    return True


def display_sidebar_info():
    """Display batch data information in sidebar"""
    if st.session_state.batch_questions:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Loaded Batch Data")
        st.sidebar.info(f"**Questions loaded:** {len(st.session_state.batch_questions)}")
        st.sidebar.info(f"**Answers loaded:** {len(st.session_state.batch_answers)}")
        st.sidebar.info(f"**Sources loaded:** {len(st.session_state.batch_sources)}")
        
        # Show sample question
        if st.sidebar.checkbox("Show sample question"):
            sample_idx = st.sidebar.selectbox("Select sample:", range(min(5, len(st.session_state.batch_questions))))
            st.sidebar.text_area("Sample Question:", st.session_state.batch_questions[sample_idx], height=100, disabled=True)
