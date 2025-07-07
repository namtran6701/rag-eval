#!/usr/bin/env python3
"""
Streamlit App for RAG Evaluation System

This app provides a web interface for evaluating RAG (Retrieval-Augmented Generation) responses.
It allows users to:
1. Input single questions or batch questions
2. Run evaluations using the RAG evaluation pipeline
3. View detailed results and metrics
4. Download evaluation results
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import uuid
from typing import Dict, Any, List

# Import your existing modules
from main import RAGEvaluationPipeline


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

def initialize_session_state():
    """Initialize session state variables"""
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None

def create_pipeline():
    """Create and cache the RAG evaluation pipeline"""
    if st.session_state.pipeline is None:
        try:
            st.session_state.pipeline = RAGEvaluationPipeline()
            return True
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {str(e)}")
            return False
    return True



def get_score_color(score: Any) -> str:
    """
    Get color for score display based on score value
    
    Args:
        score: The score value
        
    Returns:
        CSS color string
    """
    if score == "N/A":
        return "gray"
    
    try:
        numeric_score = int(score)
        if numeric_score >= 4:
            return "green"
        elif numeric_score >= 3:
            return "orange"
        else:
            return "red"
    except (ValueError, TypeError):
        return "gray"

def display_score_metric(title: str, emoji: str, metric_data: Any, metric_name: str = None):
    """
    Display a single score metric with consistent formatting
    
    Args:
        title: The title for the metric
        emoji: The emoji to display
        metric_data: The metric data (dict or direct value)
        metric_name: The name of the metric (e.g., "relevance", "groundedness")
    """
    st.subheader(f"{emoji} {title}")
    
    if isinstance(metric_data, dict) and metric_name:
        # Handle nested structure: metric_data[metric_name] contains the evaluation results
        metric_results = metric_data.get(metric_name, {})
        
        score = "N/A"
        reason = "No reason provided"
        
        # Extract score from nested structure
        if isinstance(metric_results, dict):
            # Try different score key formats within the nested results
            possible_score_keys = [
                metric_name,                    # "relevance", "groundedness"
                f"gpt_{metric_name}",          # "gpt_relevance", "gpt_groundedness"
                "score",                       # generic "score"
            ]
            
            for key in possible_score_keys:
                if key in metric_results:
                    score = metric_results[key]
                    break
            
            # Extract reason from nested structure
            possible_reason_keys = [
                f"{metric_name}_reason",       # "relevance_reason", "groundedness_reason"
                "reason",                      # generic "reason"
            ]
            
            for key in possible_reason_keys:
                if key in metric_results:
                    reason = metric_results[key]
                    break
        

        # Ensure score is numeric and convert to int
        if score != "N/A":
            try:
                score = int(score)
            except (ValueError, TypeError):
                score = "N/A"
        
        # Display score with color coding
        score_color = get_score_color(score)
        if score != "N/A":
            st.markdown(f"<h2 style='color: {score_color};'>{score}/5</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: {score_color};'>{score}</h2>", unsafe_allow_html=True)
        
        st.write("**Reasoning:**")
        st.write(reason)
    else:
        st.write("**Note:** Unable to display metric - invalid format")

def extract_scores_from_result(result: Dict[str, Any]) -> tuple:
    """
    Extract relevance and groundedness scores from a result
    
    Args:
        result: The evaluation result
        
    Returns:
        Tuple of (relevance_score, groundedness_score)
    """
    if "error" in result or "evaluation" not in result:
        return 0, 0
    
    eval_data = result["evaluation"]
    
    # Extract from nested structure
    rel_data = eval_data.get("relevance", {})
    ground_data = eval_data.get("groundedness", {})
    
    # Extract scores from nested dictionaries
    rel_score = 0
    ground_score = 0
    
    if isinstance(rel_data, dict):
        rel_score = rel_data.get("relevance", 0)
    
    if isinstance(ground_data, dict):
        ground_score = ground_data.get("groundedness", 0)
    
    # Convert to numeric, default to 0 if not convertible
    try:
        rel_score = int(rel_score) if rel_score is not None else 0
    except (ValueError, TypeError):
        rel_score = 0
        
    try:
        ground_score = int(ground_score) if ground_score is not None else 0
    except (ValueError, TypeError):
        ground_score = 0
    
    return rel_score, ground_score

def display_evaluation_metrics(result: Dict[str, Any]):
    """Display evaluation metrics in a formatted way"""
    if "evaluation" not in result:
        st.warning("⚠️ No evaluation data found in result")
        return
    
    evaluation = result["evaluation"]
    

    # Create columns for metrics
    col1, col2 = st.columns(2)
    
    with col1:
        display_score_metric("Relevance Score", "🎯", evaluation, "relevance")
    
    with col2:
        display_score_metric("Groundedness Score", "📚", evaluation, "groundedness")

def display_single_result(result: Dict[str, Any], index: int = 0):
    """Display a single evaluation result"""
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    # Create expandable section for each result
    with st.expander(f"📝 Question {index + 1}: {result['question'][:100]}{'...' if len(result['question']) > 100 else ''}", expanded=True):
        
        # Display question and answer
        st.write("**Question:**")
        st.write(result['question'])
        
        st.write("**Answer:**")
        st.write(result['answer'])
        
        # Display sources
        if result.get('sources'):
            st.write("**Sources:**")
            st.text_area(
                "Sources",
                result['sources'], 
                height=100, 
                key=f"sources_{uuid.uuid4().hex[:8]}", 
                label_visibility="hidden"
            )

def create_summary_chart(results: List[Dict[str, Any]]):
    """Create a summary chart of evaluation results"""
    if not results:
        return
    
    # Extract scores for visualization
    relevance_scores = []
    groundedness_scores = []
    questions = []
    
    for i, result in enumerate(results):
        rel_score, ground_score = extract_scores_from_result(result)
        relevance_scores.append(rel_score)
        groundedness_scores.append(ground_score)
        questions.append(f"Q{i+1}")
    
    if relevance_scores and groundedness_scores:
        # Create a grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Relevance',
            x=questions,
            y=relevance_scores,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Groundedness',
            x=questions,
            y=groundedness_scores,
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title='Evaluation Scores Overview',
            xaxis_title='Questions',
            yaxis_title='Score (1-5)',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🔍 RAG Evaluation System</div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Initialize pipeline
    if not create_pipeline():
        st.stop()
    
    # Evaluation mode selection
    eval_mode = st.sidebar.radio(
        "Select Evaluation Mode:",
        ["Single Question", "Batch Questions"],
        help="Choose whether to evaluate one question or multiple questions"
    )
    
    # Main content area
    if eval_mode == "Single Question":
        st.header("📝 Single Question Evaluation")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is consumer segmentation?"
        )
        
        # Evaluate button
        if st.button("🚀 Evaluate Question", type="primary"):
            if question.strip():
                with st.spinner("Evaluating your question..."):
                    try:
                        result = st.session_state.pipeline.evaluate_question(
                            question.strip()
                        )
                        
                        # Store result
                        st.session_state.evaluation_results = [result]
                        
                        # Success message only - results will be displayed in the results section
                        st.success("✅ Evaluation completed!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during evaluation: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question to evaluate.")
    
    else:  # Batch Questions
        st.header("📊 Batch Questions Evaluation")
        
        # Batch input options
        input_method = st.radio(
            "How would you like to input questions?",
            ["Text Area", "Upload File"]
        )
        
        questions = []
        
        if input_method == "Text Area":
            questions_text = st.text_area(
                "Enter questions (one per line):",
                height=200,
                placeholder="What is consumer segmentation?\nHow does market analysis work?\nWhat are the benefits of customer analytics?"
            )
            
            if questions_text:
                questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        
        else:  # Upload File
            uploaded_file = st.file_uploader(
                "Upload a file with questions",
                type=['txt', 'csv'],
                help="Upload a .txt file with one question per line, or a .csv file with questions in the first column"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.txt'):
                        content = uploaded_file.read().decode('utf-8')
                        questions = [q.strip() for q in content.split('\n') if q.strip()]
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        questions = df.iloc[:, 0].dropna().tolist()
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Show questions preview
        if questions:
            st.write(f"**Found {len(questions)} questions:**")
            for i, q in enumerate(questions[:5]):  # Show first 5
                st.write(f"{i+1}. {q}")
            if len(questions) > 5:
                st.write(f"... and {len(questions) - 5} more")
        
        # Evaluate batch button
        if st.button("🚀 Evaluate Batch", type="primary"):
            if questions:
                with st.spinner(f"Evaluating {len(questions)} questions..."):
                    try:
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        for i, question in enumerate(questions):
                            status_text.text(f"Evaluating question {i+1}/{len(questions)}")
                            
                            result = st.session_state.pipeline.evaluate_question(
                                question
                            )
                            results.append(result)
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(questions))
                        
                        # Store results
                        st.session_state.evaluation_results = results
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success(f"✅ Batch evaluation completed! Evaluated {len(questions)} questions.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during batch evaluation: {str(e)}")
            else:
                st.warning("⚠️ Please enter questions to evaluate.")
    
    # Display results if available
    if st.session_state.evaluation_results:
        st.header("📊 Results")
        
        # Display evaluation metrics for the first result (or aggregate for batch)
        if st.session_state.evaluation_results:
            first_result = st.session_state.evaluation_results[0]
            if "evaluation" in first_result:
                display_evaluation_metrics(first_result)
        
        # Detailed results
        st.subheader("🔍 Detailed Results")
        for i, result in enumerate(st.session_state.evaluation_results):
            display_single_result(result, i)

if __name__ == "__main__":
    main() 