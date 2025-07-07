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
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def evaluate_single_question_worker(question: str, pipeline) -> Dict[str, Any]:
    """
    Worker function for parallel evaluation of a single question
    
    Args:
        question: The question to evaluate
        pipeline: The RAG evaluation pipeline
        
    Returns:
        Dictionary containing evaluation result
    """
    try:
        return pipeline.evaluate_question(question)
    except Exception as e:
        return {
            "question": question,
            "error": str(e)
        }

def evaluate_questions_parallel(questions: List[str], pipeline, max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Evaluate multiple questions in parallel
    
    Args:
        questions: List of questions to evaluate
        pipeline: The RAG evaluation pipeline
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of evaluation results
    """
    results = []
    completed_count = 0
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a container for real-time results
    results_container = st.container()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_question = {
            executor.submit(evaluate_single_question_worker, question, pipeline): question 
            for question in questions
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_question):
            question = future_to_question[future]
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
                
                # Update progress
                progress = completed_count / len(questions)
                progress_bar.progress(progress)
                status_text.text(f"Completed: {completed_count}/{len(questions)} questions")
                
                # Show real-time completion status
                if completed_count <= 5:  # Show first 5 completions
                    with results_container:
                        if "error" not in result:
                            st.success(f"✅ Question {completed_count}: Evaluated successfully")
                        else:
                            st.error(f"❌ Question {completed_count}: {result['error']}")
                
            except Exception as e:
                # Handle any unexpected errors
                results.append({
                    "question": question,
                    "error": f"Unexpected error: {str(e)}"
                })
                completed_count += 1
                progress_bar.progress(completed_count / len(questions))
                status_text.text(f"Completed: {completed_count}/{len(questions)} questions")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    results_container.empty()
    
    # Sort results to match original question order
    question_to_result = {result.get('question', ''): result for result in results}
    ordered_results = []
    for question in questions:
        if question in question_to_result:
            ordered_results.append(question_to_result[question])
        else:
            # Fallback for any missing results
            ordered_results.append({
                "question": question,
                "error": "Result not found"
            })
    
    return ordered_results



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
    
    # Extract scores for display
    rel_score, ground_score = extract_scores_from_result(result)
    
    # Create expandable section for each result
    with st.expander(f"📝 Question {index + 1}: {result['question'][:100]}{'...' if len(result['question']) > 100 else ''}", expanded=True):
        
        # Display scores at the top
        score_col1, score_col2 = st.columns(2)
        
        with score_col1:
            rel_color = get_score_color(rel_score)
            st.markdown(f"**🎯 Relevance Score:** <span style='color: {rel_color}; font-weight: bold; font-size: 1.2em;'>{rel_score}/5</span>", unsafe_allow_html=True)
        
        with score_col2:
            ground_color = get_score_color(ground_score)
            st.markdown(f"**📚 Groundedness Score:** <span style='color: {ground_color}; font-weight: bold; font-size: 1.2em;'>{ground_score}/5</span>", unsafe_allow_html=True)
        
        st.markdown("---")  # Separator line
        
        # Display question and answer
        st.write("**Question:**")
        st.write(result['question'])
        
        st.write("**Answer:**")
        st.write(result['answer'])
        
        # Display sources
        if result.get('sources'):
            st.write("**Context:**")
            st.text_area(
                "Sources",
                result['sources'], 
                height=100, 
                key=f"sources_{uuid.uuid4().hex[:8]}", 
                label_visibility="hidden"
            )

def calculate_batch_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for batch evaluation results
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary containing various statistics
    """
    if not results:
        return {}
    
    relevance_scores = []
    groundedness_scores = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    for result in results:
        if "error" in result:
            failed_evaluations += 1
            continue
            
        successful_evaluations += 1
        rel_score, ground_score = extract_scores_from_result(result)
        relevance_scores.append(rel_score)
        groundedness_scores.append(ground_score)
    
    stats = {
        'total_questions': len(results),
        'successful_evaluations': successful_evaluations,
        'failed_evaluations': failed_evaluations,
        'success_rate': (successful_evaluations / len(results)) * 100 if results else 0
    }
    
    if relevance_scores:
        stats.update({
            'relevance_avg': sum(relevance_scores) / len(relevance_scores),
            'relevance_min': min(relevance_scores),
            'relevance_max': max(relevance_scores),
            'relevance_scores': relevance_scores
        })
    
    if groundedness_scores:
        stats.update({
            'groundedness_avg': sum(groundedness_scores) / len(groundedness_scores),
            'groundedness_min': min(groundedness_scores),
            'groundedness_max': max(groundedness_scores),
            'groundedness_scores': groundedness_scores
        })
    
    return stats

def display_batch_overview(stats: Dict[str, Any]):
    """Display overview statistics for batch evaluation"""
    st.subheader("Batch Evaluation Overview")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Questions", 
            value=stats.get('total_questions', 0)
        )
    
    with col2:
        st.metric(
            label="Successful Evaluations", 
            value=stats.get('successful_evaluations', 0)
        )
    
    with col3:
        success_rate = stats.get('success_rate', 0)
        st.metric(
            label="Success Rate", 
            value=f"{success_rate:.1f}%"
        )
    
    with col4:
        if stats.get('failed_evaluations', 0) > 0:
            st.metric(
                label="Failed Evaluations", 
                value=stats.get('failed_evaluations', 0)
            )
        else:
            st.metric(
                label="Failed Evaluations", 
                value="0"
            )

def display_batch_scores(stats: Dict[str, Any]):
    """Display score statistics for batch evaluation"""
    if not stats.get('relevance_scores') and not stats.get('groundedness_scores'):
        st.warning("No score data available for visualization")
        return
    
    st.subheader("🎯 Score Analysis")
    
    # Score summary in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Relevance Scores")
        if stats.get('relevance_scores'):
            avg_score = stats.get('relevance_avg', 0)
            min_score = stats.get('relevance_min', 0)
            max_score = stats.get('relevance_max', 0)
            
            score_color = get_score_color(avg_score)
            st.markdown(f"<h2 style='color: {score_color};'>Average: {avg_score:.1f}/5</h2>", unsafe_allow_html=True)
            st.write(f"**Range:** {min_score} - {max_score}")
            
            # Score distribution
            relevance_counts = {i: stats['relevance_scores'].count(i) for i in range(1, 6)}
            for score, count in relevance_counts.items():
                if count > 0:
                    percentage = (count / len(stats['relevance_scores'])) * 100
                    st.write(f"Score {score}: {count} questions ({percentage:.1f}%)")
    
    with col2:
        st.markdown("### Groundedness Scores")
        if stats.get('groundedness_scores'):
            avg_score = stats.get('groundedness_avg', 0)
            min_score = stats.get('groundedness_min', 0)
            max_score = stats.get('groundedness_max', 0)
            
            score_color = get_score_color(avg_score)
            st.markdown(f"<h2 style='color: {score_color};'>Average: {avg_score:.1f}/5</h2>", unsafe_allow_html=True)
            st.write(f"**Range:** {min_score} - {max_score}")
            
            # Score distribution
            groundedness_counts = {i: stats['groundedness_scores'].count(i) for i in range(1, 6)}
            for score, count in groundedness_counts.items():
                if count > 0:
                    percentage = (count / len(stats['groundedness_scores'])) * 100
                    st.write(f"Score {score}: {count} questions ({percentage:.1f}%)")

def create_batch_visualizations(results: List[Dict[str, Any]], stats: Dict[str, Any]):
    """Create comprehensive visualizations for batch results"""
    if not results or not stats.get('relevance_scores'):
        return
    
    st.subheader("📈 Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Score Trends", "Score Distribution"])
    
    with tab1:
        # Line chart showing scores across questions
        questions = [f"Q{i+1}" for i in range(len(results))]
        relevance_scores = stats.get('relevance_scores', [])
        groundedness_scores = stats.get('groundedness_scores', [])
        
        if relevance_scores and groundedness_scores:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=questions,
                y=relevance_scores,
                mode='lines+markers',
                name='Relevance',
                line=dict(color='lightblue', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=questions,
                y=groundedness_scores,
                mode='lines+markers',
                name='Groundedness',
                line=dict(color='lightgreen', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title='Score Trends Across Questions',
                xaxis_title='Questions',
                yaxis_title='Score (1-5)',
                height=400,
                yaxis=dict(range=[0, 5])
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Histogram of score distributions
        col1, col2 = st.columns(2)
        
        with col1:
            if stats.get('relevance_scores'):
                fig = px.histogram(
                    x=stats['relevance_scores'],
                    nbins=5,
                    title='Relevance Score Distribution',
                    labels={'x': 'Score', 'y': 'Count'},
                    color_discrete_sequence=['lightblue']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if stats.get('groundedness_scores'):
                fig = px.histogram(
                    x=stats['groundedness_scores'],
                    nbins=5,
                    title='Groundedness Score Distribution',
                    labels={'x': 'Score', 'y': 'Count'},
                    color_discrete_sequence=['lightgreen']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def display_batch_results(results: List[Dict[str, Any]]):
    """Display comprehensive batch evaluation results"""
    if not results:
        st.warning("No results to display")
        return
    
    # Calculate statistics
    stats = calculate_batch_statistics(results)
    
    # Display overview
    display_batch_overview(stats)
    
    # Display score analysis
    display_batch_scores(stats)
    
    # Display visualizations
    create_batch_visualizations(results, stats)
    
    # Detailed results section
    st.subheader("🔍 Detailed Question Results")
    
    # Add option to show/hide detailed results
    show_detailed = st.checkbox("Show detailed results for each question", value=False)
    
    if show_detailed:
        # Group results for better organization
        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]
        
        if successful_results:
            st.markdown("#### ✅ Successful Evaluations")
            for i, result in enumerate(successful_results):
                # Find original index
                original_index = results.index(result)
                display_single_result(result, original_index)
        
        if failed_results:
            st.markdown("#### ❌ Failed Evaluations")
            for i, result in enumerate(failed_results):
                original_index = results.index(result)
                with st.expander(f"❌ Question {original_index + 1}: Failed", expanded=False):
                    st.error(f"Error: {result['error']}")
                    if 'question' in result:
                        st.write(f"**Question:** {result['question']}")

def display_single_evaluation_results(results: List[Dict[str, Any]]):
    """Display results for single question evaluation"""
    if not results:
        return
    
    result = results[0]  # Single question result
    
    
    # Display metrics
    if "evaluation" in result:
        display_evaluation_metrics(result)
    
    # Display detailed result
    st.subheader("🔍 Detailed Result")
    display_single_result(result, 0)

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
        
        # Execution mode settings
        st.subheader("🚀 Execution Settings")
        
        execution_mode = st.radio(
            "Execution Mode",
            ["Parallel (3 workers)", "Sequential"],
            index=0,
            help="Parallel mode uses 3 workers for optimal speed without hitting API rate limits"
        )
        
        # Fixed number of workers
        max_workers = 3
        
        # Evaluate batch button
        if st.button("🚀 Evaluate Batch", type="primary"):
            if questions:
                start_time = time.time()
                
                try:
                    if execution_mode == "Parallel (3 workers)" and len(questions) > 1:
                        st.info(f"🔄 Running parallel evaluation with {max_workers} workers...")
                        results = evaluate_questions_parallel(
                            questions, 
                            st.session_state.pipeline, 
                            max_workers=max_workers
                        )
                    else:
                        # Sequential evaluation (fallback or single question)
                        st.info("🔄 Running sequential evaluation...")
                        with st.spinner(f"Evaluating {len(questions)} questions..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            results = []
                            for i, question in enumerate(questions):
                                status_text.text(f"Evaluating question {i+1}/{len(questions)}")
                                
                                result = st.session_state.pipeline.evaluate_question(question)
                                results.append(result)
                                
                                # Update progress
                                progress_bar.progress((i + 1) / len(questions))
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    
                    # Store results
                    st.session_state.evaluation_results = results
                    
                    # Display completion message with performance metrics
                    successful_count = len([r for r in results if "error" not in r])
                    failed_count = len(results) - successful_count
                    
                    # Create completion summary
                    completion_col1, completion_col2 = st.columns(2)
                    
                    with completion_col1:
                        st.metric(
                            "Execution Time",
                            f"{execution_time:.1f}s",
                            delta=f"~{execution_time/len(questions):.1f}s per question"
                        )
                    
                    with completion_col2:
                        st.metric(
                            "Success Rate",
                            f"{successful_count}/{len(questions)}",
                            delta=f"{(successful_count/len(questions)*100):.1f}%"
                        )
                    
                    if failed_count > 0:
                        st.warning(f"⚠️ {failed_count} questions failed to evaluate. Check detailed results below.")
                    
                    st.success(f"✅ Batch evaluation completed! Evaluated {len(questions)} questions in {execution_time:.1f} seconds.")
                    
                except Exception as e:
                    st.error(f"❌ Error during batch evaluation: {str(e)}")
            else:
                st.warning("⚠️ Please enter questions to evaluate.")
    
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
        st.subheader("💾 Download Results")
        
        # Prepare data for download
        download_data = []
        for i, result in enumerate(st.session_state.evaluation_results):
            if "error" not in result:
                rel_score, ground_score = extract_scores_from_result(result)
                download_data.append({
                    'question_number': i + 1,
                    'question': result.get('question', 'N/A'),
                    'answer': result.get('answer', 'N/A'),
                    'relevance_score': rel_score,
                    'groundedness_score': ground_score,
                    'sources': result.get('sources', 'N/A')
                })
            else:
                download_data.append({
                    'question_number': i + 1,
                    'question': result.get('question', 'N/A'),
                    'answer': 'Error',
                    'relevance_score': 'N/A',
                    'groundedness_score': 'N/A',
                    'sources': 'N/A',
                    'error': result.get('error', 'Unknown error')
                })
        
        if download_data:
            df_download = pd.DataFrame(download_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as CSV
                csv_data = df_download.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"rag_evaluation_results_{len(st.session_state.evaluation_results)}_questions.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download as JSON
                json_data = json.dumps(st.session_state.evaluation_results, indent=2)
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_data,
                    file_name=f"rag_evaluation_results_{len(st.session_state.evaluation_results)}_questions.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main() 