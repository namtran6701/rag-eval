"""
Results Display Component

This component handles the display of evaluation results, batch results, and download functionality.
"""

import streamlit as st
import pandas as pd
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
from components.utils import get_score_color, extract_scores_from_result


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


def display_single_result(result: Dict[str, Any], index: int = 0):
    """Display a single evaluation result"""
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    # Extract scores for display
    rel_score, ground_score = extract_scores_from_result(result)
    
    # Extract reasoning from evaluation data
    rel_reason = "No reasoning provided"
    ground_reason = "No reasoning provided"
    
    if "evaluation" in result:
        evaluation = result["evaluation"]
        
        # Extract relevance reasoning
        if "relevance" in evaluation:
            rel_data = evaluation["relevance"]
            if isinstance(rel_data, dict):
                rel_reason = rel_data.get("relevance_reason", "No reasoning provided")
        
        # Extract groundedness reasoning
        if "groundedness" in evaluation:
            ground_data = evaluation["groundedness"]
            if isinstance(ground_data, dict):
                ground_reason = ground_data.get("groundedness_reason", "No reasoning provided")
    
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
        
        # Display reasoning for each score
        reasoning_col1, reasoning_col2 = st.columns(2)
        
        with reasoning_col1:
            st.markdown("**Reasoning:**")
            st.write(rel_reason)
        
        with reasoning_col2:
            st.markdown("**Reasoning:**")
            st.write(ground_reason)
        
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
                key=f"sources_{index}", 
                label_visibility="hidden"
            )


from components.utils import extract_scores_from_result


from components.utils import get_score_color


def display_download_section(results: List[Dict[str, Any]]):
    """Display download section for results"""
    if not results:
        return
    
    st.subheader("💾 Download Results")
    
    # Prepare data for download
    download_data = []
    for i, result in enumerate(results):
        if "error" not in result:
            rel_score, ground_score = extract_scores_from_result(result)
            
            relevance_reason = "No reasoning provided"
            groundedness_reason = "No reasoning provided"
            
            if "evaluation" in result:
                evaluation = result["evaluation"]
                if "relevance" in evaluation and isinstance(evaluation["relevance"], dict):
                    relevance_reason = evaluation["relevance"].get("relevance_reason", "No reasoning provided")
                if "groundedness" in evaluation and isinstance(evaluation["groundedness"], dict):
                    groundedness_reason = evaluation["groundedness"].get("groundedness_reason", "No reasoning provided")
            
            download_data.append({
                'question_number': i + 1,
                'question': result.get('question', 'N/A'),
                'answer': result.get('answer', 'N/A'),
                'relevance_score': rel_score,
                'relevance_reason': relevance_reason,
                'groundedness_score': ground_score,
                'groundedness_reason': groundedness_reason,
                'sources': result.get('sources', 'N/A')
            })
        else:
            download_data.append({
                'question_number': i + 1,
                'question': result.get('question', 'N/A'),
                'answer': 'Error',
                'relevance_score': 'N/A',
                'relevance_reason': 'N/A',
                'groundedness_score': 'N/A',
                'groundedness_reason': 'N/A',
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
                file_name=f"rag_evaluation_results_{len(results)}_questions.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download as JSON
            json_data = json.dumps(results, indent=2)
            st.download_button(
                label="📋 Download as JSON",
                data=json_data,
                file_name=f"rag_evaluation_results_{len(results)}_questions.json",
                mime="application/json"
            )
