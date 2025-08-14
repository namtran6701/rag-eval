"""
Utility Component

This component contains shared utility functions and constants used across components.
"""

import streamlit as st


def get_score_color(score: any) -> str:
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


def extract_scores_from_result(result: dict) -> tuple:
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


def display_score_metric(title: str, emoji: str, metric_data: any, metric_name: str = None):
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


def get_batch_data():
    """
    Get the loaded batch evaluation data from session state
    
    Returns:
        tuple: (questions, answers, sources) arrays
    """
    return (
        st.session_state.get('batch_questions', []),
        st.session_state.get('batch_answers', []),
        st.session_state.get('batch_sources', [])
    )
