"""
Single Question Evaluation Component

This component handles the single question evaluation interface and logic.
"""

import streamlit as st
from typing import Dict, Any, List
from components.utils import get_score_color, extract_scores_from_result, display_score_metric


def display_single_evaluation_interface(pipeline) -> List[Dict[str, Any]]:
    """
    Display the single question evaluation interface
    
    Args:
        pipeline: The RAG evaluation pipeline
        
    Returns:
        List of evaluation results
    """
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
                    result = pipeline.evaluate_question(
                        question.strip()
                    )
                    
                    # Success message only - results will be displayed in the results section
                    st.success("✅ Evaluation completed!")
                    
                    return [result]
                    
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
                    return []
        else:
            st.warning("⚠️ Please enter a question to evaluate.")
            return []
    
    return []


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


from components.utils import display_score_metric


from components.utils import get_score_color


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
