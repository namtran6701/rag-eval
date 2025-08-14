"""
Batch Question Evaluation Component

This component handles the batch question evaluation interface and logic.
"""

import streamlit as st
import pandas as pd
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from components.utils import extract_scores_from_result


def display_batch_evaluation_interface(pipeline, batch_questions: List[str]) -> List[Dict[str, Any]]:
    """
    Display the batch question evaluation interface
    
    Args:
        pipeline: The RAG evaluation pipeline
        batch_questions: List of pre-loaded batch questions
        
    Returns:
        List of evaluation results
    """
    st.header("📊 Batch Questions Evaluation")
    
    # Batch input options
    input_method = st.radio(
        "How would you like to input questions?",
        ["Text Area", "Upload File", "Use Pre-loaded Data"] if batch_questions else ["Text Area", "Upload File"]
    )
    
    questions = []
    
    if input_method == "Use Pre-loaded Data":
        if batch_questions:
            st.success(f"✅ Using {len(batch_questions)} pre-loaded questions from batch_evaluation.csv")
            questions = batch_questions.copy()
            
            # Show preview of loaded data
            st.subheader("📋 Data Preview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Questions", len(batch_questions))
            with col2:
                st.metric("Answers", len(st.session_state.get('batch_answers', [])))
            with col3:
                st.metric("Sources", len(st.session_state.get('batch_sources', [])))
            
            # Show sample data
            if st.checkbox("Show sample data"):
                sample_idx = st.selectbox("Select sample:", range(min(10, len(questions))))
                st.write("**Sample Question:**")
                st.text_area("Question:", batch_questions[sample_idx], height=100, disabled=True)
                st.write("**Sample Answer:**")
                sample_answer = st.session_state.get('batch_answers', [])[sample_idx] if sample_idx < len(st.session_state.get('batch_answers', [])) else "No answer available"
                st.text_area("Answer:", sample_answer[:500] + "..." if len(sample_answer) > 500 else sample_answer, height=150, disabled=True)
                st.write("**Sample Sources:**")
                sample_sources = st.session_state.get('batch_sources', [])[sample_idx] if sample_idx < len(st.session_state.get('batch_sources', [])) else "No sources available"
                st.text_area("Sources:", sample_sources[:300] + "..." if len(sample_sources) > 300 else sample_sources, height=100, disabled=True)
        else:
            st.error("❌ No pre-loaded data available. Please check if the CSV file exists.")
    
    elif input_method == "Text Area":
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
                        pipeline, 
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
                            
                            result = pipeline.evaluate_question(question)
                            results.append(result)
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(questions))
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
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
                
                return results
                
            except Exception as e:
                st.error(f"❌ Error during batch evaluation: {str(e)}")
                return []
        else:
            st.warning("⚠️ Please enter questions to evaluate.")
            return []
    
    return []


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


from components.utils import extract_scores_from_result
