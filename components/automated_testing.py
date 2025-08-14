"""
Automated Testing Component

This component handles the automated testing interface and logic.
"""

import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from components.utils import get_batch_data


def display_automated_testing_interface(pipeline) -> List[Dict[str, Any]]:
    """
    Display the automated testing interface
    
    Args:
        pipeline: The RAG evaluation pipeline
        
    Returns:
        List of automated test results
    """
    st.header("🧪 Automated Testing")
    
    # Get batch data
    questions, answers, sources = get_batch_data()
    
    if not questions:
        st.error("❌ No batch data available. Please ensure the batch_evaluation.csv file is loaded.")
        return []
    
    # Show data loading status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Questions Loaded", len(questions))
    with col2:
        st.metric("Answers Loaded", len(answers))
    with col3:
        st.metric("Sources Loaded", len(sources))
    
    # Check for data consistency
    if not (len(questions) == len(answers) == len(sources)):
        st.error(f"⚠️ Data inconsistency detected! Questions: {len(questions)}, Answers: {len(answers)}, Sources: {len(sources)}")
        return []
    
    st.success(f"✅ Ready to test against {len(questions)} complete records from batch_evaluation.csv")
    
    # Testing configuration
    st.subheader("⚙️ Test Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        # Question range selection
        max_questions = len(questions)
        test_range = st.select_slider(
            "Select question range to test:",
            options=["First 5", "First 10", "First 20", "First 50", "All Questions"],
            value="First 5"
        )
        
        # Convert selection to indices
        range_mapping = {
            "First 5": list(range(min(5, max_questions))),
            "First 10": list(range(min(10, max_questions))),
            "First 20": list(range(min(20, max_questions))),
            "First 50": list(range(min(50, max_questions))),
            "All Questions": list(range(max_questions))
        }
        question_indices = range_mapping[test_range]
        
    with col2:
        # Execution settings
        execution_mode = st.radio(
            "Execution Mode:",
            ["Sequential"],
            index=0,
            help="Sequential mode is faster but uses more resources"
        )
        
        max_workers = 1
    
    # Show test preview
    st.subheader("📋 Test Preview")
    st.info(f"Will test {len(question_indices)} questions with {execution_mode.lower()} execution")
    
    # Show sample questions that will be tested
    if st.checkbox("Show questions to be tested"):
        sample_questions = question_indices[:5]  # Show first 5
        for i, idx in enumerate(sample_questions):
            if idx < len(questions):
                st.write(f"**{idx + 1}.** {questions[idx][:100]}...")
            else:
                st.error(f"Index {idx} is out of range (max: {len(questions)-1})")
        if len(question_indices) > 5:
            st.write(f"... and {len(question_indices) - 5} more questions")
    
    # Debug option
    if st.checkbox("Show debug information"):
        st.subheader("🔧 Debug Information")
        st.write("**Session State Data:**")
        st.write(f"- batch_questions length: {len(st.session_state.batch_questions)}")
        st.write(f"- batch_answers length: {len(st.session_state.batch_answers)}")
        st.write(f"- batch_sources length: {len(st.session_state.batch_sources)}")
        
        st.write("**Current Data:**")
        st.write(f"- questions length: {len(questions)}")
        st.write(f"- answers length: {len(answers)}")
        st.write(f"- sources length: {len(sources)}")
        
        st.write("**Selected Indices:**")
        st.write(f"- Range: {test_range}")
        st.write(f"- Indices: {question_indices[:10]}{'...' if len(question_indices) > 10 else ''}")
        st.write(f"- Max valid index: {len(questions) - 1}")
        
        # Test a single index
        if st.button("Test single index (0)"):
            if len(questions) > 0:
                test_result = evaluate_single_question_automated(0, pipeline)
                st.json(test_result)
    
    # Run automated testing
    if st.button("🚀 Start Automated Testing", type="primary"):
        if question_indices:
            start_time = time.time()
            
            st.info(f"🔄 Starting automated testing of {len(question_indices)} questions...")
            
            try:
                # Sequential execution
                st.info("🔄 Running sequential automated testing...")
                with st.spinner(f"Testing {len(question_indices)} questions..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    for i, question_idx in enumerate(question_indices):
                        status_text.text(f"Testing question {i+1}/{len(question_indices)}")
                        
                        result = evaluate_single_question_automated(question_idx, pipeline)
                        results.append(result)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(question_indices))
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                
                # Calculate execution time
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Success message
                successful_tests = len([r for r in results if "error" not in r])
                st.success(f"✅ Automated testing completed! {successful_tests}/{len(results)} tests successful in {execution_time:.2f} seconds")
                
                return results
                
            except Exception as e:
                st.error(f"❌ Error during automated testing: {str(e)}")
                return []
    
    return []


from components.utils import get_batch_data


def calculate_text_similarity(text1: str, text2: str) -> Dict[str, float]:
    """
    Calculate similarity between two texts using multiple methods
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Dictionary with different similarity scores
    """
    # Clean texts
    text1_clean = re.sub(r'\s+', ' ', text1.strip())
    text2_clean = re.sub(r'\s+', ' ', text2.strip())
    
    # Sequence similarity (character-based)
    sequence_similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()
    
    # TF-IDF Cosine similarity (word-based)
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cosine_sim = 0.0
    
    # Word overlap similarity
    words1 = set(text1_clean.lower().split())
    words2 = set(text2_clean.lower().split())
    word_overlap = len(words1.intersection(words2)) / max(len(words1.union(words2)), 1)
    
    return {
        "sequence_similarity": sequence_similarity,
        "cosine_similarity": cosine_sim,
        "word_overlap": word_overlap,
        "average_similarity": (sequence_similarity + cosine_sim + word_overlap) / 3
    }


def calculate_length_comparison(text1: str, text2: str) -> Dict[str, Any]:
    """
    Compare lengths of two texts
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Dictionary with length comparison metrics
    """
    len1 = len(text1)
    len2 = len(text2)
    
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0
    length_diff = abs(len1 - len2)
    length_diff_percentage = (length_diff / max(len1, len2)) * 100 if max(len1, len2) > 0 else 0
    
    # Word counts
    words1 = len(text1.split())
    words2 = len(text2.split())
    word_ratio = min(words1, words2) / max(words1, words2) if max(words1, words2) > 0 else 1.0
    
    return {
        "text1_length": len1,
        "text2_length": len2,
        "length_ratio": length_ratio,
        "length_difference": length_diff,
        "length_diff_percentage": length_diff_percentage,
        "text1_words": words1,
        "text2_words": words2,
        "word_ratio": word_ratio,
        "length_similarity_score": length_ratio  # Score based on length similarity
    }


def extract_text_samples(text: str, sample_size: int = 200) -> List[str]:
    """
    Extract arbitrary samples from text for comparison
    
    Args:
        text: Input text
        sample_size: Size of each sample
        
    Returns:
        List of text samples
    """
    if len(text) <= sample_size:
        return [text]
    
    samples = []
    # Beginning sample
    samples.append(text[:sample_size])
    
    # Middle sample
    mid_start = len(text) // 2 - sample_size // 2
    samples.append(text[mid_start:mid_start + sample_size])
    
    # End sample
    samples.append(text[-sample_size:])
    
    return samples


def evaluate_single_question_automated(question_idx: int, pipeline) -> Dict[str, Any]:
    """
    Automated evaluation of a single question from the batch data
    
    Args:
        question_idx: Index of the question in the batch arrays
        pipeline: RAG evaluation pipeline
        
    Returns:
        Dictionary containing comprehensive evaluation results
    """
    try:
        questions, answers, sources = get_batch_data()
        
        # Debug information
        debug_info = {
            "questions_length": len(questions),
            "answers_length": len(answers),
            "sources_length": len(sources),
            "requested_index": question_idx
        }
        
        # Check if arrays have different lengths
        if not (len(questions) == len(answers) == len(sources)):
            return {
                "error": f"Array length mismatch: questions={len(questions)}, answers={len(answers)}, sources={len(sources)}",
                "debug_info": debug_info
            }
        
        # Check if index is out of range
        if question_idx >= len(questions):
            return {
                "error": f"Question index {question_idx} out of range (max: {len(questions)-1})",
                "debug_info": debug_info
            }
        
        # Check if arrays are empty
        if len(questions) == 0:
            return {
                "error": "No questions loaded in batch data",
                "debug_info": debug_info
            }
        
        question = questions[question_idx]
        stored_answer = answers[question_idx]
        stored_sources = sources[question_idx]
        
        # Get new response from pipeline
        result = pipeline.evaluate_question(question)
        generated_answer = result.get('response', '')
        generated_sources = result.get('sources', '')
        
        # Perform similarity comparison
        similarity_metrics = calculate_text_similarity(stored_answer, generated_answer)
        
        # Perform length comparison
        length_metrics = calculate_length_comparison(stored_answer, generated_answer)
        
        # Extract and compare samples
        stored_samples = extract_text_samples(stored_answer)
        generated_samples = extract_text_samples(generated_answer)
        
        sample_similarities = []
        for i, (stored_sample, generated_sample) in enumerate(zip(stored_samples, generated_samples)):
            sample_sim = calculate_text_similarity(stored_sample, generated_sample)
            sample_similarities.append({
                "sample_index": i,
                "similarity": sample_sim["average_similarity"]
            })
        
        # Compile comprehensive results
        automated_result = {
            "question_index": question_idx,
            "question": question,
            "stored_answer": stored_answer,
            "generated_answer": generated_answer,
            "stored_sources": stored_sources,
            "generated_sources": generated_sources,
            "similarity_metrics": similarity_metrics,
            "length_metrics": length_metrics,
            "sample_similarities": sample_similarities,
            "evaluation": result.get('evaluation', {}),
            "timestamp": time.time()
        }
        
        return automated_result
        
    except Exception as e:
        return {
            "question_index": question_idx,
            "error": f"Automated evaluation failed: {str(e)}"
        }


def display_automated_test_results(results: List[Dict[str, Any]]):
    """Display results from automated testing"""
    if not results:
        st.info("No automated test results to display.")
        return
    
    st.subheader("🧪 Automated Testing Results")
    
    # Calculate summary metrics
    successful_tests = [r for r in results if "error" not in r]
    failed_tests = [r for r in results if "error" in r]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tests", len(results))
    with col2:
        st.metric("Successful", len(successful_tests), delta=f"{len(successful_tests)/len(results)*100:.1f}%")
    with col3:
        st.metric("Failed", len(failed_tests), delta=f"-{len(failed_tests)/len(results)*100:.1f}%" if failed_tests else "0%")
    with col4:
        if successful_tests:
            avg_similarity = np.mean([r['similarity_metrics']['average_similarity'] for r in successful_tests])
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
        else:
            st.metric("Avg Similarity", "N/A")
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Similarity Analysis", "📏 Length Analysis", "🔍 Detailed Results"])
    
    with tab1:
        if successful_tests:
            # Create overview dataframe
            overview_data = []
            for result in successful_tests:
                overview_data.append({
                    "Question #": result['question_index'] + 1,
                    "Avg Similarity": result['similarity_metrics']['average_similarity'],
                    "Length Ratio": result['length_metrics']['length_ratio'],
                    "Word Ratio": result['length_metrics']['word_ratio'],
                    "Cosine Similarity": result['similarity_metrics']['cosine_similarity'],
                    "Sequence Similarity": result['similarity_metrics']['sequence_similarity']
                })
            
            df_overview = pd.DataFrame(overview_data)
            st.dataframe(df_overview, use_container_width=True)
            
            # Download button for results
            csv_data = df_overview.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_data,
                file_name=f"automated_test_results_{int(time.time())}.csv",
                mime="text/csv"
            )
    
    with tab2:
        if successful_tests:
            # Similarity distribution
            similarities = [r['similarity_metrics']['average_similarity'] for r in successful_tests]
            
            fig_hist = px.histogram(
                x=similarities,
                nbins=20,
                title="Distribution of Average Similarity Scores",
                labels={"x": "Average Similarity Score", "y": "Count"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Similarity types comparison
            similarity_types = []
            for result in successful_tests:
                sim_metrics = result['similarity_metrics']
                similarity_types.append({
                    "Question": result['question_index'] + 1,
                    "Sequence": sim_metrics['sequence_similarity'],
                    "Cosine": sim_metrics['cosine_similarity'],
                    "Word Overlap": sim_metrics['word_overlap']
                })
            
            df_sim = pd.DataFrame(similarity_types)
            fig_line = px.line(
                df_sim.melt(id_vars=['Question'], var_name='Similarity Type', value_name='Score'),
                x='Question', y='Score', color='Similarity Type',
                title="Similarity Scores by Type Across Questions"
            )
            st.plotly_chart(fig_line, use_container_width=True)
    
    with tab3:
        if successful_tests:
            # Length analysis
            length_data = []
            for result in successful_tests:
                length_metrics = result['length_metrics']
                length_data.append({
                    "Question": result['question_index'] + 1,
                    "Stored Answer Length": length_metrics['text1_length'],
                    "Generated Answer Length": length_metrics['text2_length'],
                    "Length Ratio": length_metrics['length_ratio'],
                    "Length Difference %": length_metrics['length_diff_percentage']
                })
            
            df_length = pd.DataFrame(length_data)
            
            # Length comparison scatter plot
            fig_scatter = px.scatter(
                df_length,
                x="Stored Answer Length",
                y="Generated Answer Length",
                color="Length Ratio",
                title="Answer Length Comparison",
                hover_data=["Question", "Length Difference %"]
            )
            fig_scatter.add_shape(
                type="line",
                x0=0, y0=0,
                x1=max(df_length["Stored Answer Length"].max(), df_length["Generated Answer Length"].max()),
                y1=max(df_length["Stored Answer Length"].max(), df_length["Generated Answer Length"].max()),
                line=dict(dash="dash", color="red"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Length ratio distribution
            fig_ratio = px.histogram(
                df_length,
                x="Length Ratio",
                nbins=20,
                title="Distribution of Length Ratios"
            )
            st.plotly_chart(fig_ratio, use_container_width=True)
    
    with tab4:
        # Detailed individual results
        st.subheader("Individual Test Results")
        
        for i, result in enumerate(successful_tests[:10]):  # Show first 10 detailed results
            with st.expander(f"Question {result['question_index'] + 1}: {result['question'][:100]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📊 Similarity Metrics:**")
                    sim_metrics = result['similarity_metrics']
                    st.write(f"- Average: {sim_metrics['average_similarity']:.3f}")
                    st.write(f"- Sequence: {sim_metrics['sequence_similarity']:.3f}")
                    st.write(f"- Cosine: {sim_metrics['cosine_similarity']:.3f}")
                    st.write(f"- Word Overlap: {sim_metrics['word_overlap']:.3f}")
                    
                    st.write("**📏 Length Metrics:**")
                    len_metrics = result['length_metrics']
                    st.write(f"- Length Ratio: {len_metrics['length_ratio']:.3f}")
                    st.write(f"- Word Ratio: {len_metrics['word_ratio']:.3f}")
                    st.write(f"- Length Diff %: {len_metrics['length_diff_percentage']:.1f}%")
                
                with col2:
                    st.write("**🎯 Sample Similarities:**")
                    for sample in result['sample_similarities']:
                        sample_names = ["Beginning", "Middle", "End"]
                        st.write(f"- {sample_names[sample['sample_index']]}: {sample['similarity']:.3f}")
                
                # Show text comparison
                if st.checkbox(f"Show text comparison for Question {result['question_index'] + 1}"):
                    st.write("**📝 Stored Answer:**")
                    st.text_area("Stored", result['stored_answer'][:500] + "..." if len(result['stored_answer']) > 500 else result['stored_answer'], height=100, disabled=True, key=f"stored_{i}")
                    
                    st.write("**🤖 Generated Answer:**")
                    st.text_area("Generated", result['generated_answer'][:500] + "..." if len(result['generated_answer']) > 500 else result['generated_answer'], height=100, disabled=True, key=f"generated_{i}")
    
    # Show failed tests if any
    if failed_tests:
        st.subheader("❌ Failed Tests")
        for result in failed_tests:
            error_msg = result.get('error', 'Unknown error')
            question_num = result.get('question_index', 'Unknown') + 1 if 'question_index' in result else 'Unknown'
            st.error(f"Question {question_num}: {error_msg}")
            
            # Show debug info if available
            if 'debug_info' in result:
                with st.expander(f"Debug info for Question {question_num}"):
                    st.json(result['debug_info'])
