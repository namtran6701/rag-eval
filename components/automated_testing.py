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
from components.utils import get_batch_data
from indicators import ragas_evaluate

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
        
    # Single question testing section
    st.subheader("🔬 Single Question Testing")
    
    # Create question selector
    question_options = [f"Q{i+1}: {questions[i][:80]}..." for i in range(len(questions))]
    selected_question_idx = st.selectbox(
        "Select a question to test:",
        options=range(len(questions)),
        format_func=lambda x: question_options[x] if x < len(question_options) else f"Q{x+1}",
        index=0
    )
    
    if st.button("Test selected question"):
        if len(questions) > 0 and selected_question_idx < len(questions):
            with st.spinner(f"Testing question {selected_question_idx + 1}..."):
                # Get individual test result (without RAGAS)
                test_result = pipeline.automated_evaluate_question(questions[selected_question_idx], answers[selected_question_idx], sources[selected_question_idx])
                
                # Perform RAGAS evaluation for the single question
                if "error" not in test_result:
                    st.info("🎯 Running RAGAS evaluation...")
                    try:
                        # Run batch RAGAS evaluation with single question
                        ragas_result = ragas_evaluate(
                            [test_result['question_text']], 
                            [test_result['answer']], 
                            [test_result['expected_answer']], 
                            [test_result['sources']]
                        )
                        
                        # Map RAGAS results back to the test result
                        if len(ragas_result.scores) > 0:
                            ragas_score = ragas_result.scores[0]
                            test_result['ragas_metrics'] = {
                                "context_recall": ragas_score.get('context_recall', 0.0),
                                "faithfulness": ragas_score.get('faithfulness', 0.0),
                                "factual_correctness": ragas_score.get('factual_correctness', 0.0),
                                "pending": False  # Mark as completed
                            }
                        else:
                            test_result['ragas_metrics'] = {
                                "context_recall": 0.0,
                                "faithfulness": 0.0,
                                "factual_correctness": 0.0,
                                "error": "No RAGAS scores returned"
                            }
                        
                        st.success("✅ RAGAS evaluation completed!")
                        
                    except Exception as e:
                        st.warning(f"⚠️ RAGAS evaluation failed: {str(e)}")
                        test_result['ragas_metrics'] = {
                            "context_recall": 0.0,
                            "faithfulness": 0.0,
                            "factual_correctness": 0.0,
                            "error": str(e)
                        }
            
            display_automated_test_results([test_result])
        
    
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
                        
                        # Get the actual question text for the index
                        if question_idx < len(questions):
                            question_text = questions[question_idx]
                            result = pipeline.automated_evaluate_question(question_text, answers[question_idx], sources[question_idx])
                            results.append(result)
                        else:
                            results.append({
                                "error": f"Question index {question_idx} out of range (max: {len(questions)-1})",
                                "question_index": question_idx
                            })
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(question_indices))
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                
                # Perform batch RAGAS evaluation for successful tests
                successful_results = [r for r in results if "error" not in r]
                if successful_results:
                    st.info("🎯 Running batch RAGAS evaluation...")
                    with st.spinner("Evaluating with RAGAS metrics..."):
                        try:
                            # Prepare data for RAGAS evaluation
                            ragas_questions = [r['question_text'] for r in successful_results]
                            ragas_answers = [r['answer'] for r in successful_results]
                            ragas_expected_answers = [r['expected_answer'] for r in successful_results]
                            ragas_sources = [r['sources'] for r in successful_results]
                            
                            # Run batch RAGAS evaluation
                            ragas_result = ragas_evaluate(ragas_questions, ragas_answers, ragas_expected_answers, ragas_sources)
                            
                            # Map RAGAS results back to individual test results
                            for i, result in enumerate(successful_results):
                                if i < len(ragas_result.scores):
                                    ragas_score = ragas_result.scores[i]
                                    result['ragas_metrics'] = {
                                        "context_recall": ragas_score.get('context_recall', 0.0),
                                        "faithfulness": ragas_score.get('faithfulness', 0.0),
                                        "factual_correctness": ragas_score.get('factual_correctness', 0.0),
                                        "pending": False  # Mark as completed
                                    }
                                else:
                                    # Fallback if RAGAS results are shorter than expected
                                    result['ragas_metrics'] = {
                                        "context_recall": 0.0,
                                        "faithfulness": 0.0,
                                        "factual_correctness": 0.0,
                                        "error": "RAGAS result index out of range"
                                    }
                            
                            st.success("✅ RAGAS evaluation completed successfully!")
                            
                        except Exception as e:
                            st.warning(f"⚠️ RAGAS evaluation failed: {str(e)}")
                            # Set error state for all successful results
                            for result in successful_results:
                                result['ragas_metrics'] = {
                                    "context_recall": 0.0,
                                    "faithfulness": 0.0,
                                    "factual_correctness": 0.0,
                                    "error": str(e)
                                }
                
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

def display_automated_test_results(results: List[Dict[str, Any]]):
    """Display optimized results from automated testing with space-efficient layout"""
    if not results:
        st.info("No automated test results to display.")
        return
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .comparison-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #e9ecef;
    }
    .metric-good { color: #28a745; font-weight: bold; }
    .metric-fair { color: #ffc107; font-weight: bold; }
    .metric-poor { color: #dc3545; font-weight: bold; }
    .text-comparison {
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.4;
    }
    .similarity-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        margin: 2px;
    }
    .sim-excellent { background-color: #d4edda; color: #155724; }
    .sim-good { background-color: #fff3cd; color: #856404; }
    .sim-poor { background-color: #f8d7da; color: #721c24; }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("🧪 Automated Testing Results")
    
    # Calculate summary metrics
    successful_tests = [r for r in results if "error" not in r]
    failed_tests = [r for r in results if "error" in r]
    
    # Summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Tests", len(results))
    with col2:
        st.metric("Successful", len(successful_tests), delta=f"{len(successful_tests)/len(results)*100:.1f}%")
    with col3:
        st.metric("Failed", len(failed_tests), delta=f"-{len(failed_tests)/len(results)*100:.1f}%" if failed_tests else "0%")
    with col4:
        if successful_tests:
            avg_similarity = np.mean([r['answer_similarity_metrics']['average_similarity'] for r in successful_tests])
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
        else:
            st.metric("Avg Similarity", "N/A")
    with col5:
        if successful_tests:
            # Only calculate average for completed RAGAS evaluations
            completed_ragas = [r for r in successful_tests if not r.get('ragas_metrics', {}).get('pending', False) and 'error' not in r.get('ragas_metrics', {})]
            if completed_ragas:
                avg_faithfulness = np.mean([r.get('ragas_metrics', {}).get('faithfulness', 0.0) for r in completed_ragas])
                st.metric("Avg Faithfulness", f"{avg_faithfulness:.3f}")
            else:
                pending_count = len([r for r in successful_tests if r.get('ragas_metrics', {}).get('pending', False)])
                if pending_count > 0:
                    st.metric("Avg Faithfulness", f"Pending ({pending_count})")
                else:
                    st.metric("Avg Faithfulness", "N/A")
        else:
            st.metric("Avg Faithfulness", "N/A")
    with col6:
        if successful_tests:
            # Only calculate average for completed RAGAS evaluations
            completed_ragas = [r for r in successful_tests if not r.get('ragas_metrics', {}).get('pending', False) and 'error' not in r.get('ragas_metrics', {})]
            if completed_ragas:
                avg_factual = np.mean([r.get('ragas_metrics', {}).get('factual_correctness', 0.0) for r in completed_ragas])
                st.metric("Avg Factual Correctness", f"{avg_factual:.3f}")
            else:
                pending_count = len([r for r in successful_tests if r.get('ragas_metrics', {}).get('pending', False)])
                if pending_count > 0:
                    st.metric("Avg Factual Correctness", f"Pending ({pending_count})")
                else:
                    st.metric("Avg Factual Correctness", "N/A")
        else:
            st.metric("Avg Factual Correctness", "N/A")
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "📈 Similarity Analysis", "📏 Length Analysis", "🎯 RAGAS Analysis", "🔍 Detailed Results", "📋 Side-by-Side Comparison"])
    
    with tab1:
        if successful_tests:
            # Create overview dataframe
            overview_data = []
            for i, result in enumerate(successful_tests):
                ragas_metrics = result.get('ragas_metrics', {})
                
                # Handle RAGAS metrics display
                if ragas_metrics.get('pending', False):
                    context_recall = "Pending"
                    faithfulness = "Pending"
                    factual_correctness = "Pending"
                elif 'error' in ragas_metrics:
                    context_recall = "Error"
                    faithfulness = "Error"
                    factual_correctness = "Error"
                else:
                    context_recall = f"{ragas_metrics.get('context_recall', 0.0):.3f}"
                    faithfulness = f"{ragas_metrics.get('faithfulness', 0.0):.3f}"
                    factual_correctness = f"{ragas_metrics.get('factual_correctness', 0.0):.3f}"
                
                overview_data.append({
                    "Question #": i + 1,
                    "Avg Similarity": f"{result['answer_similarity_metrics']['average_similarity']:.3f}",
                    "Length Ratio": f"{result['length_metrics']['length_ratio']:.3f}",
                    "Word Ratio": f"{result['length_metrics']['word_ratio']:.3f}",
                    "Cosine Sim": f"{result['answer_similarity_metrics']['cosine_similarity']:.3f}",
                    "Sequence Sim": f"{result['answer_similarity_metrics']['sequence_similarity']:.3f}",
                    "Word Overlap": f"{result['answer_similarity_metrics']['word_overlap']:.3f}",
                    "Context Recall": context_recall,
                    "Faithfulness": faithfulness,
                    "Factual Correctness": factual_correctness
                })
            
            df_overview = pd.DataFrame(overview_data)
            st.dataframe(df_overview, use_container_width=True, height=300)
            
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
            similarities = [r['answer_similarity_metrics']['average_similarity'] for r in successful_tests]
            
            fig_hist = px.histogram(
                x=similarities,
                nbins=20,
                title="Distribution of Average Similarity Scores",
                labels={"x": "Average Similarity Score", "y": "Count"}
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Similarity types comparison
            similarity_types = []
            for i, result in enumerate(successful_tests):
                answer_sim_metrics = result['answer_similarity_metrics']
                similarity_types.append({
                    "Question": i + 1,
                    "Answer Sequence": answer_sim_metrics['sequence_similarity'],
                    "Answer Cosine": answer_sim_metrics['cosine_similarity'],
                    "Answer Word Overlap": answer_sim_metrics['word_overlap']
                })
            
            df_sim = pd.DataFrame(similarity_types)
            fig_line = px.line(
                df_sim.melt(id_vars=['Question'], var_name='Similarity Type', value_name='Score'),
                x='Question', y='Score', color='Similarity Type',
                title="Similarity Scores by Type Across Questions"
            )
            fig_line.update_layout(height=400)
            st.plotly_chart(fig_line, use_container_width=True)
    
    with tab3:
        if successful_tests:
            # Length analysis
            length_data = []
            for i, result in enumerate(successful_tests):
                length_metrics = result['length_metrics']
                length_data.append({
                    "Question": i + 1,
                    "Expected Length": length_metrics['text1_length'],
                    "Generated Length": length_metrics['text2_length'],
                    "Length Ratio": length_metrics['length_ratio'],
                    "Length Difference %": length_metrics['length_diff_percentage']
                })
            
            df_length = pd.DataFrame(length_data)
            
            # Length comparison scatter plot
            fig_scatter = px.scatter(
                df_length,
                x="Expected Length",
                y="Generated Length",
                color="Length Ratio",
                title="Answer Length Comparison",
                hover_data=["Question", "Length Difference %"]
            )
            fig_scatter.add_shape(
                type="line",
                x0=0, y0=0,
                x1=max(df_length["Expected Length"].max(), df_length["Generated Length"].max()),
                y1=max(df_length["Expected Length"].max(), df_length["Generated Length"].max()),
                line=dict(dash="dash", color="red"),
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Length ratio distribution
            fig_ratio = px.histogram(
                df_length,
                x="Length Ratio",
                nbins=20,
                title="Distribution of Length Ratios"
            )
            fig_ratio.update_layout(height=300)
            st.plotly_chart(fig_ratio, use_container_width=True)
    
    with tab4:
        # RAGAS Analysis Tab
        if successful_tests:
            st.subheader("🎯 RAGAS Metrics Analysis")
            
            # RAGAS metrics distribution
            ragas_data = []
            for i, result in enumerate(successful_tests):
                ragas_metrics = result.get('ragas_metrics', {})
                if 'error' not in ragas_metrics and not ragas_metrics.get('pending', False):
                    ragas_data.append({
                        "Question": i + 1,
                        "Context Recall": ragas_metrics.get('context_recall', 0.0),
                        "Faithfulness": ragas_metrics.get('faithfulness', 0.0),
                        "Factual Correctness": ragas_metrics.get('factual_correctness', 0.0)
                    })
            
            if ragas_data:
                df_ragas = pd.DataFrame(ragas_data)
                
                # RAGAS metrics line chart
                fig_ragas_line = px.line(
                    df_ragas.melt(id_vars=['Question'], var_name='RAGAS Metric', value_name='Score'),
                    x='Question', y='Score', color='RAGAS Metric',
                    title="RAGAS Metrics Across Questions",
                    labels={"Score": "RAGAS Score (0-1)", "Question": "Question Number"}
                )
                fig_ragas_line.update_layout(height=400)
                st.plotly_chart(fig_ragas_line, use_container_width=True)
                
                # RAGAS metrics distribution histograms
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_context = px.histogram(
                        df_ragas, x="Context Recall", nbins=10,
                        title="Context Recall Distribution"
                    )
                    fig_context.update_layout(height=300)
                    st.plotly_chart(fig_context, use_container_width=True)
                
                with col2:
                    fig_faith = px.histogram(
                        df_ragas, x="Faithfulness", nbins=10,
                        title="Faithfulness Distribution"
                    )
                    fig_faith.update_layout(height=300)
                    st.plotly_chart(fig_faith, use_container_width=True)
                
                with col3:
                    fig_factual = px.histogram(
                        df_ragas, x="Factual Correctness", nbins=10,
                        title="Factual Correctness Distribution"
                    )
                    fig_factual.update_layout(height=300)
                    st.plotly_chart(fig_factual, use_container_width=True)
                
                # RAGAS vs Similarity correlation
                similarity_scores = [r['answer_similarity_metrics']['average_similarity'] for r in successful_tests if 'error' not in r.get('ragas_metrics', {}) and not r.get('ragas_metrics', {}).get('pending', False)]
                faithfulness_scores = [r.get('ragas_metrics', {}).get('faithfulness', 0.0) for r in successful_tests if 'error' not in r.get('ragas_metrics', {}) and not r.get('ragas_metrics', {}).get('pending', False)]
                
                if len(similarity_scores) == len(faithfulness_scores) and len(similarity_scores) > 0:
                    correlation_data = pd.DataFrame({
                        "Similarity Score": similarity_scores,
                        "Faithfulness": faithfulness_scores,
                        "Question": list(range(1, len(similarity_scores) + 1))
                    })
                    
                    fig_corr = px.scatter(
                        correlation_data,
                        x="Similarity Score", y="Faithfulness",
                        title="Similarity vs RAGAS Faithfulness Correlation",
                        hover_data=["Question"]
                    )
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Calculate and display correlation coefficient
                    correlation = np.corrcoef(similarity_scores, faithfulness_scores)[0, 1]
                    st.info(f"📊 Correlation between Similarity and Faithfulness: {correlation:.3f}")
            
            else:
                # Check if RAGAS evaluations are pending or have errors
                pending_ragas = [r for r in successful_tests if r.get('ragas_metrics', {}).get('pending', False)]
                error_ragas = [r.get('ragas_metrics', {}).get('error') for r in successful_tests if 'error' in r.get('ragas_metrics', {})]
                
                if pending_ragas:
                    st.info(f"⏳ {len(pending_ragas)} RAGAS evaluations are pending. Run the automated testing to complete them.")
                elif error_ragas:
                    st.warning("⚠️ RAGAS evaluation errors occurred.")
                    st.error("RAGAS Evaluation Errors:")
                    for i, error in enumerate(error_ragas):
                        if error:
                            st.write(f"Question {i+1}: {error}")
                else:
                    st.warning("⚠️ No valid RAGAS metrics found.")
    
    with tab5:
        # Compact detailed individual results
        st.subheader("Individual Test Results")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            similarity_threshold = st.slider("Min Similarity Score", 0.0, 1.0, 0.0, 0.1, key="similarity_filter")
        with col2:
            show_count = st.selectbox("Show Results", [5, 10, 20, "All"], index=1, key="show_count_filter")
        
        # Filter results
        filtered_results = [r for r in successful_tests if r['answer_similarity_metrics']['average_similarity'] >= similarity_threshold]
        
        if show_count != "All":
            filtered_results = filtered_results[:show_count]
        
        for i, result in enumerate(filtered_results):
            question_preview = result['question_text'][:80] + "..." if len(result['question_text']) > 80 else result['question_text']
            
            # Get similarity color
            avg_sim = result['answer_similarity_metrics']['average_similarity']
            sim_color = "🟢" if avg_sim >= 0.7 else "🟡" if avg_sim >= 0.5 else "🔴"
            
            with st.expander(f"{sim_color} Q{i+1} (Sim: {avg_sim:.3f}): {question_preview}", expanded=False):
                # Compact metrics display
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.markdown("**📊 Similarity**")
                    answer_sim = result['answer_similarity_metrics']
                    st.write(f"• Avg: {answer_sim['average_similarity']:.3f}")
                    st.write(f"• Cosine: {answer_sim['cosine_similarity']:.3f}")
                    st.write(f"• Sequence: {answer_sim['sequence_similarity']:.3f}")
                    st.write(f"• Word Overlap: {answer_sim['word_overlap']:.3f}")
                
                with metrics_col2:
                    st.markdown("**📏 Length**")
                    len_metrics = result['length_metrics']
                    st.write(f"• Ratio: {len_metrics['length_ratio']:.3f}")
                    st.write(f"• Word Ratio: {len_metrics['word_ratio']:.3f}")
                    st.write(f"• Diff %: {len_metrics['length_diff_percentage']:.1f}%")
                    st.write(f"• Expected: {len_metrics['text1_length']} chars")
                    st.write(f"• Generated: {len_metrics['text2_length']} chars")
                
                with metrics_col3:
                    st.markdown("**🎯 RAGAS**")
                    ragas_metrics = result.get('ragas_metrics', {})
                    if ragas_metrics.get('pending', False):
                        st.write("⏳ Pending evaluation...")
                    elif 'error' in ragas_metrics:
                        st.write(f"❌ Error: {ragas_metrics['error'][:30]}...")
                    else:
                        st.write(f"• Context Recall: {ragas_metrics.get('context_recall', 0.0):.3f}")
                        st.write(f"• Faithfulness: {ragas_metrics.get('faithfulness', 0.0):.3f}")
                        st.write(f"• Factual Correct: {ragas_metrics.get('factual_correctness', 0.0):.3f}")
                
                with metrics_col4:
                    st.markdown("**🎯 Samples**")
                    sample_names = ["Beginning", "Middle", "End"]
                    for sample in result['sample_similarities']:
                        st.write(f"• {sample_names[sample['sample_index']]}: {sample['similarity']:.3f}")
                
                # Show full texts in collapsible sections
                if st.checkbox(f"Show full content for Q{i+1}", key=f"show_content_{i}"):
                    st.markdown("**❓ Question:**")
                    st.text_area("", result['question_text'], height=60, disabled=True, key=f"question_{i}", label_visibility="collapsed")
                    
                    content_col1, content_col2 = st.columns(2)
                    with content_col1:
                        st.markdown("**📝 Expected Answer:**")
                        st.text_area("", result['expected_answer'], height=200, disabled=True, key=f"expected_{i}", label_visibility="collapsed")
                    
                    with content_col2:
                        st.markdown("**🤖 Generated Answer:**")
                        st.text_area("", result['answer'], height=200, disabled=True, key=f"generated_{i}", label_visibility="collapsed")
                    
                    if result.get('expected_sources') or result.get('sources'):
                        sources_col1, sources_col2 = st.columns(2)
                        with sources_col1:
                            st.markdown("**📚 Expected Sources:**")
                            st.text_area("", result.get('expected_sources', 'N/A'), height=100, disabled=True, key=f"exp_sources_{i}", label_visibility="collapsed")
                        
                        with sources_col2:
                            st.markdown("**🔗 Generated Sources:**")
                            st.text_area("", result.get('sources', 'N/A'), height=100, disabled=True, key=f"gen_sources_{i}", label_visibility="collapsed")
    
    with tab6:
        # New side-by-side comparison tab
        st.subheader("📋 Side-by-Side Text Comparison")
        
        if successful_tests:
            # Question selector
            question_options = [f"Q{i+1}: {result['question_text'][:50]}..." for i, result in enumerate(successful_tests)]
            selected_idx = st.selectbox("Select question to compare:", range(len(question_options)), format_func=lambda x: question_options[x])
            
            if selected_idx is not None:
                result = successful_tests[selected_idx]
                
                # Display metrics for selected question
                st.markdown("**📊 Evaluation Metrics:**")
                metrics_row1, metrics_row2, metrics_row3, metrics_row4, metrics_row5, metrics_row6 = st.columns(6)
                
                with metrics_row1:
                    avg_sim = result['answer_similarity_metrics']['average_similarity']
                    sim_color = "#28a745" if avg_sim >= 0.7 else "#ffc107" if avg_sim >= 0.5 else "#dc3545"
                    st.metric("Average Similarity", f"{avg_sim:.3f}")
                
                with metrics_row2:
                    st.metric("Cosine Similarity", f"{result['answer_similarity_metrics']['cosine_similarity']:.3f}")
                
                with metrics_row3:
                    st.metric("Length Ratio", f"{result['length_metrics']['length_ratio']:.3f}")
                
                with metrics_row4:
                    st.metric("Word Overlap", f"{result['answer_similarity_metrics']['word_overlap']:.3f}")
                
                with metrics_row5:
                    ragas_metrics = result.get('ragas_metrics', {})
                    if ragas_metrics.get('pending', False):
                        faithfulness = "Pending"
                    else:
                        faithfulness = ragas_metrics.get('faithfulness', 0.0) if 'error' not in ragas_metrics else 0.0
                        faithfulness = f"{faithfulness:.3f}" if isinstance(faithfulness, (int, float)) else str(faithfulness)
                    st.metric("Faithfulness", faithfulness)
                
                with metrics_row6:
                    if ragas_metrics.get('pending', False):
                        factual_correctness = "Pending"
                    else:
                        factual_correctness = ragas_metrics.get('factual_correctness', 0.0) if 'error' not in ragas_metrics else 0.0
                        factual_correctness = f"{factual_correctness:.3f}" if isinstance(factual_correctness, (int, float)) else str(factual_correctness)
                    st.metric("Factual Correctness", factual_correctness)
                
                st.markdown("---")
                
                # Question display
                st.markdown("**❓ Question:**")
                st.info(result['question_text'])
                
                # Side-by-side answer comparison
                st.markdown("**📝 Answer Comparison:**")
                answer_col1, answer_col2 = st.columns(2)
                
                with answer_col1:
                    st.markdown("**Expected Answer**")
                    expected_length = len(result['expected_answer'])
                    st.caption(f"Length: {expected_length} characters, {len(result['expected_answer'].split())} words")
                    st.text_area("", result['expected_answer'], height=300, disabled=True, key=f"side_expected_{selected_idx}", label_visibility="collapsed")
                
                with answer_col2:
                    st.markdown("**Generated Answer**")
                    generated_length = len(result['answer'])
                    st.caption(f"Length: {generated_length} characters, {len(result['answer'].split())} words")
                    st.text_area("", result['answer'], height=300, disabled=True, key=f"side_generated_{selected_idx}", label_visibility="collapsed")
                
                # Sources comparison (if available)
                if result.get('expected_sources') or result.get('sources'):
                    st.markdown("**📚 Sources Comparison:**")
                    sources_col1, sources_col2 = st.columns(2)
                    
                    with sources_col1:
                        st.markdown("**Expected Sources**")
                        expected_sources = result.get('expected_sources', 'N/A')
                        if expected_sources != 'N/A':
                            st.caption(f"Length: {len(expected_sources)} characters")
                        st.text_area("", expected_sources, height=150, disabled=True, key=f"side_exp_sources_{selected_idx}", label_visibility="collapsed")
                    
                    with sources_col2:
                        st.markdown("**Generated Sources**")
                        generated_sources = result.get('sources', 'N/A')
                        if generated_sources != 'N/A':
                            st.caption(f"Length: {len(generated_sources)} characters")
                        st.text_area("", generated_sources, height=150, disabled=True, key=f"side_gen_sources_{selected_idx}", label_visibility="collapsed")
                
                # Text analysis section
                st.markdown("**🔍 Text Analysis:**")
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.markdown("**Length Comparison**")
                    length_diff = abs(len(result['expected_answer']) - len(result['answer']))
                    length_diff_pct = result['length_metrics']['length_diff_percentage']
                    st.write(f"• Difference: {length_diff} characters ({length_diff_pct:.1f}%)")
                    st.write(f"• Length ratio: {result['length_metrics']['length_ratio']:.3f}")
                    st.write(f"• Word ratio: {result['length_metrics']['word_ratio']:.3f}")
                
                with analysis_col2:
                    st.markdown("**Sample Similarities**")
                    sample_names = ["Beginning", "Middle", "End"]
                    for sample in result['sample_similarities']:
                        sample_sim = sample['similarity']
                        sample_emoji = "🟢" if sample_sim >= 0.7 else "🟡" if sample_sim >= 0.5 else "🔴"
                        st.write(f"{sample_emoji} {sample_names[sample['sample_index']]}: {sample_sim:.3f}")
    
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