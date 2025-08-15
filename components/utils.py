"""
Utility Component

This component contains shared utility functions and constants used across components.
"""

import streamlit as st
import pandas as pd
import json
import ast


def parse_sources_to_formatted_list(sources_string: str) -> list:
    """
    Parse sources string and transform it into a formatted list of documents
    
    Args:
        sources_string: String representation of sources dictionary containing subqueries
        
    Returns:
        List of formatted strings with content, source, and separator line
    """
    if not sources_string or sources_string.strip() == "":
        return []
    
    # Clean the input string to handle common external service issues
    cleaned_string = sources_string.strip()
    
    # Remove common prefixes that external services might add
    prefixes_to_remove = [
        "/ Content:",
        "Content:",
        "Sources:",
        "Data:",
        "Response:"
    ]
    
    for prefix in prefixes_to_remove:
        if cleaned_string.startswith(prefix):
            cleaned_string = cleaned_string[len(prefix):].strip()
    
    try:
        # Strategy 1: Try to parse as JSON first
        try:
            sources_dict = json.loads(cleaned_string)
        except json.JSONDecodeError as json_error:
            
            # Strategy 2: Try to evaluate as Python literal
            try:
                sources_dict = ast.literal_eval(cleaned_string)
            except (ValueError, SyntaxError) as ast_error:
                
                # Strategy 3: Try to fix common JSON issues
                try:
                    # Replace single quotes with double quotes for JSON compatibility
                    json_fixed = cleaned_string.replace("'", '"')
                    
                    # Handle trailing commas
                    json_fixed = json_fixed.replace(',}', '}').replace(',]', ']')
                    
                    # Handle incomplete JSON by finding the last complete object
                    if json_fixed.count('{') != json_fixed.count('}'):
                        # Find the last complete closing brace
                        last_complete_brace = json_fixed.rfind('}')
                        if last_complete_brace > 0:
                            json_fixed = json_fixed[:last_complete_brace + 1]
                    
                    sources_dict = json.loads(json_fixed)
                    
                except json.JSONDecodeError as fixed_error:
                    
                    # Strategy 4: Try to extract partial data using regex
                    import re
                    
                    # Look for document patterns in the string
                    doc_pattern = r"'content':\s*'([^']*)',\s*'source':\s*'([^']*)'"
                    matches = re.findall(doc_pattern, cleaned_string)
                    
                    if matches:
                        formatted_documents = []
                        for content, source in matches:
                            formatted_doc = f"{content}\nSource: {source}\n\n{'=' * 30}\n\n"
                            formatted_documents.append(formatted_doc)
                        return formatted_documents
                    else:
                        # Strategy 5: Try to extract any content and source pairs we can find
                        
                        # Look for any content patterns
                        content_pattern = r"'content':\s*'([^']*)'"
                        source_pattern = r"'source':\s*'([^']*)'"
                        
                        content_matches = re.findall(content_pattern, cleaned_string)
                        source_matches = re.findall(source_pattern, cleaned_string)
                        
                        if content_matches and source_matches:
                            
                            # Pair them up (assuming they're in order)
                            formatted_documents = []
                            for i in range(min(len(content_matches), len(source_matches))):
                                content = content_matches[i]
                                source = source_matches[i]
                                formatted_doc = f"{content}\nSource: {source}\n{'=' * 30}\n"
                                formatted_documents.append(formatted_doc)
                            
                            if formatted_documents:
                                return formatted_documents
                        
                        # Strategy 6: Last resort - try to extract any readable content
                        
                        # Look for any text that looks like content
                        text_pattern = r"'([^']{50,})'"  # Any quoted string with at least 50 characters
                        text_matches = re.findall(text_pattern, cleaned_string)
                        
                        if text_matches:
                            formatted_documents = []
                            for i, text in enumerate(text_matches[:5]):  # Limit to first 5
                                formatted_doc = f"{text}\nSource: Unknown\n{'=' * 30}"
                                formatted_documents.append(formatted_doc)
                            return formatted_documents
                        
                        raise ValueError(f"Could not parse sources string using any method: {cleaned_string[:200]}...")
        
        formatted_documents = []
        
        # Iterate through each subquery in the sources dictionary
        for subquery_key, subquery_data in sources_dict.items():
            if isinstance(subquery_data, dict) and 'documents' in subquery_data:
                documents = subquery_data['documents']
                
                # Process each document in the subquery
                for doc in documents:
                    if isinstance(doc, dict) and 'content' in doc and 'source' in doc:
                        content = doc['content'].strip()
                        source = doc['source'].strip()
                        
                        # Format the document as requested
                        formatted_doc = f"{content}\nSource: {source}\n{'=' * 30} \n Content: \n"
                        formatted_documents.append(formatted_doc)
        
        return formatted_documents
        
    except Exception as e:
        # If parsing fails completely, return an error message
        return [f"Error parsing sources: {str(e)}\n{'=' * 30}"]


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


def display_indicator_interpretation_guide():
    """
    Display a comprehensive guide for interpreting evaluation indicators
    """
    st.header("📊 How to Interpret Evaluation Indicators")
    
    # Create tabs for different types of indicators
    tab1, tab2, tab3 = st.tabs(["📈 Similarity Metrics", "📏 Length Metrics", "🎯 Overall Guidance"])
    
    with tab1:
        st.subheader("Text Similarity Indicators")
        st.write("All similarity scores range from **0.0 to 1.0**, where higher values indicate better similarity.")
        
        # Similarity metrics table
        similarity_data = {
            "Metric": ["Sequence Similarity", "Cosine Similarity", "Word Overlap", "Average Similarity"],
            "Description": [
                "Character-level similarity between texts",
                "Semantic similarity using TF-IDF vectors", 
                "Percentage of shared words between texts",
                "Average of all similarity metrics"
            ],
            "Good (0.6-1.0)": ["✅ Very similar texts", "✅ Semantically similar", "✅ High word overlap", "✅ Overall high similarity"],
            "Fair (0.3-0.6)": ["⚠️ Moderate similarity", "⚠️ Some semantic overlap", "⚠️ Moderate overlap", "⚠️ Moderate similarity"],
            "Poor (0.0-0.3)": ["❌ Low similarity", "❌ Little semantic similarity", "❌ Minimal overlap", "❌ Low similarity"]
        }
        
        df_similarity = pd.DataFrame(similarity_data)
        st.dataframe(df_similarity, use_container_width=True)
        
        # Color-coded examples
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("**Good Scores (0.6-1.0)**\nIndicates strong alignment between expected and actual responses")
        with col2:
            st.warning("**Fair Scores (0.3-0.6)**\nSuggests partial alignment, may need review")
        with col3:
            st.error("**Poor Scores (0.0-0.3)**\nIndicates significant differences, requires attention")
    
    with tab2:
        st.subheader("Length Comparison Indicators")
        
        # Length metrics - Good indicators (higher is better)
        st.write("**📈 Higher is Better:**")
        length_good_data = {
            "Metric": ["Length Ratio", "Word Ratio"],
            "Range": ["0.0 - 1.0", "0.0 - 1.0"],
            "Description": [
                "Ratio of shorter text to longer text",
                "Ratio of fewer words to more words"
            ],
            "Good (0.8-1.0)": ["✅ Similar lengths", "✅ Similar word counts"],
            "Fair (0.5-0.8)": ["⚠️ Moderate difference", "⚠️ Moderate difference"],
            "Poor (0.0-0.5)": ["❌ Very different lengths", "❌ Very different word counts"]
        }
        
        df_length_good = pd.DataFrame(length_good_data)
        st.dataframe(df_length_good, use_container_width=True)
        
        # Length metrics - Bad indicators (lower is better)
        st.write("**📉 Lower is Better:**")
        length_bad_data = {
            "Metric": ["Length Difference %", "Length Difference"],
            "Range": ["0% - 100%+", "Absolute number"],
            "Description": [
                "Percentage difference in text lengths",
                "Absolute character difference between texts"
            ],
            "Good": ["✅ 0-20% difference", "✅ Small numbers relative to text size"],
            "Fair": ["⚠️ 20-50% difference", "⚠️ Moderate numbers"],
            "Poor": ["❌ 50%+ difference", "❌ Large numbers indicating very different lengths"]
        }
        
        df_length_bad = pd.DataFrame(length_bad_data)
        st.dataframe(df_length_bad, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Overall Interpretation Guidelines")
        
        # RAG-specific guidance
        st.write("**For RAG Evaluation Context:**")
        st.info("""
        • **Similarity scores above 0.6** generally indicate good alignment between expected and actual answers
        • **Length ratios above 0.7** suggest appropriately sized responses  
        • **Length difference percentages below 30%** indicate reasonable response sizing
        """)
        
        # Red flags
        st.write("**🚨 Red Flags to Watch For:**")
        st.error("""
        • **Average similarity below 0.3** suggests poor answer quality
        • **Length ratio below 0.3** might indicate truncated or overly verbose responses
        • **Length difference percentage above 70%** could indicate significant sizing issues
        """)
        
        # Best practices
        st.write("**💡 Best Practice Tips:**")
        st.success("""
        • Look at the **Average Similarity** score as your primary indicator
        • Cross-reference individual metrics to understand why a score might be high or low
        • **High word overlap but low cosine similarity** might indicate keyword matching without semantic understanding
        • **High sequence similarity but low word overlap** might indicate similar structure but different content
        """)
        
        # Score interpretation scale
        st.write("**📊 Quick Reference Scale:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🟢 Excellent (0.8-1.0)**
            - Very high similarity
            - Strong alignment
            - Minimal length differences
            """)
        
        with col2:
            st.markdown("""
            **🟡 Good (0.6-0.8)**
            - Good similarity
            - Reasonable alignment  
            - Acceptable length differences
            """)
        
        with col3:
            st.markdown("""
            **🔴 Needs Review (0.0-0.6)**
            - Low similarity
            - Poor alignment
            - Significant differences
            """)


if __name__ == "__main__":
    """
    Test the parse_sources_to_formatted_list function
    """
    import os
    
    # Path to the test file - test both malformed and well-formed data
    test_files = ["test_malformed_sources.txt", "test_sources_example.txt"]
    
    for test_file_path in test_files:
        print(f"\n{'='*60}")
        print(f"Testing with file: {test_file_path}")
        print(f"{'='*60}")
        
        try:
            # Check if test file exists
            if not os.path.exists(test_file_path):
                print(f"Test file {test_file_path} not found. Skipping...")
                continue
            else:
                # Read the test file
                with open(test_file_path, 'r') as file:
                    test_sources_string = file.read().strip()
            
            print("Testing parse_sources_to_formatted_list function...")
            print(f"Input string: {test_sources_string[:100]}...")
            print("-" * 50)
            
            # Test the function
            result = parse_sources_to_formatted_list(test_sources_string)
            
            print(f"Function returned {len(result)} formatted documents:")
            print("-" * 50)
            
            for i, doc in enumerate(result, 1):
                print(f"Document {i}:")
                print(doc)
                print()
            
            print("Test completed successfully!")
            
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")
