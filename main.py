#!/usr/bin/env python3
"""
Main module for RAG evaluation system

This module orchestrates the complete RAG evaluation process:
1. Make API calls to the RAG system
2. Parse responses to extract answer, rewritten query, and sources
3. Evaluate the response using Azure OpenAI evaluators
"""

from typing import Dict, Any, List
import json
from rag_api_client import RAGApiClient
from rag_evaluator import RAGEvaluator
from indicators import calculate_text_similarity, calculate_length_comparison, extract_text_samples, ragas_evaluate
from config import Config
from components.utils import parse_sources_to_formatted_list
import streamlit as st

class RAGEvaluationPipeline:
    """Main pipeline for RAG evaluation"""
    
    def __init__(self, 
                 api_client: RAGApiClient = None,
                 evaluator: RAGEvaluator = None):
        """
        Initialize the RAG evaluation pipeline
        
        Args:
            api_client: RAG API client instance
            evaluator: RAG evaluator instance
        """
        self.api_client = api_client or RAGApiClient()
        self.evaluator = evaluator or RAGEvaluator()
    
    def evaluate_question(self, 
                         question: str, 
                         conversation_id: str = "") -> Dict[str, Any]:
        """
        Evaluate a single question through the complete RAG pipeline
        
        Args:
            question: The question to evaluate
            conversation_id: Optional conversation ID for context
            
        Returns:
            Dictionary containing RAG data, evaluation results, and any errors
        """
        # Step 1: Get RAG response
        rag_data = self.api_client.get_rag_data(question, conversation_id)
        
        if "error" in rag_data:
            return {"error": rag_data["error"]}
        
        # Step 2: Extract components
        answer = rag_data.get("answer", "")
        sources_raw = rag_data.get("sources", "")
        
        # Step 2.1: Parse and format sources
        sources_formatted = parse_sources_to_formatted_list(sources_raw)
        
        # Step 3: Evaluate using the original question (use raw sources for evaluation)
        evaluation_results = self.evaluator.evaluate_rag_response(
            query=question,
            response=answer,
            context=sources_raw
        )
        
        # Step 4: Combine results
        return {
            "question": question,
            "answer": answer,
            "sources": sources_raw,  # Keep raw sources for compatibility
            "sources_formatted": sources_formatted,  # Add formatted sources
            "evaluation": evaluation_results
        }
    
    def evaluate_batch(self, 
                      questions: list, 
                      conversation_id: str = "") -> list:
        """
        Evaluate a batch of questions
        
        Args:
            questions: List of questions to evaluate
            conversation_id: Optional conversation ID for context
            
        Returns:
            List of evaluation results for each question
        """
        results = []
        
        for i, question in enumerate(questions):
            print(f"Evaluating question {i+1}/{len(questions)}: {question}")
            
            result = self.evaluate_question(question, conversation_id)
            results.append(result)
            
            # Print summary for each question
            if "error" not in result:
                eval_data = result.get("evaluation", {})
                relevance = eval_data.get("relevance", {})
                groundedness = eval_data.get("groundedness", {})
                
                print(f"  Relevance: {relevance}")
                print(f"  Groundedness: {groundedness}")
            else:
                print(f"  Error: {result['error']}")
            print()
        
        return results
    
    def automated_evaluate_question(self, question_text: str, expected_answer: str, expected_sources: str, conversation_id: str = "") -> Dict[str, Any]:
        """
        Automated evaluation of a single question from the batch data
        
        Args:
            question_text: The question text to evaluate
            pipeline: RAG evaluation pipeline
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        # Step 1: Get RAG response
        rag_data = self.api_client.get_rag_data(question_text, conversation_id)
        answer = rag_data.get("answer", "")
        sources_raw = rag_data.get("sources", "")
        sources_formatted_list = parse_sources_to_formatted_list(sources_raw)
        sources_formatted = '\n'.join(sources_formatted_list)
        sources_formatted = '/\nContent: \n' + sources_formatted
            
        
        # Compare the provided question with the batch question
        question_similarity = calculate_text_similarity(answer, expected_answer)
        
        # Perform similarity comparison between answers
        answer_similarity_metrics = calculate_text_similarity(answer, expected_answer)
        sources_similarity_metrics = calculate_text_similarity(sources_raw, expected_sources)
        
        # Perform length comparison
        length_metrics = calculate_length_comparison(answer, expected_answer)
        sources_length_metrics = calculate_length_comparison(sources_raw, expected_sources)
        
        # Extract and compare samples
        stored_samples = extract_text_samples(answer)
        generated_samples = extract_text_samples(expected_answer)
        
        sample_similarities = []
        for i, (stored_sample, generated_sample) in enumerate(zip(stored_samples, generated_samples)):
            sample_sim = calculate_text_similarity(stored_sample, generated_sample)
            sample_similarities.append({
                "sample_index": i,
                "similarity": sample_sim["average_similarity"]
            })
        
        # RAGAS evaluation will be done at batch level for efficiency
        ragas_metrics = {
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "factual_correctness": 0.0,
            "pending": True  # Flag to indicate RAGAS evaluation is pending
        }
        # Compile comprehensive results
        automated_result = {
            "question_text": question_text,
            "expected_answer": expected_answer,
            "expected_sources": expected_sources,
            "answer": answer,
            "sources": sources_formatted,
            "sources_raw": sources_raw,
            "sources_formatted_list": sources_formatted_list,
            "question_similarity": question_similarity,
            "answer_similarity_metrics": answer_similarity_metrics,
            "sources_similarity_metrics": sources_similarity_metrics,
            "length_metrics": length_metrics,
            "sources_length_metrics": sources_length_metrics,
            "sample_similarities": sample_similarities,
            "ragas_metrics": ragas_metrics,
        }
        
        return automated_result
    
    def ragas_evaluate(self, questions: List[str], answers: List[str], expected_answers: List[str], sources: List[str]) -> Dict[str, Any]:
        """
        Evaluate the answer using RAGAS
        """
        return ragas_evaluate(questions, answers, expected_answers, sources)

def main():
    """Main function for running RAG evaluation"""
    
    # Example usage
    question = "Consumer segmentation"
    
    try:
        # Initialize pipeline
        pipeline = RAGEvaluationPipeline()
        
        # Evaluate single question
        result = pipeline.evaluate_question(question)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        # Display results
        print("=== RAG Evaluation Results ===")
        print(f"Question: {result['question']}")
        print(f"Answer Length: {len(result['answer'])} characters")
        print(f"Sources Length: {len(result['sources'])} characters")
        
        # Display evaluation metrics
        evaluation = result.get("evaluation", {})
        print(f"\nEvaluation Results:")
        print(f"Relevance: {evaluation.get('relevance', 'N/A')}")
        print(f"Groundedness: {evaluation.get('groundedness', 'N/A')}")
        
        # Optionally save results to file
        with open("evaluation_results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        print("\nResults saved to evaluation_results.json")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main() 