#!/usr/bin/env python3
"""
Main module for RAG evaluation system

This module orchestrates the complete RAG evaluation process:
1. Make API calls to the RAG system
2. Parse responses to extract answer, rewritten query, and sources
3. Evaluate the response using Azure OpenAI evaluators
"""

from typing import Dict, Any, Optional
import json
from rag_api_client import RAGApiClient
from rag_evaluator import RAGEvaluator
from config import Config

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
        sources = rag_data.get("sources", "")
        
        # Step 3: Evaluate using the original question
        evaluation_results = self.evaluator.evaluate_rag_response(
            query=question,
            response=answer,
            context=sources
        )
        
        # Step 4: Combine results
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
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