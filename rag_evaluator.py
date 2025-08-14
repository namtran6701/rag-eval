from typing import Dict, Any
from azure.ai.evaluation import AzureOpenAIModelConfiguration, GroundednessEvaluator, RelevanceEvaluator
from config import Config
from ragas import EvaluationDataset

class RAGEvaluator:
    """Evaluator for RAG system responses using Azure OpenAI"""
    
    def __init__(self, 
                 azure_openai_service: str = None,
                 deployment_name: str = None):
        """
        Initialize the RAG evaluator
        
        Args:
            azure_openai_service: Azure OpenAI service name
            deployment_name: Azure OpenAI deployment name
        """
        # Validate configuration
        Config.validate()
        
        self.azure_openai_service = azure_openai_service or Config.AZURE_OPENAI_SERVICE
        self.deployment_name = deployment_name or Config.AZURE_OPENAI_DEPLOYMENT_NAME
        self.api_key = Config.AZURE_OPENAI_KEY
        
        # Configure Azure OpenAI model
        self.model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=f"https://{self.azure_openai_service}.openai.azure.com",
            azure_deployment=self.deployment_name,
            api_key=self.api_key
        )
        
        # Initialize evaluators
        self._relevance_evaluator = None
        self._groundedness_evaluator = None
    
    @property
    def relevance_evaluator(self):
        """Lazy initialization of relevance evaluator"""
        if self._relevance_evaluator is None:
            self._relevance_evaluator = RelevanceEvaluator(
                model_config=self.model_config
            )
        return self._relevance_evaluator
    
    @property
    def groundedness_evaluator(self):
        """Lazy initialization of groundedness evaluator"""
        if self._groundedness_evaluator is None:
            self._groundedness_evaluator = GroundednessEvaluator(
                prompty_file="prompts/prompt_groundedness.prompty",
                model_config=self.model_config
            )
        return self._groundedness_evaluator
    
    def evaluate_relevance(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """
        Evaluate the relevance of a response to a query given context
        
        Args:
            query: The input query
            response: The generated response
            context: The context/sources used
            
        Returns:
            Dictionary containing relevance evaluation results
        """
        try:
            result = self.relevance_evaluator(
                query=query, 
                response=response, 
                context=context
            )
            return result
        except Exception as e:
            return {"error": f"Relevance evaluation failed: {str(e)}"}
    
    def evaluate_groundedness(self, response: str, context: str) -> Dict[str, Any]:
        """
        Evaluate the groundedness of a response given context
        
        Args:
            response: The generated response
            context: The context/sources used
            
        Returns:
            Dictionary containing groundedness evaluation results
        """
        try:
            result = self.groundedness_evaluator(
                response=response, 
                context=context
            )
            return result
        except Exception as e:
            return {"error": f"Groundedness evaluation failed: {str(e)}"}
    
    def evaluate_rag_response(self, 
                            query: str, 
                            response: str, 
                            context: str) -> Dict[str, Any]:
        """
        Perform complete RAG evaluation including relevance and groundedness
        
        Args:
            query: The input query
            response: The generated response
            context: The context/sources used
            
        Returns:
            Dictionary containing both relevance and groundedness evaluation results
        """
        results = {
            "query": query,
            "response_length": len(response),
            "context_length": len(context)
        }
        
        # Evaluate relevance
        #relevance_result = self.evaluate_relevance(query, response, context)
        #results["relevance"] = relevance_result
        
        # Evaluate groundedness
        #groundedness_result = self.evaluate_groundedness(response, context)
        #results["groundedness"] = groundedness_result
        
        return results