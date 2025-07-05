from typing import Dict, Any, Optional

class RAGDataParser:
    """Parser for RAG API responses"""
    
    def get_sources_from_thoughts(self, thoughts_text: str) -> str:
        """
        Extract content after "Context Retrieved using the rewritten query: " from thoughts text
        
        Args:
            thoughts_text: The thoughts text from the API response
            
        Returns:
            Everything after "Context Retrieved using the rewritten query: ", or empty string if not found
        """
        if not thoughts_text:
            return ""
        
        # Find the context section and return everything after it
        context_marker = "Context Retrieved using the rewritten query:"
        context_start = thoughts_text.find(context_marker)
        
        if context_start != -1:
            # Return everything after the marker
            content = thoughts_text[context_start + len(context_marker):].strip()
            
            # Handle case where there are 0 documents (content is just "/" or "/ ")
            if content in ["/", "/ ", ""]:
                return ""
            
            return content
        
        return ""
    
    def parse_rag_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the complete RAG API response into structured data
        
        Args:
            api_response: The raw API response containing 'json_data' and 'markdown_content'
            
        Returns:
            Dictionary containing 'answer', 'sources', or 'error' if failed
        """
        try:
            # Extract the answer (markdown content)
            answer = api_response.get("markdown_content", "")
            
            # Extract sources from thoughts
            sources = ""
            
            if "json_data" in api_response and "thoughts" in api_response["json_data"]:
                thoughts_list = api_response["json_data"]["thoughts"]
                if thoughts_list:
                    thoughts_text = thoughts_list[0]
                    sources = self.get_sources_from_thoughts(thoughts_text)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            return {"error": f"Error parsing RAG response: {str(e)}"} 