import requests
import json
import re
from typing import Dict, Any, Optional
from config import Config

class RAGApiClient:
    """Client for interacting with the RAG API"""
    
    def __init__(self, 
                 api_url: str = None,
                 client_principal_id: str = None,
                 client_principal_name: str = None,
                 client_principal_organization: str = None):
        """
        Initialize the RAG API client
        
        Args:
            api_url: The RAG API endpoint URL
            client_principal_id: Client principal ID for authentication
            client_principal_name: Client principal name
            client_principal_organization: Client principal organization
        """
        self.api_url = api_url or Config.RAG_API_URL
        self.client_principal_id = client_principal_id or Config.DEFAULT_CLIENT_PRINCIPAL_ID
        self.client_principal_name = client_principal_name or Config.DEFAULT_CLIENT_PRINCIPAL_NAME
        self.client_principal_organization = client_principal_organization or Config.DEFAULT_CLIENT_PRINCIPAL_ORGANIZATION
    
    def make_api_call(self, question: str, conversation_id: str = "") -> Dict[str, Any]:
        """
        Make a call to the RAG endpoint and return parsed response
        
        Args:
            question: The question to ask
            conversation_id: Optional conversation ID for context
            
        Returns:
            Dictionary containing 'json_data' and 'markdown_content', or 'error' if failed
        """
        payload = {
            "conversation_id": conversation_id,
            "question": question,
            "client_principal_id": self.client_principal_id,
            "client_principal_name": self.client_principal_name,
            "client_principal_organization": self.client_principal_organization,
        }

        try:
            response = requests.post(
                self.api_url, 
                json=payload, 
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            # Extract just the JSON part from response
            raw_response = response.text
            json_match = re.search(r"^(\{.*?\})", raw_response, re.DOTALL)

            if json_match:
                json_part = json_match.group(1)
                json_data = json.loads(json_part)

                # Extract the markdown content (everything after the JSON)
                markdown_content = raw_response[json_match.end():].strip()

                # Return both JSON data and markdown content
                return {
                    "json_data": json_data, 
                    "markdown_content": markdown_content
                }
            else:
                return {"error": "Could not parse JSON from response"}

        except requests.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON response: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def get_rag_data(self, question: str, conversation_id: str = "") -> Dict[str, Any]:
        """
        Get complete RAG data including answer and sources
        
        Args:
            question: The question to ask
            conversation_id: Optional conversation ID for context
            
        Returns:
            Dictionary containing 'answer', 'sources', or 'error' if failed
        """
        from data_parser import RAGDataParser  # Import here to avoid circular imports
        
        # Make the API call
        result = self.make_api_call(question, conversation_id)
        
        # Check for errors
        if "error" in result:
            return {"error": result["error"]}
        
        # Parse the response
        parser = RAGDataParser()
        return parser.parse_rag_response(result) 