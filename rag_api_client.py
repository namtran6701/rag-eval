import requests
import json
import re
from typing import Dict, Any, Optional
from config import Config
from auth import get_azure_key_vault_secret

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
        keySecretName = "orchestrator-host--functionKey"
        functionKey = get_azure_key_vault_secret(keySecretName)
        headers = {"Content-Type": "text/event-stream", "x-functions-key": functionKey}
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
                headers=headers
            )
            response.raise_for_status()

                        # Find the end of the JSON part
            raw_response = response.text
            json_end_index = -1
            brace_count = 0
            in_string = False

            for i, char in enumerate(raw_response):
                if char == '"':
                    # Check for escaped quotes
                    if i > 0 and raw_response[i-1] != '\\':
                        in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end_index = i + 1
                            break
            
            if json_end_index != -1:
                json_part = raw_response[:json_end_index]
                markdown_content = raw_response[json_end_index:].strip()
                
                try:
                    json_data = json.loads(json_part)
                    return {
                        "json_data": json_data,
                        "markdown_content": markdown_content
                    }
                except json.JSONDecodeError as e:
                    return {"error": f"Failed to parse JSON response: {str(e)}"}
            else:
                return {"error": "Could not find valid JSON in response"}

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