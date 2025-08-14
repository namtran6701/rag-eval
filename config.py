import os
import dotenv
import streamlit as st

# Load environment variables
dotenv.load_dotenv()

class Config:
    """Configuration class for RAG evaluation system"""
    
    # Azure OpenAI Configuration
    if os.getenv("ENV") == "local":
        AZURE_OPENAI_SERVICE = os.getenv("AZURE_OPENAI_SERVICE")
        AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    else:
        AZURE_OPENAI_SERVICE = st.secrets["AZURE_OPENAI_SERVICE"]
        AZURE_OPENAI_DEPLOYMENT_NAME = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]
        AZURE_OPENAI_KEY = st.secrets["AZURE_OPENAI_KEY"]
    
    # RAG API Configuration
    RAG_API_URL = os.getenv("RAG_API_URL")
    
    # Default client configuration (should be configurable in production)
    DEFAULT_CLIENT_PRINCIPAL_ID = os.getenv("DEFAULT_CLIENT_PRINCIPAL_ID")
    DEFAULT_CLIENT_PRINCIPAL_NAME = os.getenv("DEFAULT_CLIENT_PRINCIPAL_NAME")
    DEFAULT_CLIENT_PRINCIPAL_ORGANIZATION = os.getenv("DEFAULT_CLIENT_PRINCIPAL_ORGANIZATION")
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        required_vars = [
            'AZURE_OPENAI_SERVICE',
            'AZURE_OPENAI_DEPLOYMENT_NAME'
        ]
        
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}") 