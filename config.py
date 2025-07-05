import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

class Config:
    """Configuration class for RAG evaluation system"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # RAG API Configuration
    RAG_API_URL = "https://fnorch0-vm2b2htvuuclm.azurewebsites.net/api/orc"
    
    # Default client configuration (should be configurable in production)
    DEFAULT_CLIENT_PRINCIPAL_ID = "96567627-0cce-45b4-97f0-9972d03a268d"
    DEFAULT_CLIENT_PRINCIPAL_NAME = "sheep"
    DEFAULT_CLIENT_PRINCIPAL_ORGANIZATION = "6c33b530-22f6-49ca-831b-25d587056237"
    
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