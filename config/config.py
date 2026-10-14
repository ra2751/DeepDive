import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    # OpenAI API Key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    Milvus_ENDPOINT = os.getenv("Milvus_ENDPOINT")
    Milvus_API_KEY = os.getenv("Milvus_API_KEY")
    COLLECTION_NAME = "ai_research_assistant"
    TOP_K = 5

    # Model Configuration
    LLM_MODEL = "gpt-4o-mini"
    #LLM_VALIDATION_MODEL = "gpt-4.1-nano"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50 

    @classmethod
    def get_llm(cls):
        """ Initialize and return the LLM models"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)
    
    def get_llm_validation(cls):
        """ Initialize and return the LLM models"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_VALIDATION_MODEL)