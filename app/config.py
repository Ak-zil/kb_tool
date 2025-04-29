"""
Configuration settings for the Marketing Knowledge Base application.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent


class LLMConfig(BaseModel):
    """Configuration for the LLM service."""
    provider: str = Field(default="groq")
    api_key: str = Field(default=os.getenv("GROQ_API_KEY", ""))
    model_name: str = Field(default=os.getenv("LLM_MODEL_NAME", "llama3-70b-8192"))
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1024)


class VectorDBConfig(BaseModel):
    """Configuration for the vector database."""
    provider: str = Field(default="chroma")
    persist_directory: str = Field(default=str(BASE_DIR / "data" / "vector_store"))
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5") # "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = Field(default="company_knowledge")


class MetricsDBConfig(BaseModel):
    """Configuration for the metrics database."""
    url: str = Field(default=f"sqlite:///{BASE_DIR}/data/metrics/metrics.db")
    echo: bool = Field(default=False)


class S3Config(BaseModel):
    """Configuration for S3 storage."""
    bucket_name: str = Field(default=os.getenv("S3_BUCKET_NAME", "marketing-knowledge-base"))
    region: str = Field(default=os.getenv("AWS_REGION", "us-east-1"))
    access_key_id: str = Field(default=os.getenv("AWS_ACCESS_KEY_ID", ""))
    secret_access_key: str = Field(default=os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    endpoint_url: Optional[str] = Field(default=os.getenv("AWS_ENDPOINT_URL", None))
    
    # Guidelines specific paths
    guidelines_prefix: str = Field(default="guidelines/")
    guideline_assets_prefix: str = Field(default="guidelines/assets/")


class AppConfig(BaseModel):
    """Main application configuration."""
    app_name: str = Field(default="Marketing Knowledge Base")
    debug: bool = Field(default=os.getenv("DEBUG", "False").lower() == "true")
    api_prefix: str = Field(default="/api")
    knowledge_base_dir: str = Field(default=str(BASE_DIR / "data" / "knowledge_base"))
    chat_summaries_dir: str = Field(default=str(BASE_DIR / "data" / "chat_summaries"))
    
    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    metrics_db: MetricsDBConfig = Field(default_factory=MetricsDBConfig)
    s3: S3Config = Field(default_factory=S3Config)


# Create global config instance
config = AppConfig()


def get_settings() -> AppConfig:
    """Returns the application settings."""
    return config