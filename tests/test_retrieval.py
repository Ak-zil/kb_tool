"""
Tests for the retrieval functionality.
"""

import pytest
import os
import tempfile
from pathlib import Path
import sys

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.vector_store import (
    ingest_documents,
    query_documents,
    clear_vector_store
)
from app.core.retrieval import retrieval_engine


@pytest.fixture(scope="module")
def test_documents():
    """Create temporary test documents."""
    temp_dir = tempfile.TemporaryDirectory()
    
    # Create sample documents
    doc1_path = os.path.join(temp_dir.name, "doc1.txt")
    with open(doc1_path, "w") as f:
        f.write("This is a test document about machine learning and artificial intelligence. "
                "It contains information about neural networks and deep learning.")
    
    doc2_path = os.path.join(temp_dir.name, "doc2.txt")
    with open(doc2_path, "w") as f:
        f.write("Marketing strategies for technology companies include content marketing, "
                "social media campaigns, and search engine optimization.")
    
    doc3_path = os.path.join(temp_dir.name, "doc3.txt")
    with open(doc3_path, "w") as f:
        f.write("Company metrics and KPIs include customer acquisition cost, lifetime value, "
                "retention rate, and conversion rate. These metrics help measure business performance.")
    
    # Ingest documents
    doc_paths = [doc1_path, doc2_path, doc3_path]
    ingest_documents(doc_paths)
    
    yield doc_paths
    
    # Cleanup
    clear_vector_store()
    temp_dir.cleanup()


def test_query_documents(test_documents):
    """Test querying documents from the vector store."""
    # Test AI-related query
    ai_results = query_documents("What is artificial intelligence?", top_k=1)
    assert len(ai_results) == 1
    assert "machine learning" in ai_results[0]["content"].lower()
    
    # Test marketing-related query
    marketing_results = query_documents("What are good marketing strategies?", top_k=1)
    assert len(marketing_results) == 1
    assert "marketing strategies" in marketing_results[0]["content"].lower()
    
    # Test metrics-related query
    metrics_results = query_documents("What metrics should we track?", top_k=1)
    assert len(metrics_results) == 1
    assert "metrics" in metrics_results[0]["content"].lower()


def test_multiple_results():
    """Test retrieving multiple results."""
    results = query_documents("marketing", top_k=2)
    assert len(results) == 2
    
    # Results should include relevance scores
    assert "relevance_score" in results[0]
    
    # Results should be ordered by relevance
    assert results[0]["relevance_score"] <= results[1]["relevance_score"]


def test_no_results():
    """Test query with no relevant results."""
    results = query_documents("quantum computing blockchain crypto", top_k=2)
    
    # Should still return something, but with low relevance
    assert len(results) > 0
    
    # First result should have a low relevance score
    assert results[0]["relevance_score"] > 0.5  # Scores are distances in Chroma


def test_retrieval_engine(test_documents, monkeypatch):
    """Test the retrieval engine."""
    # Mock the database session
    class MockDB:
        def query(self, *args, **kwargs):
            return self
        
        def filter(self, *args, **kwargs):
            return self
        
        def order_by(self, *args, **kwargs):
            return self
        
        def offset(self, *args, **kwargs):
            return self
        
        def limit(self, *args, **kwargs):
            return []
        
        def all(self):
            return []
    
    # Test querying the knowledge base
    results = retrieval_engine.query_knowledge_base("What is artificial intelligence?")
    assert len(results) > 0
    assert any("machine learning" in r["content"].lower() for r in results)
    
    # Test retrieving and generating with mocked DB
    response, context = retrieval_engine.retrieve_and_generate(
        db=MockDB(),
        query="What are the key marketing metrics?",
        include_knowledge=True,
        include_metrics=False,
        include_summaries=False
    )
    
    assert response is not None and len(response) > 0
    assert "knowledge_results" in context
    assert len(context["knowledge_results"]) > 0