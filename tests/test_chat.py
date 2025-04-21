"""
Tests for the chat API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import os
import tempfile
from pathlib import Path
import sys

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.db.vector_store import ingest_documents, clear_vector_store


@pytest.fixture(scope="module")
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(scope="module")
def test_documents():
    """Create temporary test documents."""
    temp_dir = tempfile.TemporaryDirectory()
    
    # Create sample documents
    doc1_path = os.path.join(temp_dir.name, "doc1.txt")
    with open(doc1_path, "w") as f:
        f.write("This is a test document about our company, TechCorp. "
                "TechCorp was founded in 2010 and specializes in AI solutions.")
    
    doc2_path = os.path.join(temp_dir.name, "doc2.txt")
    with open(doc2_path, "w") as f:
        f.write("TechCorp's marketing strategy focuses on content marketing, "
                "social media campaigns, and industry partnerships.")
    
    # Ingest documents
    doc_paths = [doc1_path, doc2_path]
    ingest_documents(doc_paths)
    
    yield doc_paths
    
    # Cleanup
    clear_vector_store()
    temp_dir.cleanup()


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_endpoint(client, test_documents):
    """Test the chat endpoint."""
    # Test a simple query
    response = client.post(
        "/api/chat/message",
        json={
            "messages": [
                {"role": "user", "content": "Tell me about TechCorp"}
            ],
            "include_context": True,
            "include_evaluation": False,
            "stream": False
        }
    )
    
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"]["role"] == "assistant"
    assert len(response.json()["message"]["content"]) > 0
    
    # Check that context is included
    assert "context" in response.json()
    assert "knowledge_results" in response.json()["context"]
    assert len(response.json()["context"]["knowledge_results"]) > 0


def test_chat_with_evaluation(client, test_documents):
    """Test the chat endpoint with evaluation."""
    response = client.post(
        "/api/chat/message",
        json={
            "messages": [
                {"role": "user", "content": "What is TechCorp's marketing strategy?"}
            ],
            "include_context": True,
            "include_evaluation": True,
            "stream": False
        }
    )
    
    assert response.status_code == 200
    assert "message" in response.json()
    assert "context" in response.json()
    assert "evaluation" in response.json()
    
    # Check evaluation structure
    evaluation = response.json()["evaluation"]
    assert "response" in evaluation
    assert "retrieval" in evaluation
    
    # Check response evaluation
    resp_eval = evaluation["response"]
    assert "overall_score" in resp_eval
    assert "metrics" in resp_eval
    
    # Check retrieval evaluation
    retr_eval = evaluation["retrieval"]
    assert "overall_score" in retr_eval
    assert "metrics" in retr_eval


def test_chat_missing_user_message(client):
    """Test the chat endpoint with missing user message."""
    response = client.post(
        "/api/chat/message",
        json={
            "messages": [
                {"role": "assistant", "content": "Hello, how can I help you?"}
            ],
            "include_context": False,
            "include_evaluation": False,
            "stream": False
        }
    )
    
    assert response.status_code == 400
    assert "No user message found" in response.json()["detail"]


def test_chat_with_conversation_history(client, test_documents):
    """Test the chat endpoint with conversation history."""
    response = client.post(
        "/api/chat/message",
        json={
            "messages": [
                {"role": "user", "content": "What does TechCorp do?"},
                {"role": "assistant", "content": "TechCorp specializes in AI solutions and was founded in 2010."},
                {"role": "user", "content": "What is their marketing strategy?"}
            ],
            "include_context": True,
            "include_evaluation": False,
            "stream": False
        }
    )
    
    assert response.status_code == 200
    assert "message" in response.json()
    assert "content marketing" in response.json()["message"]["content"].lower()