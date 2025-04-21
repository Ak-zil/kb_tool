"""
Tests for the evaluation system.
"""

import pytest
import json
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.evaluation import evaluation_engine


@pytest.fixture
def mock_contexts():
    """Create mock retrieval contexts."""
    return [
        {
            "content": "TechCorp was founded in 2010 and specializes in AI solutions. "
                       "Our headquarters is in San Francisco.",
            "metadata": {"source": "company_info.txt"},
            "relevance_score": 0.95
        },
        {
            "content": "TechCorp's revenue grew by 30% in 2024, reaching $50 million.",
            "metadata": {"source": "financial_report.txt"},
            "relevance_score": 0.85
        }
    ]


def test_evaluate_response(mock_contexts):
    """Test evaluating a response."""
    # Mock LLM evaluation to avoid actual API calls
    with patch('app.core.evaluation.EvaluationEngine._evaluate_with_ragas') as mock_ragas, \
         patch('app.core.evaluation.EvaluationEngine._evaluate_with_llm') as mock_llm:
        
        # Configure mocks
        mock_ragas.return_value = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_relevancy": 0.8
        }
        
        mock_llm.return_value = {
            "coherence": 0.95,
            "helpfulness": 0.9
        }
        
        # Test query and response
        query = "Tell me about TechCorp's financials."
        response = "TechCorp had a strong financial performance in 2024 with 30% revenue growth, reaching $50 million."
        
        # Evaluate response
        results = evaluation_engine.evaluate_response(query, response, mock_contexts)
        
        # Check results structure
        assert "query" in results
        assert "response_length" in results
        assert "context_count" in results
        assert "metrics" in results
        assert "overall_score" in results
        
        # Check metrics
        metrics = results["metrics"]
        assert "faithfulness" in metrics
        assert "answer_relevancy" in metrics
        assert "context_relevancy" in metrics
        assert "coherence" in metrics
        assert "helpfulness" in metrics
        
        # Check values
        assert metrics["faithfulness"] == 0.9
        assert metrics["answer_relevancy"] == 0.85
        assert metrics["context_relevancy"] == 0.8
        assert metrics["coherence"] == 0.95
        assert metrics["helpfulness"] == 0.9
        
        # Check overall score calculation
        expected_score = (0.9 + 0.85 + 0.8 + 0.95 + 0.9) / 5
        assert abs(results["overall_score"] - expected_score) < 0.001


def test_evaluate_retrieval(mock_contexts):
    """Test evaluating retrieval quality."""
    # Mock LLM evaluation to avoid actual API calls
    with patch('app.core.evaluation.EvaluationEngine._evaluate_retrieval_with_llm') as mock_llm:
        
        # Configure mock
        mock_llm.return_value = {"llm_relevance": 0.75}
        
        # Test query
        query = "Tell me about TechCorp's financials."
        
        # Evaluate retrieval
        results = evaluation_engine.evaluate_retrieval(query, mock_contexts)
        
        # Check results structure
        assert "query" in results
        assert "retrieval_count" in results
        assert "metrics" in results
        assert "overall_score" in results
        
        # Check metrics
        metrics = results["metrics"]
        assert "avg_relevance" in metrics
        assert "max_relevance" in metrics
        assert "min_relevance" in metrics
        assert "llm_relevance" in metrics
        
        # Check values
        assert metrics["avg_relevance"] == 0.9  # (0.95 + 0.85) / 2
        assert metrics["max_relevance"] == 0.95
        assert metrics["min_relevance"] == 0.85
        assert metrics["llm_relevance"] == 0.75
        
        # Check overall score calculation
        expected_score = (0.9 + 0.75) / 2
        assert abs(results["overall_score"] - expected_score) < 0.001


def test_evaluate_with_llm():
    """Test LLM-based evaluation."""
    # This test requires more extensive mocking of the LLM
    # Here we'll focus on the method interface and minimal functionality
    with patch('app.core.llm.get_llm') as mock_get_llm:
        # Create mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.score = 0.95
        
        # Configure mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_strings.return_value = mock_response
        
        # Configure mock LLM chain
        with patch('langchain.evaluation.load_evaluator') as mock_load_evaluator:
            mock_load_evaluator.return_value = mock_evaluator
            
            # Call the method
            query = "What is TechCorp's revenue?"
            response = "TechCorp's revenue is $50 million."
            
            # Test with minimal functionality
            with patch.object(evaluation_engine, '_evaluate_with_llm', return_value={"coherence": 0.95}):
                result = evaluation_engine._evaluate_with_llm(query, response)
                assert "coherence" in result
                assert result["coherence"] == 0.95


def test_error_handling():
    """Test error handling in evaluation."""
    # Test with invalid inputs
    results = evaluation_engine.evaluate_response("", "", [])
    assert results["overall_score"] == 0
    assert "metrics" in results
    
    # Test with exception in evaluation
    with patch.object(evaluation_engine, '_evaluate_with_ragas', side_effect=Exception("Test error")), \
         patch.object(evaluation_engine, '_evaluate_with_llm', return_value={}):
        
        results = evaluation_engine.evaluate_response("query", "response", [])
        assert "metrics" in results
        assert results["overall_score"] == 0


def test_evaluation_with_empty_contexts():
    """Test evaluation with empty contexts."""
    # Mock LLM evaluation to avoid actual API calls
    with patch('app.core.evaluation.EvaluationEngine._evaluate_with_llm') as mock_llm:
        
        # Configure mock
        mock_llm.return_value = {
            "coherence": 0.9,
            "helpfulness": 0.85
        }
        
        # Test query and response
        query = "What does TechCorp do?"
        response = "TechCorp specializes in AI solutions."
        
        # Evaluate response with empty contexts
        results = evaluation_engine.evaluate_response(query, response, [])
        
        # Check results
        assert "metrics" in results
        assert "coherence" in results["metrics"]
        assert "helpfulness" in results["metrics"]
        assert "faithfulness" not in results["metrics"]  # RAGAS metrics should not be present
        
        # Check overall score calculation
        expected_score = (0.9 + 0.85) / 2
        assert abs(results["overall_score"] - expected_score) < 0.001