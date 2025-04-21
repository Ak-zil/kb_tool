#!/usr/bin/env python
"""
Script for running evaluation tests on the knowledge base retrieval and response generation.
"""

import argparse
import csv
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.retrieval import retrieval_engine
from app.core.evaluation import evaluation_engine
from app.config import get_settings
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_db_session():
    """Create and return a database session."""
    settings = get_settings()
    engine = create_engine(settings.metrics_db.url, echo=settings.metrics_db.echo)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def load_test_queries(file_path: str) -> List[Dict[str, Any]]:
    """
    Load test queries from a file.
    Supports both CSV and JSON formats.
    
    Args:
        file_path: Path to the test queries file
    
    Returns:
        List of test query objects
    """
    logger.info(f"Loading test queries from: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        # Load from CSV
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    
    elif file_ext == '.json':
        # Load from JSON
        with open(file_path, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
            
            # Handle different JSON formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "queries" in data:
                return data["queries"]
            else:
                return [data]
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def run_evaluation(
    test_queries: List[Dict[str, Any]], 
    db: Session,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run evaluation on test queries.
    
    Args:
        test_queries: List of test query objects
        db: Database session
        output_file: Optional path to output file for detailed results
    
    Returns:
        Evaluation results summary
    """
    logger.info(f"Running evaluation on {len(test_queries)} test queries")
    
    results = []
    
    for i, query_obj in enumerate(test_queries, 1):
        logger.info(f"Evaluating query {i}/{len(test_queries)}: {query_obj.get('query', '')[:50]}...")
        
        # Get the query text
        query = query_obj.get("query", "")
        
        # Get expected answer if available
        expected_answer = query_obj.get("expected_answer", "")
        
        # Skip empty queries
        if not query:
            logger.warning(f"Skipping empty query at index {i}")
            continue
        
        try:
            # Process the query
            response, retrieval_context = retrieval_engine.retrieve_and_generate(
                db=db,
                query=query
            )
            
            # Evaluate the results
            context_docs = retrieval_context.get("knowledge_results", [])
            
            # Evaluate response
            response_evaluation = evaluation_engine.evaluate_response(
                query=query,
                response=response,
                contexts=context_docs
            )
            
            # Evaluate retrieval
            retrieval_evaluation = evaluation_engine.evaluate_retrieval(
                query=query,
                retrieved_contexts=context_docs
            )
            
            # Add to results
            result = {
                "query": query,
                "expected_answer": expected_answer,
                "generated_response": response,
                "contexts_count": len(context_docs),
                "response_evaluation": response_evaluation,
                "retrieval_evaluation": retrieval_evaluation,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error evaluating query {query}: {str(e)}")
            
            # Add failed result
            results.append({
                "query": query,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    # Calculate summary statistics
    summary = calculate_summary(results)
    
    # Write detailed results to file if requested
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": summary,
                    "detailed_results": results
                }, f, indent=2)
            
            logger.info(f"Detailed evaluation results written to: {output_file}")
        except Exception as e:
            logger.error(f"Error writing results to {output_file}: {str(e)}")
    
    return summary


def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics from evaluation results.
    
    Args:
        results: List of evaluation results
    
    Returns:
        Summary statistics
    """
    # Count successful evaluations
    successful = [r for r in results if "error" not in r]
    
    if not successful:
        return {
            "total_queries": len(results),
            "successful_evaluations": 0,
            "failed_evaluations": len(results),
            "error_rate": 1.0
        }
    
    # Extract scores
    response_scores = [
        r["response_evaluation"]["overall_score"] 
        for r in successful 
        if "response_evaluation" in r and "overall_score" in r["response_evaluation"]
    ]
    
    retrieval_scores = [
        r["retrieval_evaluation"]["overall_score"] 
        for r in successful 
        if "retrieval_evaluation" in r and "overall_score" in r["retrieval_evaluation"]
    ]
    
    # Calculate averages
    avg_response_score = sum(response_scores) / len(response_scores) if response_scores else 0
    avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0
    
    # Calculate average individual metrics if available
    response_metrics = {}
    retrieval_metrics = {}
    
    if successful and "response_evaluation" in successful[0] and "metrics" in successful[0]["response_evaluation"]:
        # Get all metric keys
        metric_keys = set()
        for r in successful:
            if "response_evaluation" in r and "metrics" in r["response_evaluation"]:
                metric_keys.update(r["response_evaluation"]["metrics"].keys())
        
        # Calculate averages for each metric
        for key in metric_keys:
            values = [
                r["response_evaluation"]["metrics"].get(key, 0) 
                for r in successful
                if "response_evaluation" in r and "metrics" in r["response_evaluation"] 
                and key in r["response_evaluation"]["metrics"]
            ]
            
            if values:
                response_metrics[key] = sum(values) / len(values)
    
    if successful and "retrieval_evaluation" in successful[0] and "metrics" in successful[0]["retrieval_evaluation"]:
        # Get all metric keys
        metric_keys = set()
        for r in successful:
            if "retrieval_evaluation" in r and "metrics" in r["retrieval_evaluation"]:
                metric_keys.update(r["retrieval_evaluation"]["metrics"].keys())
        
        # Calculate averages for each metric
        for key in metric_keys:
            values = [
                r["retrieval_evaluation"]["metrics"].get(key, 0) 
                for r in successful
                if "retrieval_evaluation" in r and "metrics" in r["retrieval_evaluation"] 
                and key in r["retrieval_evaluation"]["metrics"]
            ]
            
            if values:
                retrieval_metrics[key] = sum(values) / len(values)
    
    # Return summary
    return {
        "total_queries": len(results),
        "successful_evaluations": len(successful),
        "failed_evaluations": len(results) - len(successful),
        "error_rate": (len(results) - len(successful)) / len(results) if results else 0,
        "avg_response_score": avg_response_score,
        "avg_retrieval_score": avg_retrieval_score,
        "response_metrics": response_metrics,
        "retrieval_metrics": retrieval_metrics,
        "avg_overall_score": (avg_response_score + avg_retrieval_score) / 2 if (avg_response_score and avg_retrieval_score) else 0
    }


def main():
    """Main entry point for the evaluation runner script."""
    parser = argparse.ArgumentParser(
        description="Run evaluation tests on the Marketing Knowledge Base"
    )
    
    parser.add_argument(
        "queries_file", 
        help="Path to file containing test queries (CSV or JSON)"
    )
    
    parser.add_argument(
        "--output", 
        help="Path to output file for detailed results (JSON)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.queries_file):
        logger.error(f"Queries file does not exist: {args.queries_file}")
        return
    
    # Get database session
    db = get_db_session()
    
    try:
        # Load test queries
        test_queries = load_test_queries(args.queries_file)
        
        if not test_queries:
            logger.error("No test queries found")
            return
        
        # Run evaluation
        summary = run_evaluation(
            test_queries=test_queries,
            db=db,
            output_file=args.output
        )
        
        # Print summary
        logger.info("Evaluation Summary:")
        logger.info(f"  Total Queries: {summary['total_queries']}")
        logger.info(f"  Successful Evaluations: {summary['successful_evaluations']}")
        logger.info(f"  Failed Evaluations: {summary['failed_evaluations']}")
        logger.info(f"  Error Rate: {summary['error_rate']:.2f}")
        logger.info(f"  Average Response Score: {summary['avg_response_score']:.4f}")
        logger.info(f"  Average Retrieval Score: {summary['avg_retrieval_score']:.4f}")
        logger.info(f"  Average Overall Score: {summary['avg_overall_score']:.4f}")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()