"""
Core retrieval functionality for the Marketing Knowledge Base.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.db.vector_store import query_documents
from app.db.metrics_db import get_metrics, get_chat_summaries
from app.core.llm import extract_metrics_query, answer_with_metrics, generate_response
from app.core.metrics_engine import format_metrics_for_context

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Main engine for retrieving information from the knowledge base.
    Handles integration between vector store, metrics database, and LLM.
    """
    
    def query_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant documents.
        
        Args:
            query: The query string
            top_k: Number of top results to return
        
        Returns:
            List of relevant documents
        """
        logger.info(f"Querying knowledge base: '{query}'")
        return query_documents(query, top_k=top_k)
    
    def query_metrics(self, db: Session, query: str) -> List[Dict[str, Any]]:
        """
        Query the metrics database based on the user's query.
        Uses LLM to interpret the query and extract parameters.
        
        Args:
            db: Database session
            query: User's original query
        
        Returns:
            List of relevant metrics
        """
        logger.info(f"Querying metrics database: '{query}'")
        
        try:
            # Extract structured query using LLM
            metrics_query = extract_metrics_query(query)
            
            # Parse the structured query
            try:
                query_params = json.loads(metrics_query)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse metrics query: {metrics_query}")
                query_params = {}
            
            # Extract parameters
            metrics_names = query_params.get("metrics", [])
            category = query_params.get("category")
            time_period = query_params.get("time_period")
            
            # Query database
            # For now, we're just using category and name filtering
            # In a real implementation, time_period would need to be parsed into date ranges
            results = []
            
            if category:
                metrics = get_metrics(db, category=category)
                results.extend(metrics)
            
            for metric_name in metrics_names:
                # Simple partial matching for now
                metrics = get_metrics(db, name=metric_name)
                results.extend(metrics)
            
            # If no specific queries matched, return recent metrics
            if not results:
                results = get_metrics(db, limit=10)
            
            return [self._format_metric(metric) for metric in results]
            
        except Exception as e:
            logger.error(f"Error querying metrics: {str(e)}")
            return []
    
    def query_chat_summaries(self, db: Session, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query recent chat summaries.
        
        Args:
            db: Database session
            query: User's original query
            limit: Maximum number of summaries to return
        
        Returns:
            List of relevant chat summaries
        """
        logger.info(f"Querying chat summaries: '{query}'")
        
        # For now, we just return the most recent summaries
        # In a real implementation, we would use the query to filter summaries
        summaries = get_chat_summaries(db, limit=limit)
        
        return [self._format_chat_summary(summary) for summary in summaries]
    
    def retrieve_and_generate(
        self, 
        db: Session,
        query: str, 
        include_knowledge: bool = True,
        include_metrics: bool = True,
        include_summaries: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Retrieve information and generate a response.
        
        Args:
            db: Database session
            query: User's query
            include_knowledge: Whether to include knowledge base results
            include_metrics: Whether to include metrics
            include_summaries: Whether to include chat summaries
        
        Returns:
            Tuple of (generated response, retrieval context)
        """
        logger.info(f"Processing query: '{query}'")
        
        # Initialize context and retrieval sources
        retrieval_context = {
            "knowledge_results": [],
            "metrics_results": [],
            "summary_results": []
        }
        
        # Check if the query is metrics-related
        is_metrics_query = self._is_metrics_related(query)
        
        # Retrieve information from different sources
        if include_knowledge:
            knowledge_results = self.query_knowledge_base(query)
            retrieval_context["knowledge_results"] = knowledge_results
        
        if include_metrics and (is_metrics_query or not include_knowledge):
            metrics_results = self.query_metrics(db, query)
            retrieval_context["metrics_results"] = metrics_results
        
        if include_summaries:
            summary_results = self.query_chat_summaries(db, query)
            retrieval_context["summary_results"] = summary_results
        
        # Determine how to handle the query based on retrieved information
        if is_metrics_query and retrieval_context["metrics_results"]:
            # Format metrics for context
            metrics_context = format_metrics_for_context(retrieval_context["metrics_results"])
            
            # Generate response focused on metrics
            response = answer_with_metrics(query, metrics_context)
        else:
            # Combine all retrieval results into a single context
            combined_context = self._combine_contexts(retrieval_context)
            
            # Generate response with the combined context
            response = generate_response(query, context=combined_context)
        
        return response, retrieval_context
    
    def _is_metrics_related(self, query: str) -> bool:
        """
        Check if the query is related to metrics or figures.
        
        Args:
            query: User's query
        
        Returns:
            Boolean indicating if query is metrics-related
        """
        metrics_keywords = [
            "metric", "figure", "statistic", "number", "percent", "growth",
            "sales", "revenue", "profit", "customer", "user", "acquisition",
            "churn", "retention", "conversion", "engagement", "roi", "roas",
            "how many", "how much", "rate", "ratio", "average", "mean", "median"
        ]
        
        lower_query = query.lower()
        return any(keyword in lower_query for keyword in metrics_keywords)
    
    def _format_metric(self, metric) -> Dict[str, Any]:
        """
        Format a metric object for use in contexts.
        
        Args:
            metric: Metric database object
        
        Returns:
            Dictionary with formatted metric data
        """
        # Parse metadata if available
        metadata = {}
        if metric.metadata:
            try:
                metadata = json.loads(metric.metadata)
            except json.JSONDecodeError:
                pass
        
        return {
            "id": metric.id,
            "name": metric.name,
            "value": metric.value,
            "unit": metric.unit,
            "category": metric.category,
            "subcategory": metric.subcategory,
            "timestamp": metric.timestamp.isoformat(),
            "description": metric.description,
            "metadata": metadata
        }
    
    def _format_chat_summary(self, summary) -> Dict[str, Any]:
        """
        Format a chat summary object for use in contexts.
        
        Args:
            summary: ChatSummary database object
        
        Returns:
            Dictionary with formatted summary data
        """
        # Parse topics and metadata if available
        topics = []
        if summary.topics:
            try:
                topics = json.loads(summary.topics)
            except json.JSONDecodeError:
                pass
        
        metadata = {}
        if summary.metadata:
            try:
                metadata = json.loads(summary.metadata)
            except json.JSONDecodeError:
                pass
        
        return {
            "id": summary.id,
            "title": summary.title,
            "summary": summary.summary,
            "topics": topics,
            "timestamp": summary.timestamp.isoformat(),
            "metadata": metadata
        }
    
    def _combine_contexts(self, retrieval_context: Dict[str, Any]) -> str:
        """
        Combine different retrieval contexts into a single context string.
        
        Args:
            retrieval_context: Dictionary with different retrieval results
        
        Returns:
            Combined context as a string
        """
        parts = []
        
        # Add knowledge base results
        if retrieval_context["knowledge_results"]:
            knowledge_texts = []
            for i, result in enumerate(retrieval_context["knowledge_results"], 1):
                knowledge_texts.append(f"Document {i}:\n{result['content']}")
            
            parts.append("KNOWLEDGE BASE INFORMATION:\n" + "\n\n".join(knowledge_texts))
        
        # Add metrics results
        if retrieval_context["metrics_results"]:
            metrics_context = format_metrics_for_context(retrieval_context["metrics_results"])
            parts.append("COMPANY METRICS:\n" + metrics_context)
        
        # Add chat summary results
        if retrieval_context["summary_results"]:
            summary_texts = []
            for i, result in enumerate(retrieval_context["summary_results"], 1):
                summary_texts.append(f"Summary {i} - {result['title']}:\n{result['summary']}")
            
            parts.append("RECENT DISCUSSIONS:\n" + "\n\n".join(summary_texts))
        
        # Combine all parts
        if parts:
            return "\n\n".join(parts)
        else:
            return "No relevant information found in the company knowledge base."


# Create a global instance for use in API endpoints
retrieval_engine = RetrievalEngine()