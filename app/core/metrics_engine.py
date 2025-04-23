"""
Metrics engine for processing and formatting company metrics.
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


# def format_metrics_for_context(metrics: List[Dict[str, Any]]) -> str:
#     """
#     Format metrics data into a readable context string for the LLM.
    
#     Args:
#         metrics: List of metric dictionaries
    
#     Returns:
#         Formatted context string
#     """
#     if not metrics:
#         return "No relevant metrics found."
    
#     # Group metrics by category
#     categories = defaultdict(list)
#     for metric in metrics:
#         categories[metric["category"]].append(metric)
    
#     # Format each category
#     context_parts = []
#     for category, category_metrics in categories.items():
#         context_parts.append(f"## {category.upper()} METRICS:")
        
#         # Group by subcategory if available
#         subcategories = defaultdict(list)
#         for metric in category_metrics:
#             subcategory = metric.get("subcategory") or "General"
#             subcategories[subcategory].append(metric)
        
#         for subcategory, subcategory_metrics in subcategories.items():
#             if subcategory != "General":
#                 context_parts.append(f"\n### {subcategory}:")
            
#             # Format each metric
#             for metric in subcategory_metrics:
#                 date_str = ""
#                 if "timestamp" in metric:
#                     try:
#                         dt = datetime.fromisoformat(metric["timestamp"])
#                         date_str = f" (as of {dt.strftime('%B %d, %Y')})"
#                     except (ValueError, TypeError):
#                         pass
                
#                 value_str = format_metric_value(metric["value"], metric.get("unit", ""))
#                 description = f": {metric['description']}" if metric.get("description") else ""
                
#                 context_parts.append(f"- {metric['name']}: {value_str}{date_str}{description}")
            
#             context_parts.append("")  # Add blank line between subcategories
    
#     return "\n".join(context_parts)

def format_metrics_for_context(metrics: List[Dict[str, Any]]) -> str:
    """
    Format metrics data into a readable context string for the LLM.
    Fixes timestamp interpretation issues.
    
    Args:
        metrics: List of metric dictionaries
    
    Returns:
        Formatted context string
    """
    if not metrics:
        return "No relevant metrics found."
    
    # Group metrics by category
    categories = defaultdict(list)
    for metric in metrics:
        categories[metric["category"]].append(metric)
    
    # Format each category
    context_parts = []
    for category, category_metrics in categories.items():
        context_parts.append(f"## {category.upper()} METRICS:")
        
        # Group by subcategory if available
        subcategories = defaultdict(list)
        for metric in category_metrics:
            subcategory = metric.get("subcategory") or "General"
            subcategories[subcategory].append(metric)
        
        for subcategory, subcategory_metrics in subcategories.items():
            if subcategory != "General":
                context_parts.append(f"\n### {subcategory}:")
            
            # Format each metric
            for metric in subcategory_metrics:
                value_str = format_metric_value(metric["value"], metric.get("unit", ""))
                
                # Format the description, removing any confusion with timestamps
                # If the description contains the word "in" followed by a year, we want to keep this clear
                description = ""
                if metric.get("description"):
                    description = f" - {metric['description']}"
                
                # Don't include timestamp information to avoid confusion
                # The timestamp is when the metric was recorded, not what the metric is about
                # If the description has date info like "in 2024", that will still be shown
                
                context_parts.append(f"- {metric['name']}: {value_str}{description}")
            
            context_parts.append("")  # Add blank line between subcategories
    
    return "\n".join(context_parts)


# def get_metrics_summary(metrics: List[Dict[str, Any]], top_n: int = 5) -> str:
#     """
#     Generate a summary of the most important metrics.
#     Modified to avoid timestamp confusion.
    
#     Args:
#         metrics: List of metric dictionaries
#         top_n: Number of top metrics to include
    
#     Returns:
#         Summary string
#     """
#     if not metrics:
#         return "No metrics available."
    
#     # For simplicity, just take the first top_n metrics
#     # In a real implementation, you would prioritize metrics by importance
#     top_metrics = metrics[:min(top_n, len(metrics))]
    
#     summary_parts = ["Key Company Metrics:"]
#     for metric in top_metrics:
#         value_str = format_metric_value(metric["value"], metric.get("unit", ""))
        
#         # Use the description if available, removing any confusion with timestamps
#         description = ""
#         if metric.get("description"):
#             description = f" - {metric['description']}"
        
#         # Don't include timestamp information to avoid confusion
        
#         summary_parts.append(f"- {metric['name']}: {value_str}{description}")
    
#     return "\n".join(summary_parts)


def get_metrics_summary(metrics: List[Dict[str, Any]], top_n: int = 5) -> str:
    """
    Generate a summary of the most important metrics.
    Modified to avoid timestamp confusion.
    
    Args:
        metrics: List of metric dictionaries
        top_n: Number of top metrics to include
    
    Returns:
        Summary string
    """
    if not metrics:
        return "No metrics available."
    
    # For simplicity, just take the first top_n metrics
    # In a real implementation, you would prioritize metrics by importance
    top_metrics = metrics[:min(top_n, len(metrics))]
    
    summary_parts = ["Key Company Metrics:"]
    for metric in top_metrics:
        value_str = format_metric_value(metric["value"], metric.get("unit", ""))
        
        # Use the description if available, removing any confusion with timestamps
        description = ""
        if metric.get("description"):
            description = f" - {metric['description']}"
        
        # Don't include timestamp information to avoid confusion
        
        summary_parts.append(f"- {metric['name']}: {value_str}{description}")
    
    return "\n".join(summary_parts)


def format_metric_value(value: float, unit: Optional[str] = None) -> str:
    """
    Format a metric value with appropriate precision and unit.
    
    Args:
        value: The metric value
        unit: Optional unit string
    
    Returns:
        Formatted value string
    """
    # Format large numbers with commas
    if abs(value) >= 1000:
        formatted_value = f"{value:,.0f}"
    # Format percentages
    elif unit and unit.lower() in ("%", "percent", "percentage"):
        formatted_value = f"{value:.2f}%"
    # Format currency
    elif unit and unit.lower() in ("$", "usd", "dollar", "dollars"):
        if abs(value) >= 1_000_000:
            formatted_value = f"${value/1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            formatted_value = f"${value/1_000:.2f}K"
        else:
            formatted_value = f"${value:.2f}"
    # Format ratios
    elif unit and unit.lower() in ("ratio", "x"):
        formatted_value = f"{value:.2f}x"
    # Format small decimal values
    elif abs(value) < 0.01:
        formatted_value = f"{value:.4f}"
    # Format medium decimal values
    elif abs(value) < 1:
        formatted_value = f"{value:.2f}"
    # Default formatting for other numbers
    else:
        formatted_value = f"{value:.2f}"
    
    # Add unit if provided and not already included
    if unit and unit.lower() not in ("%", "percent", "percentage", "$", "usd", "dollar", "dollars", "ratio", "x"):
        formatted_value = f"{formatted_value} {unit}"
    
    return formatted_value


def extract_metrics_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract potential metrics from unstructured text.
    This is a simple implementation that could be enhanced with NLP.
    
    Args:
        text: Unstructured text that might contain metrics
    
    Returns:
        List of extracted potential metrics
    """
    logger.info("Extracting metrics from text")
    
    # This is a placeholder implementation
    # In a real system, you would use NLP techniques to extract metrics
    
    # For now, just return an empty list
    return []


# def get_metrics_summary(metrics: List[Dict[str, Any]], top_n: int = 5) -> str:
#     """
#     Generate a summary of the most important metrics.
    
#     Args:
#         metrics: List of metric dictionaries
#         top_n: Number of top metrics to include
    
#     Returns:
#         Summary string
#     """
#     if not metrics:
#         return "No metrics available."
    
#     # For simplicity, just take the first top_n metrics
#     # In a real implementation, you would prioritize metrics by importance
#     top_metrics = metrics[:min(top_n, len(metrics))]
    
#     summary_parts = ["Key Company Metrics:"]
#     for metric in top_metrics:
#         value_str = format_metric_value(metric["value"], metric.get("unit", ""))
#         date_str = ""
#         if "timestamp" in metric:
#             try:
#                 dt = datetime.fromisoformat(metric["timestamp"])
#                 date_str = f" (as of {dt.strftime('%B %Y')})"
#             except (ValueError, TypeError):
#                 pass
        
#         summary_parts.append(f"- {metric['name']}: {value_str}{date_str}")
    
#     return "\n".join(summary_parts)