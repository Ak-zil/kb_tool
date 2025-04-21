"""
Metrics API endpoints for the Marketing Knowledge Base.
"""

import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.metrics_db import (
    get_db, 
    get_metrics, 
    get_metric_by_id, 
    create_metric, 
    update_metric, 
    delete_metric
)
from app.core.metrics_engine import format_metrics_for_context, get_metrics_summary

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for request/response
class MetricCreate(BaseModel):
    """Model for creating a new metric."""
    name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Value of the metric")
    unit: Optional[str] = Field(None, description="Unit of the metric (e.g., '%', '$')")
    category: str = Field(..., description="Category of the metric")
    subcategory: Optional[str] = Field(None, description="Subcategory of the metric")
    description: Optional[str] = Field(None, description="Description of the metric")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class MetricResponse(BaseModel):
    """Model for metric response."""
    id: int = Field(..., description="ID of the metric")
    name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Value of the metric")
    unit: Optional[str] = Field(None, description="Unit of the metric")
    category: str = Field(..., description="Category of the metric")
    subcategory: Optional[str] = Field(None, description="Subcategory of the metric")
    timestamp: str = Field(..., description="Timestamp when the metric was recorded")
    description: Optional[str] = Field(None, description="Description of the metric")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class MetricUpdate(BaseModel):
    """Model for updating a metric."""
    name: Optional[str] = Field(None, description="Name of the metric")
    value: Optional[float] = Field(None, description="Value of the metric")
    unit: Optional[str] = Field(None, description="Unit of the metric")
    category: Optional[str] = Field(None, description="Category of the metric")
    subcategory: Optional[str] = Field(None, description="Subcategory of the metric")
    description: Optional[str] = Field(None, description="Description of the metric")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class MetricsSummary(BaseModel):
    """Model for metrics summary."""
    summary: str = Field(..., description="Summary of key metrics")
    metrics_count: int = Field(..., description="Total number of metrics included")
    categories: List[str] = Field(..., description="List of metric categories included")


@router.get("/", response_model=List[MetricResponse], tags=["metrics"])
async def get_all_metrics(
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    name: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Get metrics with optional filtering.
    
    Args:
        category: Filter by category
        subcategory: Filter by subcategory
        name: Filter by metric name
        limit: Maximum number of results
        skip: Number of results to skip
        db: Database session
    
    Returns:
        List of metrics
    """
    logger.info(f"Getting metrics (category={category}, subcategory={subcategory}, name={name})")
    
    try:
        metrics = get_metrics(
            db=db,
            category=category,
            subcategory=subcategory,
            name=name,
            limit=limit,
            skip=skip
        )
        
        # Format response
        result = []
        for metric in metrics:
            metadata = {}
            if metric.metadata:
                import json
                try:
                    metadata = json.loads(metric.metadata)
                except json.JSONDecodeError:
                    metadata = {}
            
            result.append({
                "id": metric.id,
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "category": metric.category,
                "subcategory": metric.subcategory,
                "timestamp": metric.timestamp.isoformat(),
                "description": metric.description,
                "metadata": metadata
            })
        
        logger.info(f"Retrieved {len(result)} metrics")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


@router.get("/summary", response_model=MetricsSummary, tags=["metrics"])
async def get_metrics_summary_endpoint(
    top_n: int = Query(5, ge=1, le=20),
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get a summary of key metrics.
    
    Args:
        top_n: Number of top metrics to include
        category: Optional category filter
        db: Database session
    
    Returns:
        Summary of key metrics
    """
    logger.info(f"Getting metrics summary (top_n={top_n}, category={category})")
    
    try:
        # Get metrics
        metrics = get_metrics(
            db=db,
            category=category,
            limit=top_n
        )
        
        if not metrics:
            return {
                "summary": "No metrics available.",
                "metrics_count": 0,
                "categories": []
            }
        
        # Format metrics
        formatted_metrics = []
        categories = set()
        
        for metric in metrics:
            categories.add(metric.category)
            
            metadata = {}
            if metric.metadata:
                import json
                try:
                    metadata = json.loads(metric.metadata)
                except json.JSONDecodeError:
                    metadata = {}
            
            formatted_metrics.append({
                "id": metric.id,
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "category": metric.category,
                "subcategory": metric.subcategory,
                "timestamp": metric.timestamp.isoformat(),
                "description": metric.description,
                "metadata": metadata
            })
        
        # Generate summary
        summary = get_metrics_summary(formatted_metrics, top_n=top_n)
        
        logger.info(f"Generated metrics summary with {len(formatted_metrics)} metrics")
        return {
            "summary": summary,
            "metrics_count": len(formatted_metrics),
            "categories": list(categories)
        }
        
    except Exception as e:
        logger.error(f"Error generating metrics summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate metrics summary: {str(e)}"
        )


@router.get("/{metric_id}", response_model=MetricResponse, tags=["metrics"])
async def get_metric(metric_id: int, db: Session = Depends(get_db)):
    """
    Get a specific metric by ID.
    
    Args:
        metric_id: ID of the metric
        db: Database session
    
    Returns:
        Metric details
    """
    logger.info(f"Getting metric by ID: {metric_id}")
    
    metric = get_metric_by_id(db, metric_id)
    if not metric:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metric with ID {metric_id} not found"
        )
    
    # Format metadata
    metadata = {}
    if metric.metadata:
        import json
        try:
            metadata = json.loads(metric.metadata)
        except json.JSONDecodeError:
            metadata = {}
    
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


@router.post("/", response_model=MetricResponse, status_code=status.HTTP_201_CREATED, tags=["metrics"])
async def create_metric_endpoint(metric: MetricCreate, db: Session = Depends(get_db)):
    """
    Create a new metric.
    
    Args:
        metric: Metric data
        db: Database session
    
    Returns:
        Created metric
    """
    logger.info(f"Creating new metric: {metric.name}")
    
    try:
        db_metric = create_metric(db, metric.dict())
        
        # Format response
        metadata = {}
        if db_metric.metadata:
            import json
            try:
                metadata = json.loads(db_metric.metadata)
            except json.JSONDecodeError:
                metadata = {}
        
        return {
            "id": db_metric.id,
            "name": db_metric.name,
            "value": db_metric.value,
            "unit": db_metric.unit,
            "category": db_metric.category,
            "subcategory": db_metric.subcategory,
            "timestamp": db_metric.timestamp.isoformat(),
            "description": db_metric.description,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error creating metric: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create metric: {str(e)}"
        )


@router.put("/{metric_id}", response_model=MetricResponse, tags=["metrics"])
async def update_metric_endpoint(metric_id: int, metric: MetricUpdate, db: Session = Depends(get_db)):
    """
    Update a metric.
    
    Args:
        metric_id: ID of the metric to update
        metric: Updated metric data
        db: Database session
    
    Returns:
        Updated metric
    """
    logger.info(f"Updating metric with ID: {metric_id}")
    
    try:
        db_metric = update_metric(db, metric_id, metric.dict(exclude_unset=True))
        if not db_metric:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Metric with ID {metric_id} not found"
            )
        
        # Format response
        metadata = {}
        if db_metric.metadata:
            import json
            try:
                metadata = json.loads(db_metric.metadata)
            except json.JSONDecodeError:
                metadata = {}
        
        return {
            "id": db_metric.id,
            "name": db_metric.name,
            "value": db_metric.value,
            "unit": db_metric.unit,
            "category": db_metric.category,
            "subcategory": db_metric.subcategory,
            "timestamp": db_metric.timestamp.isoformat(),
            "description": db_metric.description,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating metric: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update metric: {str(e)}"
        )


@router.delete("/{metric_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["metrics"])
async def delete_metric_endpoint(metric_id: int, db: Session = Depends(get_db)):
    """
    Delete a metric.
    
    Args:
        metric_id: ID of the metric to delete
        db: Database session
    
    Returns:
        No content
    """
    logger.info(f"Deleting metric with ID: {metric_id}")
    
    result = delete_metric(db, metric_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metric with ID {metric_id} not found"
        )
    
    return None


@router.get("/formatted", tags=["metrics"])
async def get_formatted_metrics(
    category: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get formatted metrics for display.
    
    Args:
        category: Optional category filter
        limit: Maximum number of metrics to include
        db: Database session
    
    Returns:
        Formatted metrics as text
    """
    logger.info(f"Getting formatted metrics (category={category}, limit={limit})")
    
    try:
        # Get metrics
        metrics = get_metrics(
            db=db,
            category=category,
            limit=limit
        )
        
        if not metrics:
            return {"formatted_text": "No metrics available."}
        
        # Format metrics
        formatted_metrics = []
        
        for metric in metrics:
            metadata = {}
            if metric.metadata:
                import json
                try:
                    metadata = json.loads(metric.metadata)
                except json.JSONDecodeError:
                    metadata = {}
            
            formatted_metrics.append({
                "id": metric.id,
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "category": metric.category,
                "subcategory": metric.subcategory,
                "timestamp": metric.timestamp.isoformat(),
                "description": metric.description,
                "metadata": metadata
            })
        
        # Generate formatted text
        formatted_text = format_metrics_for_context(formatted_metrics)
        
        logger.info(f"Generated formatted metrics with {len(formatted_metrics)} metrics")
        return {"formatted_text": formatted_text}
        
    except Exception as e:
        logger.error(f"Error generating formatted metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate formatted metrics: {str(e)}"
        )