"""
Metrics database interface for storing and retrieving company metrics and figures.
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app.config import get_settings

logger = logging.getLogger(__name__)

# Create database engine
settings = get_settings()
engine = create_engine(settings.metrics_db.url, echo=settings.metrics_db.echo)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base model class
Base = declarative_base()


class Metric(Base):
    """Database model for company metrics."""
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True, nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)
    category = Column(String(100), index=True, nullable=False)
    subcategory = Column(String(100), index=True, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    description = Column(Text, nullable=True)
    metric_metadata = Column(Text, nullable=True)  # JSON string for additional data


class ChatSummary(Base):
    """Database model for recent chat summaries."""
    __tablename__ = "chat_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    summary = Column(Text, nullable=False)
    topics = Column(Text, nullable=True)  # JSON string of topics
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    chat_summary_metadata = Column(Text, nullable=True)  # JSON string for additional data


def init_db():
    """Initialize the database by creating all tables."""
    logger.info("Initializing metrics database")
    Base.metadata.create_all(bind=engine)
    logger.info("Metrics database initialized")


def get_db():
    """
    Get database session.
    Use as a dependency in FastAPI endpoints.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Metric CRUD operations

def create_metric(db: Session, metric_data: Dict[str, Any]) -> Metric:
    """
    Create a new metric in the database.
    
    Args:
        db: Database session
        metric_data: Dictionary containing metric data
    
    Returns:
        Created metric instance
    """
    # Handle metadata if provided
    if "metadata" in metric_data and isinstance(metric_data["metadata"], dict):
        metric_data["metadata"] = json.dumps(metric_data["metadata"])
    
    db_metric = Metric(**metric_data)
    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)
    return db_metric


def get_metrics(
    db: Session, 
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    name: Optional[str] = None,
    limit: int = 100,
    skip: int = 0
) -> List[Metric]:
    """
    Get metrics from the database with optional filtering.
    
    Args:
        db: Database session
        category: Filter by category
        subcategory: Filter by subcategory
        name: Filter by metric name
        limit: Maximum number of results
        skip: Number of results to skip
    
    Returns:
        List of metrics
    """
    query = db.query(Metric)
    
    if category:
        query = query.filter(Metric.category == category)
    if subcategory:
        query = query.filter(Metric.subcategory == subcategory)
    if name:
        query = query.filter(Metric.name == name)
    
    return query.order_by(desc(Metric.timestamp)).offset(skip).limit(limit).all()


def get_metric_by_id(db: Session, metric_id: int) -> Optional[Metric]:
    """
    Get a metric by its ID.
    
    Args:
        db: Database session
        metric_id: Metric ID
    
    Returns:
        Metric instance or None if not found
    """
    return db.query(Metric).filter(Metric.id == metric_id).first()


def update_metric(db: Session, metric_id: int, metric_data: Dict[str, Any]) -> Optional[Metric]:
    """
    Update a metric in the database.
    
    Args:
        db: Database session
        metric_id: Metric ID
        metric_data: Dictionary containing updated metric data
    
    Returns:
        Updated metric instance or None if not found
    """
    db_metric = get_metric_by_id(db, metric_id)
    if not db_metric:
        return None
    
    # Handle metadata if provided
    if "metadata" in metric_data and isinstance(metric_data["metadata"], dict):
        metric_data["metadata"] = json.dumps(metric_data["metadata"])
    
    for key, value in metric_data.items():
        setattr(db_metric, key, value)
    
    db.commit()
    db.refresh(db_metric)
    return db_metric


def delete_metric(db: Session, metric_id: int) -> bool:
    """
    Delete a metric from the database.
    
    Args:
        db: Database session
        metric_id: Metric ID
    
    Returns:
        True if deleted, False if not found
    """
    db_metric = get_metric_by_id(db, metric_id)
    if not db_metric:
        return False
    
    db.delete(db_metric)
    db.commit()
    return True


# Chat summary CRUD operations

def create_chat_summary(db: Session, summary_data: Dict[str, Any]) -> ChatSummary:
    """
    Create a new chat summary in the database.
    
    Args:
        db: Database session
        summary_data: Dictionary containing summary data
    
    Returns:
        Created chat summary instance
    """
    # Handle topics and metadata if provided as dictionaries
    if "topics" in summary_data and isinstance(summary_data["topics"], (list, dict)):
        summary_data["topics"] = json.dumps(summary_data["topics"])
    
    if "metadata" in summary_data and isinstance(summary_data["metadata"], dict):
        summary_data["metadata"] = json.dumps(summary_data["metadata"])
    
    db_summary = ChatSummary(**summary_data)
    db.add(db_summary)
    db.commit()
    db.refresh(db_summary)
    return db_summary


def get_chat_summaries(db: Session, limit: int = 20, skip: int = 0) -> List[ChatSummary]:
    """
    Get chat summaries from the database.
    
    Args:
        db: Database session
        limit: Maximum number of results
        skip: Number of results to skip
    
    Returns:
        List of chat summaries
    """
    return db.query(ChatSummary).order_by(desc(ChatSummary.timestamp)).offset(skip).limit(limit).all()


def get_chat_summary_by_id(db: Session, summary_id: int) -> Optional[ChatSummary]:
    """
    Get a chat summary by its ID.
    
    Args:
        db: Database session
        summary_id: Summary ID
    
    Returns:
        Chat summary instance or None if not found
    """
    return db.query(ChatSummary).filter(ChatSummary.id == summary_id).first()


def update_chat_summary(db: Session, summary_id: int, summary_data: Dict[str, Any]) -> Optional[ChatSummary]:
    """
    Update a chat summary in the database.
    
    Args:
        db: Database session
        summary_id: Summary ID
        summary_data: Dictionary containing updated summary data
    
    Returns:
        Updated chat summary instance or None if not found
    """
    db_summary = get_chat_summary_by_id(db, summary_id)
    if not db_summary:
        return None
    
    # Handle topics and metadata if provided as dictionaries
    if "topics" in summary_data and isinstance(summary_data["topics"], (list, dict)):
        summary_data["topics"] = json.dumps(summary_data["topics"])
    
    if "metadata" in summary_data and isinstance(summary_data["metadata"], dict):
        summary_data["metadata"] = json.dumps(summary_data["metadata"])
    
    for key, value in summary_data.items():
        setattr(db_summary, key, value)
    
    db.commit()
    db.refresh(db_summary)
    return db_summary


def delete_chat_summary(db: Session, summary_id: int) -> bool:
    """
    Delete a chat summary from the database.
    
    Args:
        db: Database session
        summary_id: Summary ID
    
    Returns:
        True if deleted, False if not found
    """
    db_summary = get_chat_summary_by_id(db, summary_id)
    if not db_summary:
        return False
    
    db.delete(db_summary)
    db.commit()
    return True