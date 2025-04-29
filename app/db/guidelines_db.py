"""
Database interface for storing and retrieving marketing guidelines.
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship, backref

from app.config import get_settings
from app.db.metrics_db import Base

logger = logging.getLogger(__name__)

# Reuse the Base and session from metrics_db
settings = get_settings()
engine = create_engine(settings.metrics_db.url, echo=settings.metrics_db.echo)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Guideline(Base):
    """Database model for marketing guidelines."""
    __tablename__ = "guidelines"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True, nullable=False)
    department = Column(String(100), index=True, nullable=False)
    subcategory = Column(String(100), index=True, nullable=True)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    source_url = Column(String(255), nullable=True)  # URL to access the source document
    source_s3_key = Column(String(255), nullable=True)  # S3 key for the source document
    source_content_type = Column(String(100), nullable=True)  # MIME type of source document
    has_assets = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_data = Column(Text, nullable=True)  # JSON string for additional data
    
    # Relationship with assets
    assets = relationship("GuidelineAsset", back_populates="guideline", cascade="all, delete-orphan")


class GuidelineAsset(Base):
    """Database model for guideline assets such as images."""
    __tablename__ = "guideline_assets"
    
    id = Column(Integer, primary_key=True, index=True)
    guideline_id = Column(Integer, ForeignKey("guidelines.id"), nullable=False)
    asset_type = Column(String(50), index=True, nullable=False)  # e.g., "image", "pdf", etc.
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    s3_key = Column(String(255), nullable=False)  # S3 object key
    content_type = Column(String(100), nullable=True)  # MIME type
    order = Column(Integer, default=0)  # For ordering assets
    created_at = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(Text, nullable=True)  # JSON string for additional data
    
    # Relationship with guideline
    guideline = relationship("Guideline", back_populates="assets")


def init_guidelines_db():
    """Initialize the guidelines database tables."""
    logger.info("Initializing guidelines database tables")
    Base.metadata.create_all(bind=engine)
    logger.info("Guidelines database tables initialized")


# CRUD operations for guidelines

def create_guideline(db: Session, guideline_data: Dict[str, Any]) -> Guideline:
    """
    Create a new guideline in the database.
    
    Args:
        db: Database session
        guideline_data: Dictionary containing guideline data
    
    Returns:
        Created guideline instance
    """
    # Handle metadata if provided
    if "metadata" in guideline_data and isinstance(guideline_data["metadata"], dict):
        guideline_data["meta_data"] = json.dumps(guideline_data["metadata"])
        del guideline_data["metadata"]
    
    db_guideline = Guideline(**guideline_data)
    db.add(db_guideline)
    db.commit()
    db.refresh(db_guideline)
    return db_guideline


def get_guidelines(
    db: Session, 
    department: Optional[str] = None,
    subcategory: Optional[str] = None,
    title: Optional[str] = None,
    limit: int = 100,
    skip: int = 0
) -> List[Guideline]:
    """
    Get guidelines from the database with optional filtering.
    
    Args:
        db: Database session
        department: Filter by department
        subcategory: Filter by subcategory
        title: Filter by title (partial match)
        limit: Maximum number of results
        skip: Number of results to skip
    
    Returns:
        List of guidelines
    """
    query = db.query(Guideline)
    
    if department:
        query = query.filter(Guideline.department == department)
    if subcategory:
        query = query.filter(Guideline.subcategory == subcategory)
    if title:
        query = query.filter(Guideline.title.like(f"%{title}%"))
    
    return query.order_by(Guideline.department, Guideline.subcategory, Guideline.title).offset(skip).limit(limit).all()


def get_guideline_by_id(db: Session, guideline_id: int) -> Optional[Guideline]:
    """
    Get a guideline by its ID.
    
    Args:
        db: Database session
        guideline_id: Guideline ID
    
    Returns:
        Guideline instance or None if not found
    """
    return db.query(Guideline).filter(Guideline.id == guideline_id).first()


def update_guideline(db: Session, guideline_id: int, guideline_data: Dict[str, Any]) -> Optional[Guideline]:
    """
    Update a guideline in the database.
    
    Args:
        db: Database session
        guideline_id: Guideline ID
        guideline_data: Dictionary containing updated guideline data
    
    Returns:
        Updated guideline instance or None if not found
    """
    db_guideline = get_guideline_by_id(db, guideline_id)
    if not db_guideline:
        return None
    
    # Handle metadata if provided
    if "metadata" in guideline_data and isinstance(guideline_data["metadata"], dict):
        guideline_data["meta_data"] = json.dumps(guideline_data["metadata"])
        del guideline_data["metadata"]
    
    # Update guideline attributes
    for key, value in guideline_data.items():
        if hasattr(db_guideline, key):
            setattr(db_guideline, key, value)
    
    db.commit()
    db.refresh(db_guideline)
    return db_guideline


def delete_guideline(db: Session, guideline_id: int) -> bool:
    """
    Delete a guideline from the database.
    
    Args:
        db: Database session
        guideline_id: Guideline ID
    
    Returns:
        True if deleted, False if not found
    """
    db_guideline = get_guideline_by_id(db, guideline_id)
    if not db_guideline:
        return False
    
    db.delete(db_guideline)
    db.commit()
    return True


# CRUD operations for guideline assets

def create_guideline_asset(db: Session, asset_data: Dict[str, Any]) -> GuidelineAsset:
    """
    Create a new guideline asset in the database.
    
    Args:
        db: Database session
        asset_data: Dictionary containing asset data
    
    Returns:
        Created asset instance
    """
    # Handle metadata if provided
    if "metadata" in asset_data and isinstance(asset_data["metadata"], dict):
        asset_data["meta_data"] = json.dumps(asset_data["metadata"])
        del asset_data["metadata"]
    
    db_asset = GuidelineAsset(**asset_data)
    db.add(db_asset)
    db.commit()
    db.refresh(db_asset)
    
    # Update the has_assets flag on the parent guideline
    guideline = get_guideline_by_id(db, db_asset.guideline_id)
    if guideline and not guideline.has_assets:
        guideline.has_assets = True
        db.commit()
    
    return db_asset


def get_guideline_assets(
    db: Session, 
    guideline_id: int,
    asset_type: Optional[str] = None,
    limit: int = 100,
    skip: int = 0
) -> List[GuidelineAsset]:
    """
    Get assets for a specific guideline.
    
    Args:
        db: Database session
        guideline_id: Guideline ID
        asset_type: Filter by asset type
        limit: Maximum number of results
        skip: Number of results to skip
    
    Returns:
        List of assets
    """
    query = db.query(GuidelineAsset).filter(GuidelineAsset.guideline_id == guideline_id)
    
    if asset_type:
        query = query.filter(GuidelineAsset.asset_type == asset_type)
    
    return query.order_by(GuidelineAsset.order, GuidelineAsset.id).offset(skip).limit(limit).all()


def get_asset_by_id(db: Session, asset_id: int) -> Optional[GuidelineAsset]:
    """
    Get an asset by its ID.
    
    Args:
        db: Database session
        asset_id: Asset ID
    
    Returns:
        Asset instance or None if not found
    """
    return db.query(GuidelineAsset).filter(GuidelineAsset.id == asset_id).first()


def update_asset(db: Session, asset_id: int, asset_data: Dict[str, Any]) -> Optional[GuidelineAsset]:
    """
    Update an asset in the database.
    
    Args:
        db: Database session
        asset_id: Asset ID
        asset_data: Dictionary containing updated asset data
    
    Returns:
        Updated asset instance or None if not found
    """
    db_asset = get_asset_by_id(db, asset_id)
    if not db_asset:
        return None
    
    # Handle metadata if provided
    if "metadata" in asset_data and isinstance(asset_data["metadata"], dict):
        asset_data["meta_data"] = json.dumps(asset_data["metadata"])
        del asset_data["metadata"]
    
    # Update asset attributes
    for key, value in asset_data.items():
        if hasattr(db_asset, key):
            setattr(db_asset, key, value)
    
    db.commit()
    db.refresh(db_asset)
    return db_asset


def delete_asset(db: Session, asset_id: int) -> bool:
    """
    Delete an asset from the database.
    
    Args:
        db: Database session
        asset_id: Asset ID
    
    Returns:
        True if deleted, False if not found
    """
    db_asset = get_asset_by_id(db, asset_id)
    if not db_asset:
        return False
    
    db.delete(db_asset)
    db.commit()
    
    # Check if the guideline has any remaining assets
    remaining_assets = get_guideline_assets(db, db_asset.guideline_id)
    if not remaining_assets:
        # Update the has_assets flag on the parent guideline
        guideline = get_guideline_by_id(db, db_asset.guideline_id)
        if guideline and guideline.has_assets:
            guideline.has_assets = False
            db.commit()
    
    return True


def search_guidelines(
    db: Session,
    query: str,
    limit: int = 10
) -> List[Guideline]:
    """
    Search guidelines by text query.
    
    Args:
        db: Database session
        query: Search query
        limit: Maximum number of results
    
    Returns:
        List of matching guidelines
    """
    # Basic text search implementation
    # For production, consider using full-text search capabilities of your database
    search_term = f"%{query}%"
    
    results = db.query(Guideline).filter(
        (Guideline.title.like(search_term)) |
        (Guideline.department.like(search_term)) |
        (Guideline.subcategory.like(search_term)) |
        (Guideline.description.like(search_term)) |
        (Guideline.content.like(search_term))
    ).limit(limit).all()
    
    return results