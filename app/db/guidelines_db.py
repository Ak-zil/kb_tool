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
    content = Column(Text, nullable=True)  # Extracted content for text-only guidelines
    external_source_url = Column(String(255), nullable=True)  # Optional link to an external resource
    source_url = Column(String(255), nullable=True)  # URL to access the source document
    source_s3_key = Column(String(255), nullable=True)  # S3 key for the source document
    source_content_type = Column(String(100), nullable=True)  # MIME type of source document
    is_text_only = Column(Boolean, default=True)  # Flag to indicate if this is a text-only guideline
    is_processed = Column(Boolean, default=False)  # Flag to indicate if text-only document has been processed
    has_assets = Column(Boolean, default=False)  # If guideline has direct assets
    has_sections = Column(Boolean, default=False)  # If guideline has sections
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_data = Column(Text, nullable=True)  # JSON string for additional data
    
    # Relationship with assets
    assets = relationship("GuidelineAsset", back_populates="guideline", cascade="all, delete-orphan")
    
    # Relationship with sections
    sections = relationship("GuidelineSection", back_populates="guideline", cascade="all, delete-orphan")


class GuidelineSection(Base):
    """Database model for guideline sections."""
    __tablename__ = "guideline_sections"
    
    id = Column(Integer, primary_key=True, index=True)
    guideline_id = Column(Integer, ForeignKey("guidelines.id"), nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=True)  # Text content of the section
    order = Column(Integer, default=0)  # For ordering sections within a guideline
    has_assets = Column(Boolean, default=False)  # Flag to indicate if section has assets
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_data = Column(Text, nullable=True)  # JSON string for additional data
    
    # Relationship with guideline
    guideline = relationship("Guideline", back_populates="sections")
    
    # Relationship with section assets
    assets = relationship("SectionAsset", back_populates="section", cascade="all, delete-orphan")


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


class SectionAsset(Base):
    """Database model for section-specific assets."""
    __tablename__ = "section_assets"
    
    id = Column(Integer, primary_key=True, index=True)
    section_id = Column(Integer, ForeignKey("guideline_sections.id"), nullable=False)
    asset_type = Column(String(50), index=True, nullable=False)  # e.g., "image", "pdf", etc.
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    s3_key = Column(String(255), nullable=False)  # S3 object key
    content_type = Column(String(100), nullable=True)  # MIME type
    order = Column(Integer, default=0)  # For ordering assets
    created_at = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(Text, nullable=True)  # JSON string for additional data
    
    # Relationship with section
    section = relationship("GuidelineSection", back_populates="assets")


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
    
    # Create a dictionary with only the fields that exist in the model
    valid_fields = {
        "title", "department", "subcategory", "description", "content", 
        "external_source_url", "source_url", "source_s3_key", "source_content_type", 
        "is_text_only", "is_processed", "has_assets", "has_sections", "meta_data"
    }
    
    filtered_data = {k: v for k, v in guideline_data.items() if k in valid_fields}
    
    # Create the guideline
    db_guideline = Guideline(**filtered_data)
    db.add(db_guideline)
    db.commit()
    db.refresh(db_guideline)
    return db_guideline


def get_guidelines(
    db: Session, 
    department: Optional[str] = None,
    subcategory: Optional[str] = None,
    title: Optional[str] = None,
    is_text_only: Optional[bool] = None,
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
        is_text_only: Filter by whether guideline is text-only
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
    if is_text_only is not None:
        query = query.filter(Guideline.is_text_only == is_text_only)
    
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
    
    # Create a dictionary with only the fields that exist in the model
    valid_fields = {
        "title", "department", "subcategory", "description", "content", 
        "external_source_url", "source_url", "source_s3_key", "source_content_type", 
        "is_text_only", "is_processed", "has_assets", "has_sections", "meta_data"
    }
    
    filtered_data = {k: v for k, v in guideline_data.items() if k in valid_fields}
    
    # Update guideline attributes
    for key, value in filtered_data.items():
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


# CRUD operations for guideline sections

def create_guideline_section(db: Session, section_data: Dict[str, Any]) -> GuidelineSection:
    """
    Create a new guideline section in the database.
    
    Args:
        db: Database session
        section_data: Dictionary containing section data
    
    Returns:
        Created section instance
    """
    # Handle metadata if provided
    if "metadata" in section_data and isinstance(section_data["metadata"], dict):
        section_data["meta_data"] = json.dumps(section_data["metadata"])
        del section_data["metadata"]
    
    db_section = GuidelineSection(**section_data)
    db.add(db_section)
    db.commit()
    db.refresh(db_section)
    
    # Update the has_sections flag on the parent guideline
    guideline = get_guideline_by_id(db, db_section.guideline_id)
    if guideline and not guideline.has_sections:
        guideline.has_sections = True
        guideline.is_text_only = False  # If it has sections, it's not text-only
        db.commit()
    
    return db_section


def get_guideline_sections(
    db: Session, 
    guideline_id: int,
    limit: int = 100,
    skip: int = 0
) -> List[GuidelineSection]:
    """
    Get sections for a specific guideline.
    
    Args:
        db: Database session
        guideline_id: Guideline ID
        limit: Maximum number of results
        skip: Number of results to skip
    
    Returns:
        List of sections
    """
    return db.query(GuidelineSection).filter(
        GuidelineSection.guideline_id == guideline_id
    ).order_by(GuidelineSection.order, GuidelineSection.id).offset(skip).limit(limit).all()


def get_section_by_id(db: Session, section_id: int) -> Optional[GuidelineSection]:
    """
    Get a section by its ID.
    
    Args:
        db: Database session
        section_id: Section ID
    
    Returns:
        Section instance or None if not found
    """
    return db.query(GuidelineSection).filter(GuidelineSection.id == section_id).first()


def update_section(db: Session, section_id: int, section_data: Dict[str, Any]) -> Optional[GuidelineSection]:
    """
    Update a section in the database.
    
    Args:
        db: Database session
        section_id: Section ID
        section_data: Dictionary containing updated section data
    
    Returns:
        Updated section instance or None if not found
    """
    db_section = get_section_by_id(db, section_id)
    if not db_section:
        return None
    
    # Handle metadata if provided
    if "metadata" in section_data and isinstance(section_data["metadata"], dict):
        section_data["meta_data"] = json.dumps(section_data["metadata"])
        del section_data["metadata"]
    
    # Update section attributes
    for key, value in section_data.items():
        if hasattr(db_section, key):
            setattr(db_section, key, value)
    
    db.commit()
    db.refresh(db_section)
    return db_section


def delete_section(db: Session, section_id: int) -> bool:
    """
    Delete a section from the database.
    
    Args:
        db: Database session
        section_id: Section ID
    
    Returns:
        True if deleted, False if not found
    """
    db_section = get_section_by_id(db, section_id)
    if not db_section:
        return False
    
    guideline_id = db_section.guideline_id
    
    db.delete(db_section)
    db.commit()
    
    # Check if the guideline has any remaining sections
    remaining_sections = get_guideline_sections(db, guideline_id)
    if not remaining_sections:
        # Update the has_sections flag on the parent guideline
        guideline = get_guideline_by_id(db, guideline_id)
        if guideline and guideline.has_sections:
            guideline.has_sections = False
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


# CRUD operations for section assets

def create_section_asset(db: Session, asset_data: Dict[str, Any]) -> SectionAsset:
    """
    Create a new section asset in the database.
    
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
    
    db_asset = SectionAsset(**asset_data)
    db.add(db_asset)
    db.commit()
    db.refresh(db_asset)
    
    # Update the has_assets flag on the parent section
    section = get_section_by_id(db, db_asset.section_id)
    if section and not section.has_assets:
        section.has_assets = True
        db.commit()
    
    return db_asset


def get_section_assets(
    db: Session, 
    section_id: int,
    asset_type: Optional[str] = None,
    limit: int = 100,
    skip: int = 0
) -> List[SectionAsset]:
    """
    Get assets for a specific section.
    
    Args:
        db: Database session
        section_id: Section ID
        asset_type: Filter by asset type
        limit: Maximum number of results
        skip: Number of results to skip
    
    Returns:
        List of assets
    """
    query = db.query(SectionAsset).filter(SectionAsset.section_id == section_id)
    
    if asset_type:
        query = query.filter(SectionAsset.asset_type == asset_type)
    
    return query.order_by(SectionAsset.order, SectionAsset.id).offset(skip).limit(limit).all()


def get_section_asset_by_id(db: Session, asset_id: int) -> Optional[SectionAsset]:
    """
    Get a section asset by its ID.
    
    Args:
        db: Database session
        asset_id: Asset ID
    
    Returns:
        Asset instance or None if not found
    """
    return db.query(SectionAsset).filter(SectionAsset.id == asset_id).first()


def update_section_asset(db: Session, asset_id: int, asset_data: Dict[str, Any]) -> Optional[SectionAsset]:
    """
    Update a section asset in the database.
    
    Args:
        db: Database session
        asset_id: Asset ID
        asset_data: Dictionary containing updated asset data
    
    Returns:
        Updated asset instance or None if not found
    """
    db_asset = get_section_asset_by_id(db, asset_id)
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


def delete_section_asset(db: Session, asset_id: int) -> bool:
    """
    Delete a section asset from the database.
    
    Args:
        db: Database session
        asset_id: Asset ID
    
    Returns:
        True if deleted, False if not found
    """
    db_asset = get_section_asset_by_id(db, asset_id)
    if not db_asset:
        return False
    
    section_id = db_asset.section_id
    
    db.delete(db_asset)
    db.commit()
    
    # Check if the section has any remaining assets
    remaining_assets = get_section_assets(db, section_id)
    if not remaining_assets:
        # Update the has_assets flag on the parent section
        section = get_section_by_id(db, section_id)
        if section and section.has_assets:
            section.has_assets = False
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


def get_guideline_with_sections(
    db: Session,
    guideline_id: int
) -> Optional[Dict[str, Any]]:
    """
    Get a guideline with all its sections and assets.
    
    Args:
        db: Database session
        guideline_id: Guideline ID
    
    Returns:
        Dictionary containing guideline data with sections and assets, or None if not found
    """
    guideline = get_guideline_by_id(db, guideline_id)
    if not guideline:
        return None
    
    # Format guideline data
    guideline_data = {
        "id": guideline.id,
        "title": guideline.title,
        "department": guideline.department,
        "subcategory": guideline.subcategory,
        "description": guideline.description,
        "content": guideline.content,
        "external_source_url": guideline.external_source_url,
        "source_url": guideline.source_url,
        "source_content_type": guideline.source_content_type,
        "has_source_document": bool(guideline.source_s3_key),
        "is_text_only": guideline.is_text_only,
        "is_processed": guideline.is_processed,
        "has_assets": guideline.has_assets,
        "has_sections": guideline.has_sections,
        "created_at": guideline.created_at.isoformat(),
        "updated_at": guideline.updated_at.isoformat(),
        "sections": [],
        "assets": []
    }
    
    # Add sections if available
    if guideline.has_sections:
        sections = get_guideline_sections(db, guideline_id)
        for section in sections:
            section_data = {
                "id": section.id,
                "title": section.title,
                "content": section.content,
                "order": section.order,
                "has_assets": section.has_assets,
                "assets": []
            }
            
            # Add section assets if available
            if section.has_assets:
                section_assets = get_section_assets(db, section.id)
                for asset in section_assets:
                    # Parse metadata if available
                    metadata = {}
                    if asset.meta_data:
                        try:
                            metadata = json.loads(asset.meta_data)
                        except json.JSONDecodeError:
                            pass
                    
                    section_data["assets"].append({
                        "id": asset.id,
                        "asset_type": asset.asset_type,
                        "title": asset.title,
                        "description": asset.description,
                        "s3_key": asset.s3_key,
                        "content_type": asset.content_type,
                        "order": asset.order,
                        "metadata": metadata
                    })
            
            guideline_data["sections"].append(section_data)
    
    # Add guideline assets if available
    if guideline.has_assets:
        assets = get_guideline_assets(db, guideline_id)
        for asset in assets:
            # Parse metadata if available
            metadata = {}
            if asset.meta_data:
                try:
                    metadata = json.loads(asset.meta_data)
                except json.JSONDecodeError:
                    pass
                
            guideline_data["assets"].append({
                "id": asset.id,
                "asset_type": asset.asset_type,
                "title": asset.title,
                "description": asset.description,
                "s3_key": asset.s3_key,
                "content_type": asset.content_type,
                "order": asset.order,
                "metadata": metadata
            })
    
    return guideline_data


def mark_guideline_processed(db: Session, guideline_id: int, extracted_content: str) -> bool:
    """
    Mark a text-only guideline as processed and update its content with extracted text.
    
    Args:
        db: Database session
        guideline_id: Guideline ID
        extracted_content: Extracted text content from the document
    
    Returns:
        True if successful, False if guideline not found
    """
    guideline = get_guideline_by_id(db, guideline_id)
    if not guideline:
        return False
        
    guideline.is_processed = True
    guideline.content = extracted_content
    db.commit()
    return True


def get_unprocessed_text_guidelines(db: Session, limit: int = 10) -> List[Guideline]:
    """
    Get text-only guidelines that haven't been processed yet.
    
    Args:
        db: Database session
        limit: Maximum number of results
    
    Returns:
        List of unprocessed guidelines
    """
    return db.query(Guideline).filter(
        Guideline.is_text_only is True,
        Guideline.is_processed is False,
        Guideline.source_s3_key.isnot(None)
    ).limit(limit).all()