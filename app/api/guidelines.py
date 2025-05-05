"""
API endpoints for managing marketing guidelines.
"""

import logging
import json
import mimetypes
from typing import Dict, Any, Optional

from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    status, 
    UploadFile, 
    File, 
    Form,
    BackgroundTasks,
    Path,
    Query
)
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.metrics_db import get_db
from app.db.guidelines_db import (
    create_guideline,
    get_guidelines,
    get_guideline_by_id,
    update_guideline,
    delete_guideline,
    create_guideline_asset,
    get_guideline_assets,
    get_asset_by_id,
    update_asset,
    delete_asset,
    search_guidelines,
    # Section-related functions
    create_guideline_section,
    get_guideline_sections,
    get_section_by_id,
    update_section,
    delete_section,
    # Section asset-related functions
    create_section_asset,
    get_section_assets,
    get_section_asset_by_id,
    update_section_asset,
    delete_section_asset,
    # Get guideline with sections
    get_guideline_with_sections
)
from app.utils.s3_client import get_s3_client

logger = logging.getLogger(__name__)

router = APIRouter()

# Ensure mimetypes are initialized
mimetypes.init()


# Pydantic models for request/response
class GuidelineCreate(BaseModel):
    """Model for creating a new guideline."""
    title: str = Field(..., description="Title of the guideline")
    department: str = Field(..., description="Department the guideline belongs to")
    subcategory: Optional[str] = Field(None, description="Subcategory within the department")
    description: Optional[str] = Field(None, description="Brief description of the guideline")
    content: Optional[str] = Field(None, description="Main content of the guideline")
    external_source_url: Optional[str] = Field(None, description="Optional link to an external resource (not for uploaded files)")
    is_text_only: bool = Field(True, description="Whether this is a text-only guideline or sectioned with images")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class GuidelineResponse(BaseModel):
    """Model for guideline response."""
    id: int = Field(..., description="ID of the guideline")
    title: str = Field(..., description="Title of the guideline")
    department: str = Field(..., description="Department the guideline belongs to")
    subcategory: Optional[str] = Field(None, description="Subcategory within the department")
    description: Optional[str] = Field(None, description="Brief description of the guideline")
    content: Optional[str] = Field(None, description="Main content of the guideline")
    external_source_url: Optional[str] = Field(None, description="Link to an external resource")
    source_url: Optional[str] = Field(None, description="URL to access the source document")
    source_content_type: Optional[str] = Field(None, description="MIME type of the source document")
    has_source_document: bool = Field(False, description="Whether the guideline has a source document")
    has_assets: bool = Field(..., description="Whether the guideline has associated assets")
    is_text_only: bool = Field(True, description="Whether this is a text-only guideline or sectioned with images")
    is_processed: bool = Field(False, description="Whether text-only guideline has been processed")
    has_sections: bool = Field(False, description="Whether the guideline has associated sections")
    created_at: str = Field(..., description="Timestamp of creation")
    updated_at: str = Field(..., description="Timestamp of last update")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class GuidelineUpdate(BaseModel):
    """Model for updating a guideline."""
    title: Optional[str] = Field(None, description="Title of the guideline")
    department: Optional[str] = Field(None, description="Department the guideline belongs to")
    subcategory: Optional[str] = Field(None, description="Subcategory within the department")
    description: Optional[str] = Field(None, description="Brief description of the guideline")
    content: Optional[str] = Field(None, description="Main content of the guideline")
    source_url: Optional[str] = Field(None, description="Source URL for reference")
    is_text_only: Optional[bool] = Field(None, description="Whether this is a text-only guideline")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AssetCreate(BaseModel):
    """Model for creating a new asset."""
    guideline_id: int = Field(..., description="ID of the parent guideline")
    asset_type: str = Field(..., description="Type of asset (e.g., 'image', 'pdf')")
    title: Optional[str] = Field(None, description="Title of the asset")
    description: Optional[str] = Field(None, description="Description of the asset")
    order: Optional[int] = Field(0, description="Order of the asset within the guideline")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AssetResponse(BaseModel):
    """Model for asset response."""
    id: int = Field(..., description="ID of the asset")
    guideline_id: int = Field(..., description="ID of the parent guideline")
    asset_type: str = Field(..., description="Type of asset")
    title: Optional[str] = Field(None, description="Title of the asset")
    description: Optional[str] = Field(None, description="Description of the asset")
    s3_key: str = Field(..., description="S3 object key")
    content_type: Optional[str] = Field(None, description="MIME type of the asset")
    order: int = Field(..., description="Order of the asset within the guideline")
    url: str = Field(..., description="URL to access the asset")
    created_at: str = Field(..., description="Timestamp of creation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AssetUpdate(BaseModel):
    """Model for updating an asset."""
    title: Optional[str] = Field(None, description="Title of the asset")
    description: Optional[str] = Field(None, description="Description of the asset")
    order: Optional[int] = Field(None, description="Order of the asset within the guideline")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SectionCreate(BaseModel):
    """Model for creating a new guideline section."""
    guideline_id: int = Field(..., description="ID of the parent guideline")
    title: str = Field(..., description="Title of the section")
    content: Optional[str] = Field(None, description="Content of the section")
    order: Optional[int] = Field(0, description="Order of the section within the guideline")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SectionResponse(BaseModel):
    """Model for section response."""
    id: int = Field(..., description="ID of the section")
    guideline_id: int = Field(..., description="ID of the parent guideline")
    title: str = Field(..., description="Title of the section")
    content: Optional[str] = Field(None, description="Content of the section")
    order: int = Field(..., description="Order of the section within the guideline")
    has_assets: bool = Field(False, description="Whether the section has associated assets")
    created_at: str = Field(..., description="Timestamp of creation")
    updated_at: str = Field(..., description="Timestamp of last update")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SectionUpdate(BaseModel):
    """Model for updating a section."""
    title: Optional[str] = Field(None, description="Title of the section")
    content: Optional[str] = Field(None, description="Content of the section")
    order: Optional[int] = Field(None, description="Order of the section within the guideline")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SectionAssetCreate(BaseModel):
    """Model for creating a new section asset."""
    section_id: int = Field(..., description="ID of the parent section")
    asset_type: str = Field(..., description="Type of asset (e.g., 'image', 'pdf')")
    title: Optional[str] = Field(None, description="Title of the asset")
    description: Optional[str] = Field(None, description="Description of the asset")
    order: Optional[int] = Field(0, description="Order of the asset within the section")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SectionAssetResponse(BaseModel):
    """Model for section asset response."""
    id: int = Field(..., description="ID of the asset")
    section_id: int = Field(..., description="ID of the parent section")
    asset_type: str = Field(..., description="Type of asset")
    title: Optional[str] = Field(None, description="Title of the asset")
    description: Optional[str] = Field(None, description="Description of the asset")
    s3_key: str = Field(..., description="S3 object key")
    content_type: Optional[str] = Field(None, description="MIME type of the asset")
    order: int = Field(..., description="Order of the asset within the section")
    url: str = Field(..., description="URL to access the asset")
    created_at: str = Field(..., description="Timestamp of creation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SectionAssetUpdate(BaseModel):
    """Model for updating a section asset."""
    title: Optional[str] = Field(None, description="Title of the asset")
    description: Optional[str] = Field(None, description="Description of the asset")
    order: Optional[int] = Field(None, description="Order of the asset within the section")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class GuidelineWithAssets(GuidelineResponse):
    """Model for guideline with its assets."""
    assets: list[AssetResponse] = Field([], description="Associated assets")
    sections: list[Dict[str, Any]] = Field([], description="Associated sections")


# Utility functions
def format_guideline_response(guideline) -> Dict[str, Any]:
    """Format a guideline object for API response."""
    # Parse metadata if available
    metadata = {}
    if guideline.meta_data:
        try:
            metadata = json.loads(guideline.meta_data)
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Generate fresh presigned URL for source document if needed
    source_url = guideline.source_url
    if guideline.source_s3_key:
        s3_client = get_s3_client()
        source_url = s3_client.get_presigned_url(guideline.source_s3_key, expiration=86400)  # 1 day
    
    return {
        "id": guideline.id,
        "title": guideline.title,
        "department": guideline.department,
        "subcategory": guideline.subcategory,
        "description": guideline.description,
        "content": guideline.content,
        "external_source_url": guideline.external_source_url,
        "source_url": source_url,
        "source_content_type": guideline.source_content_type,
        "has_source_document": bool(guideline.source_s3_key),
        "has_assets": guideline.has_assets,
        "is_text_only": guideline.is_text_only,
        "is_processed": guideline.is_processed,
        "has_sections": guideline.has_sections,
        "created_at": guideline.created_at.isoformat(),
        "updated_at": guideline.updated_at.isoformat(),
        "metadata": metadata
    }


def format_asset_response(asset, include_url: bool = True) -> Dict[str, Any]:
    """Format an asset object for API response."""
    # Parse metadata if available
    metadata = {}
    if asset.meta_data:
        try:
            metadata = json.loads(asset.meta_data)
        except (json.JSONDecodeError, TypeError):
            pass
    
    response = {
        "id": asset.id,
        "guideline_id": asset.guideline_id,
        "asset_type": asset.asset_type,
        "title": asset.title,
        "description": asset.description,
        "s3_key": asset.s3_key,
        "content_type": asset.content_type,
        "order": asset.order,
        "created_at": asset.created_at.isoformat(),
        "metadata": metadata
    }
    
    # Add URL if requested
    if include_url:
        s3_client = get_s3_client()
        url = s3_client.get_presigned_url(asset.s3_key)
        response["url"] = url or ""
    
    return response


def format_section_response(section) -> Dict[str, Any]:
    """Format a section object for API response."""
    # Parse metadata if available
    metadata = {}
    if section.meta_data:
        try:
            metadata = json.loads(section.meta_data)
        except (json.JSONDecodeError, TypeError):
            pass
    
    return {
        "id": section.id,
        "guideline_id": section.guideline_id,
        "title": section.title,
        "content": section.content,
        "order": section.order,
        "has_assets": section.has_assets,
        "created_at": section.created_at.isoformat(),
        "updated_at": section.updated_at.isoformat(),
        "metadata": metadata
    }


def format_section_asset_response(asset, include_url: bool = True) -> Dict[str, Any]:
    """Format a section asset object for API response."""
    # Parse metadata if available
    metadata = {}
    if asset.meta_data:
        try:
            metadata = json.loads(asset.meta_data)
        except (json.JSONDecodeError, TypeError):
            pass
    
    response = {
        "id": asset.id,
        "section_id": asset.section_id,
        "asset_type": asset.asset_type,
        "title": asset.title,
        "description": asset.description,
        "s3_key": asset.s3_key,
        "content_type": asset.content_type,
        "order": asset.order,
        "created_at": asset.created_at.isoformat(),
        "metadata": metadata
    }
    
    # Add URL if requested
    if include_url:
        s3_client = get_s3_client()
        url = s3_client.get_presigned_url(asset.s3_key)
        response["url"] = url or ""
    
    return response


# API Endpoints for Guidelines
# API Endpoints for Guidelines
@router.post("/", response_model=GuidelineResponse, status_code=status.HTTP_201_CREATED, tags=["guidelines"])
async def create_guideline_endpoint(
    title: str = Form(...),
    department: str = Form(...),
    subcategory: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    external_source_url: Optional[str] = Form(None),
    is_text_only: Optional[str] = Form("true"),
    metadata_json: Optional[str] = Form(None),
    source_file: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Create a new marketing guideline with optional source document.
    
    For text-only guidelines (is_text_only=true), the source document will be processed 
    to extract content for the LLM.
    
    For sectioned guidelines (is_text_only=false), the source document is optional and
    serves only as a reference file.
    
    Args:
        title: Title of the guideline
        department: Department the guideline belongs to
        subcategory: Subcategory within the department
        description: Brief description of the guideline
        content: Main content of the guideline (optional)
        external_source_url: Optional external URL reference (e.g., link to company website)
        is_text_only: Whether this is a text-only guideline (default: true)
        metadata_json: Additional metadata as JSON string
        source_file: Source document file (required for text-only guidelines)
        background_tasks: Background tasks manager
        db: Database session
    
    Returns:
        Created guideline
    """
    logger.info(f"Creating new guideline: {title}")
    
    try:
        # Parse metadata if provided
        metadata = {}
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata JSON: {metadata_json}")
        
        # Convert is_text_only string to boolean
        is_text_only_value = True
        if is_text_only is not None and is_text_only.lower() == 'false':
            is_text_only_value = False
        
        # Create guideline data
        guideline_data = {
            "title": title,
            "department": department,
            "subcategory": subcategory,
            "description": description,
            "content": content,
            "external_source_url": external_source_url,
            "is_text_only": is_text_only_value,
            "is_processed": False,  # Will be processed later for text-only
            "meta_data": json.dumps(metadata) if metadata else None
        }
        
        # Validate source file for text-only guidelines
        if is_text_only_value and not source_file and not external_source_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Source file or external source URL is required for text-only guidelines"
            )
        
        # Create guideline in database
        db_guideline = create_guideline(db, guideline_data)
        guideline_id = db_guideline.id
        
        # Process source file if provided
        if source_file:
            # Determine content type
            content_type = source_file.content_type
            if not content_type:
                content_type = mimetypes.guess_type(source_file.filename)[0] or "application/octet-stream"
            
            # Generate S3 key
            s3_key = f"guidelines/{guideline_id}/source/{source_file.filename}"
            
            # Upload file to S3
            s3_client = get_s3_client()
            # Read the file content
            content = await source_file.read()
            # Upload the bytes directly
            upload_result = s3_client.upload_file(
                file_obj=content, 
                object_key=s3_key, 
                content_type=content_type,
                metadata={"guideline_id": str(guideline_id), "type": "source"}
            )
            
            if not upload_result:
                logger.warning("Failed to upload source document to S3")
            else:
                # Generate presigned URL for the source document
                source_url = s3_client.get_presigned_url(s3_key, expiration=86400*30)  # 30 days
                
                # Update guideline with source information
                guideline_update = {
                    "source_url": source_url,
                    "source_s3_key": s3_key,
                    "source_content_type": content_type
                }
                
                db_guideline = update_guideline(db, guideline_id, guideline_update)
                
                # For text-only guidelines, schedule background processing
                if is_text_only_value and background_tasks:
                    # Import the processor
                    from app.utils.guideline_processor import process_text_only_guideline
                    background_tasks.add_task(process_text_only_guideline, db, guideline_id)
                    logger.info(f"Scheduled text extraction for guideline ID {guideline_id}")
        
        # Format response
        return format_guideline_response(db_guideline)
        
    except Exception as e:
        logger.error(f"Error creating guideline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create guideline: {str(e)}"
        )


@router.get("/", response_model=list[GuidelineResponse], tags=["guidelines"])
async def get_guidelines_endpoint(
    department: Optional[str] = None,
    subcategory: Optional[str] = None,
    title: Optional[str] = None,
    is_text_only: Optional[bool] = None,
    limit: int = Query(100, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Get marketing guidelines with optional filtering.
    
    Args:
        department: Filter by department
        subcategory: Filter by subcategory
        title: Filter by title (partial match)
        is_text_only: Filter by whether guideline is text-only
        limit: Maximum number of results
        skip: Number of results to skip
        db: Database session
    
    Returns:
        List of guidelines
    """
    logger.info(f"Getting guidelines (department={department}, subcategory={subcategory}, title={title}, is_text_only={is_text_only})")
    
    try:
        # Get guidelines from database
        guidelines = get_guidelines(
            db=db,
            department=department,
            subcategory=subcategory,
            title=title,
            is_text_only=is_text_only,
            limit=limit,
            skip=skip
        )
        
        # Format response
        return [format_guideline_response(g) for g in guidelines]
        
    except Exception as e:
        logger.error(f"Error getting guidelines: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve guidelines: {str(e)}"
        )


@router.get("/{guideline_id}", response_model=GuidelineResponse, tags=["guidelines"])
async def get_guideline_endpoint(
    guideline_id: int = Path(..., ge=1),
    db: Session = Depends(get_db)
):
    """
    Get a specific guideline by ID.
    
    Args:
        guideline_id: Guideline ID
        db: Database session
    
    Returns:
        Guideline details
    """
    logger.info(f"Getting guideline with ID: {guideline_id}")
    
    guideline = get_guideline_by_id(db, guideline_id)
    if not guideline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Guideline with ID {guideline_id} not found"
        )
    
    return format_guideline_response(guideline)


@router.get("/{guideline_id}/with-assets", response_model=GuidelineWithAssets, tags=["guidelines"])
async def get_guideline_with_assets_endpoint(
    guideline_id: int = Path(..., ge=1),
    db: Session = Depends(get_db)
):
    """
    Get a specific guideline with its associated assets.
    
    Args:
        guideline_id: Guideline ID
        db: Database session
    
    Returns:
        Guideline with assets
    """
    logger.info(f"Getting guideline with assets for ID: {guideline_id}")
    
    guideline = get_guideline_by_id(db, guideline_id)
    if not guideline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Guideline with ID {guideline_id} not found"
        )
    
    # Get associated assets
    assets = get_guideline_assets(db, guideline_id)
    
    # Get associated sections if it's a sectioned guideline
    sections = []
    if guideline.has_sections:
        db_sections = get_guideline_sections(db, guideline_id)
        
        for section in db_sections:
            section_data = format_section_response(section)
            
            # Get section assets if any
            if section.has_assets:
                section_assets = get_section_assets(db, section.id)
                section_data["assets"] = [format_section_asset_response(asset) for asset in section_assets]
            else:
                section_data["assets"] = []
                
            sections.append(section_data)
    
    # Format response
    response = format_guideline_response(guideline)
    response["assets"] = [format_asset_response(asset) for asset in assets]
    response["sections"] = sections
    
    return response


@router.get("/{guideline_id}/with-sections", response_model=Dict[str, Any], tags=["guidelines"])
async def get_guideline_with_sections_endpoint(
    guideline_id: int = Path(..., ge=1),
    db: Session = Depends(get_db)
):
    """
    Get a specific guideline with all its sections and assets.
    
    Args:
        guideline_id: Guideline ID
        db: Database session
    
    Returns:
        Guideline with sections and assets
    """
    logger.info(f"Getting guideline with sections for ID: {guideline_id}")
    
    # Use the database function to get the guideline with sections
    guideline_data = get_guideline_with_sections(db, guideline_id)
    
    if not guideline_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Guideline with ID {guideline_id} not found"
        )
    
    # Generate URLs for assets
    s3_client = get_s3_client()
    
    # Process main guideline assets
    for asset in guideline_data.get("assets", []):
        asset["url"] = s3_client.get_presigned_url(asset["s3_key"])
    
    # Process section assets
    for section in guideline_data.get("sections", []):
        for asset in section.get("assets", []):
            asset["url"] = s3_client.get_presigned_url(asset["s3_key"])
    
    # Always ensure source_url is present if a source document exists
    if guideline_data.get("has_source_document"):
        guideline = get_guideline_by_id(db, guideline_id)
        if guideline and guideline.source_s3_key:
            guideline_data["source_url"] = s3_client.get_presigned_url(guideline.source_s3_key, expiration=86400)  # 1 day
    
    return guideline_data


@router.put("/{guideline_id}", response_model=GuidelineResponse, tags=["guidelines"])
async def update_guideline_endpoint(
    guideline_id: int = Path(..., ge=1),
    guideline_update: GuidelineUpdate = None,
    db: Session = Depends(get_db)
):
    """
    Update a guideline.
    
    Args:
        guideline_id: Guideline ID
        guideline_update: Updated guideline data
        db: Database session
    
    Returns:
        Updated guideline
    """
    logger.info(f"Updating guideline with ID: {guideline_id}")
    
    try:
        if guideline_update is None:
            guideline_update = GuidelineUpdate()
            
        # Update guideline in database
        db_guideline = update_guideline(db, guideline_id, guideline_update.dict(exclude_unset=True))
        if not db_guideline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Guideline with ID {guideline_id} not found"
            )
        
        # Format response
        return format_guideline_response(db_guideline)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating guideline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update guideline: {str(e)}"
        )


@router.delete("/{guideline_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["guidelines"])
async def delete_guideline_endpoint(
    guideline_id: int = Path(..., ge=1),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Delete a guideline and its assets.
    
    Args:
        guideline_id: Guideline ID
        background_tasks: Background tasks manager
        db: Database session
    
    Returns:
        No content
    """
    logger.info(f"Deleting guideline with ID: {guideline_id}")
    
    # Get guideline and its assets first to check if it exists
    guideline = get_guideline_by_id(db, guideline_id)
    if not guideline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Guideline with ID {guideline_id} not found"
        )
    
    # Get associated assets to delete from S3
    assets = get_guideline_assets(db, guideline_id)
    
    # Get associated sections and their assets
    sections = []
    section_assets = []
    if guideline.has_sections:
        sections = get_guideline_sections(db, guideline_id)
        for section in sections:
            section_assets.extend(get_section_assets(db, section.id))
    
    # Delete guideline from database (will cascade delete sections and assets)
    result = delete_guideline(db, guideline_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete guideline with ID {guideline_id}"
        )
    
    # Delete asset files from S3 in the background
    if background_tasks:
        s3_client = get_s3_client()
        
        # Delete guideline assets
        for asset in assets:
            background_tasks.add_task(s3_client.delete_file, asset.s3_key)
        
        # Delete section assets
        for asset in section_assets:
            background_tasks.add_task(s3_client.delete_file, asset.s3_key)
        
        # Delete source document if exists
        if guideline.source_s3_key:
            background_tasks.add_task(s3_client.delete_file, guideline.source_s3_key)
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/search", response_model=list[GuidelineResponse], tags=["guidelines"])
async def search_guidelines_endpoint(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Search guidelines by text query.
    
    Args:
        query: Search query
        limit: Maximum number of results
        db: Database session
    
    Returns:
        List of matching guidelines
    """
    logger.info(f"Searching guidelines with query: '{query}'")
    
    try:
        # Search guidelines in database
        results = search_guidelines(db, query, limit)
        
        # Format response
        return [format_guideline_response(g) for g in results]
        
    except Exception as e:
        logger.error(f"Error searching guidelines: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search guidelines: {str(e)}"
        )


# Section-related endpoints
@router.post("/sections", response_model=SectionResponse, status_code=status.HTTP_201_CREATED, tags=["sections"])
async def create_section(
    section: SectionCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new section for a guideline.
    
    Args:
        section: Section data
        db: Database session
    
    Returns:
        Created section
    """
    logger.info(f"Creating new section for guideline ID {section.guideline_id}: {section.title}")
    
    try:
        # Check if guideline exists
        guideline = get_guideline_by_id(db, section.guideline_id)
        if not guideline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Guideline with ID {section.guideline_id} not found"
            )
        
        # Create section data
        section_data = {
            "guideline_id": section.guideline_id,
            "title": section.title,
            "content": section.content,
            "order": section.order,
            "metadata": section.metadata
        }
        
        # Create section in database
        db_section = create_guideline_section(db, section_data)
        
        # Update guideline is_text_only flag if needed
        if guideline.is_text_only:
            update_guideline(db, section.guideline_id, {"is_text_only": False})
        
        # Format response
        return format_section_response(db_section)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating section: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create section: {str(e)}"
        )


@router.get("/guideline/{guideline_id}/sections", response_model=list[SectionResponse], tags=["sections"])
async def get_sections(
    guideline_id: int = Path(..., ge=1),
    db: Session = Depends(get_db)
):
    """
    Get all sections for a specific guideline.
    
    Args:
        guideline_id: Guideline ID
        db: Database session
    
    Returns:
        List of sections
    """
    logger.info(f"Getting sections for guideline ID: {guideline_id}")
    
    # Check if guideline exists
    guideline = get_guideline_by_id(db, guideline_id)
    if not guideline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Guideline with ID {guideline_id} not found"
        )
    
    # Get sections
    sections = get_guideline_sections(db, guideline_id)
    
    # Format response
    return [format_section_response(section) for section in sections]


@router.get("/sections/{section_id}", response_model=SectionResponse, tags=["sections"])
async def get_section_endpoint(
    section_id: int = Path(..., ge=1),
    db: Session = Depends(get_db)
):
    """
    Get a specific section by ID.
    
    Args:
        section_id: Section ID
        db: Database session
    
    Returns:
        Section details
    """
    logger.info(f"Getting section with ID: {section_id}")
    
    section = get_section_by_id(db, section_id)
    if not section:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section with ID {section_id} not found"
        )
    
    return format_section_response(section)


@router.put("/sections/{section_id}", response_model=SectionResponse, tags=["sections"])
async def update_section_endpoint(
    section_id: int = Path(..., ge=1),
    section_update: SectionUpdate = None,
    db: Session = Depends(get_db)
):
    """
    Update a section.
    
    Args:
        section_id: Section ID
        section_update: Updated section data
        db: Database session
    
    Returns:
        Updated section
    """
    logger.info(f"Updating section with ID: {section_id}")
    
    try:
        if section_update is None:
            section_update = SectionUpdate()
            
        # Update section in database
        db_section = update_section(db, section_id, section_update.dict(exclude_unset=True))
        if not db_section:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section with ID {section_id} not found"
            )
        
        # Format response
        return format_section_response(db_section)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating section: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update section: {str(e)}"
        )


@router.delete("/sections/{section_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["sections"])
async def delete_section_endpoint(
    section_id: int = Path(..., ge=1),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Delete a section and its assets.
    
    Args:
        section_id: Section ID
        background_tasks: Background tasks manager
        db: Database session
    
    Returns:
        No content
    """
    logger.info(f"Deleting section with ID: {section_id}")
    
    # Get section and its assets first to check if it exists
    section = get_section_by_id(db, section_id)
    if not section:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section with ID {section_id} not found"
        )
    
    # Get associated assets to delete from S3
    section_assets = get_section_assets(db, section_id)
    
    # Delete section from database (will cascade delete assets)
    result = delete_section(db, section_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete section with ID {section_id}"
        )
    
    # Delete asset files from S3 in the background
    if background_tasks and section_assets:
        s3_client = get_s3_client()
        for asset in section_assets:
            background_tasks.add_task(s3_client.delete_file, asset.s3_key)
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Section Asset-related endpoints
@router.post("/section-assets", response_model=SectionAssetResponse, status_code=status.HTTP_201_CREATED, tags=["section-assets"])
async def upload_section_asset(
    file: UploadFile = File(...),
    section_id: int = Form(...),
    asset_type: str = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    order: int = Form(0),
    metadata_json: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload an asset for a section.
    
    Args:
        file: The file to upload
        section_id: ID of the parent section
        asset_type: Type of asset (e.g., 'image', 'pdf')
        title: Title of the asset
        description: Description of the asset
        order: Order of the asset within the section
        metadata_json: Additional metadata as JSON string
        db: Database session
    
    Returns:
        Uploaded asset details
    """
    logger.info(f"Uploading asset for section ID {section_id}: {file.filename}")
    
    try:
        # Check if section exists
        section = get_section_by_id(db, section_id)
        if not section:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section with ID {section_id} not found"
            )
        
        # Parse metadata if provided
        metadata = {}
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata JSON: {metadata_json}")
        
        # Determine content type
        content_type = file.content_type
        if not content_type:
            content_type = mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
        
        # Generate S3 key
        guideline_id = section.guideline_id
        s3_key = f"guidelines/{guideline_id}/sections/{section_id}/{asset_type}/{file.filename}"
        
        # Upload file to S3
        s3_client = get_s3_client()
        content = await file.read()
        upload_result = s3_client.upload_file(
            file_obj=content, 
            object_key=s3_key, 
            content_type=content_type,
            metadata={"section_id": str(section_id), "guideline_id": str(guideline_id)}
        )
        
        if not upload_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upload file to S3"
            )
        
        # Create asset in database
        asset_data = {
            "section_id": section_id,
            "asset_type": asset_type,
            "title": title or file.filename,
            "description": description,
            "s3_key": s3_key,
            "content_type": content_type,
            "order": order,
            "metadata": metadata
        }
        
        db_asset = create_section_asset(db, asset_data)
        
        # Format response
        return format_section_asset_response(db_asset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading section asset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload section asset: {str(e)}"
        )


@router.get("/section-assets/{asset_id}", response_model=SectionAssetResponse, tags=["section-assets"])
async def get_section_asset_endpoint(
    asset_id: int = Path(..., ge=1),
    db: Session = Depends(get_db)
):
    """
    Get a specific section asset by ID.
    
    Args:
        asset_id: Asset ID
        db: Database session
    
    Returns:
        Asset details
    """
    logger.info(f"Getting section asset with ID: {asset_id}")
    
    asset = get_section_asset_by_id(db, asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section asset with ID {asset_id} not found"
        )
    
    return format_section_asset_response(asset)


@router.get("/section/{section_id}/assets", response_model=list[SectionAssetResponse], tags=["section-assets"])
async def get_section_assets_endpoint(
    section_id: int = Path(..., ge=1),
    asset_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get all assets for a specific section.
    
    Args:
        section_id: Section ID
        asset_type: Filter by asset type
        db: Database session
    
    Returns:
        List of assets
    """
    logger.info(f"Getting assets for section ID: {section_id}")
    
    # Check if section exists
    section = get_section_by_id(db, section_id)
    if not section:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section with ID {section_id} not found"
        )
    
    # Get assets
    assets = get_section_assets(db, section_id, asset_type=asset_type)
    
    # Format response
    return [format_section_asset_response(asset) for asset in assets]


@router.put("/section-assets/{asset_id}", response_model=SectionAssetResponse, tags=["section-assets"])
async def update_section_asset_endpoint(
    asset_id: int = Path(..., ge=1),
    asset_update: SectionAssetUpdate = None,
    db: Session = Depends(get_db)
):
    """
    Update a section asset.
    
    Args:
        asset_id: Asset ID
        asset_update: Updated asset data
        db: Database session
    
    Returns:
        Updated asset
    """
    logger.info(f"Updating section asset with ID: {asset_id}")
    
    try:
        if asset_update is None:
            asset_update = SectionAssetUpdate()
            
        # Update asset in database
        db_asset = update_section_asset(db, asset_id, asset_update.dict(exclude_unset=True))
        if not db_asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section asset with ID {asset_id} not found"
            )
        
        # Format response
        return format_section_asset_response(db_asset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating section asset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update section asset: {str(e)}"
        )


@router.delete("/section-assets/{asset_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["section-assets"])
async def delete_section_asset_endpoint(
    asset_id: int = Path(..., ge=1),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Delete a section asset.
    
    Args:
        asset_id: Asset ID
        background_tasks: Background tasks manager
        db: Database session
    
    Returns:
        No content
    """
    logger.info(f"Deleting section asset with ID: {asset_id}")
    
    # Get asset first to check if it exists and to get S3 key
    asset = get_section_asset_by_id(db, asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section asset with ID {asset_id} not found"
        )
    
    # Store S3 key for later deletion
    s3_key = asset.s3_key
    
    # Delete asset from database
    result = delete_section_asset(db, asset_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete section asset with ID {asset_id}"
        )
    
    # Delete file from S3 in the background
    if background_tasks:
        s3_client = get_s3_client()
        background_tasks.add_task(s3_client.delete_file, s3_key)
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/section-assets/{asset_id}/download", tags=["section-assets"])
async def download_section_asset(
    asset_id: int = Path(..., ge=1),
    db: Session = Depends(get_db)
):
    """
    Download a section asset file.
    
    Args:
        asset_id: Asset ID
        db: Database session
    
    Returns:
        Redirect to asset file
    """
    logger.info(f"Downloading section asset with ID: {asset_id}")
    
    # Get asset
    asset = get_section_asset_by_id(db, asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section asset with ID {asset_id} not found"
        )
    
    # Get presigned URL
    s3_client = get_s3_client()
    url = s3_client.get_presigned_url(asset.s3_key)
    
    if not url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL"
        )
    
    # Return redirect to presigned URL
    return JSONResponse(content={"url": url})


# API Endpoints for Guideline Assets
@router.post("/assets", response_model=AssetResponse, status_code=status.HTTP_201_CREATED, tags=["assets"])
async def upload_asset(
    file: UploadFile = File(...),
    guideline_id: int = Form(...),
    asset_type: str = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    order: int = Form(0),
    metadata_json: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload an asset for a guideline.
    
    Args:
        file: The file to upload
        guideline_id: ID of the parent guideline
        asset_type: Type of asset (e.g., 'image', 'pdf')
        title: Title of the asset
        description: Description of the asset
        order: Order of the asset within the guideline
        metadata_json: Additional metadata as JSON string
        db: Database session
    
    Returns:
        Uploaded asset details
    """
    logger.info(f"Uploading asset for guideline ID {guideline_id}: {file.filename}")
    
    try:
        # Check if guideline exists
        guideline = get_guideline_by_id(db, guideline_id)
        if not guideline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Guideline with ID {guideline_id} not found"
            )
        
        # Parse metadata if provided
        metadata = {}
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata JSON: {metadata_json}")
        
        # Determine content type
        content_type = file.content_type
        if not content_type:
            content_type = mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
        
        # Generate S3 key
        s3_key = f"guidelines/{guideline_id}/{asset_type}/{file.filename}"
        
        # Upload file to S3
        s3_client = get_s3_client()
        content = await file.read()
        upload_result = s3_client.upload_file(
            file_obj=content, 
            object_key=s3_key, 
            content_type=content_type,
            metadata={"guideline_id": str(guideline_id)}
        )
        
        if not upload_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upload file to S3"
            )
        
        # Create asset in database
        asset_data = {
            "guideline_id": guideline_id,
            "asset_type": asset_type,
            "title": title or file.filename,
            "description": description,
            "s3_key": s3_key,
            "content_type": content_type,
            "order": order,
            "metadata": metadata
        }
        
        db_asset = create_guideline_asset(db, asset_data)
        
        # Format response
        return format_asset_response(db_asset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading asset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload asset: {str(e)}"
        )


@router.get("/assets/{asset_id}", response_model=AssetResponse, tags=["assets"])
async def get_asset_endpoint(
    asset_id: int = Path(..., ge=1),
    db: Session = Depends(get_db)
):
    """
    Get a specific asset by ID.
    
    Args:
        asset_id: Asset ID
        db: Database session
    
    Returns:
        Asset details
    """
    logger.info(f"Getting asset with ID: {asset_id}")
    
    asset = get_asset_by_id(db, asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset with ID {asset_id} not found"
        )
    
    return format_asset_response(asset)


@router.get("/guideline/{guideline_id}/assets", response_model=list[AssetResponse], tags=["assets"])
async def get_guideline_assets_endpoint(
    guideline_id: int = Path(..., ge=1),
    asset_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get all assets for a specific guideline.
    
    Args:
        guideline_id: Guideline ID
        asset_type: Filter by asset type
        db: Database session
    
    Returns:
        List of assets
    """
    logger.info(f"Getting assets for guideline ID: {guideline_id}")
    
    # Check if guideline exists
    guideline = get_guideline_by_id(db, guideline_id)
    if not guideline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Guideline with ID {guideline_id} not found"
        )
    
    # Get assets
    assets = get_guideline_assets(db, guideline_id, asset_type=asset_type)
    
    # Format response
    return [format_asset_response(asset) for asset in assets]


@router.put("/assets/{asset_id}", response_model=AssetResponse, tags=["assets"])
async def update_asset_endpoint(
    asset_id: int = Path(..., ge=1),
    asset_update: AssetUpdate = None,
    db: Session = Depends(get_db)
):
    """
    Update an asset.
    
    Args:
        asset_id: Asset ID
        asset_update: Updated asset data
        db: Database session
    
    Returns:
        Updated asset
    """
    logger.info(f"Updating asset with ID: {asset_id}")
    
    try:
        if asset_update is None:
            asset_update = AssetUpdate()
            
        # Update asset in database
        db_asset = update_asset(db, asset_id, asset_update.dict(exclude_unset=True))
        if not db_asset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Asset with ID {asset_id} not found"
            )
        
        # Format response
        return format_asset_response(db_asset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating asset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update asset: {str(e)}"
        )


@router.delete("/assets/{asset_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["assets"])
async def delete_asset_endpoint(
    asset_id: int = Path(..., ge=1),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Delete an asset.
    
    Args:
        asset_id: Asset ID
        background_tasks: Background tasks manager
        db: Database session
    
    Returns:
        No content
    """
    logger.info(f"Deleting asset with ID: {asset_id}")
    
    # Get asset first to check if it exists and to get S3 key
    asset = get_asset_by_id(db, asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset with ID {asset_id} not found"
        )
    
    # Store S3 key for later deletion
    s3_key = asset.s3_key
    
    # Delete asset from database
    result = delete_asset(db, asset_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete asset with ID {asset_id}"
        )
    
    # Delete file from S3 in the background
    if background_tasks:
        s3_client = get_s3_client()
        background_tasks.add_task(s3_client.delete_file, s3_key)
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/assets/{asset_id}/download", tags=["assets"])
async def download_asset(
    asset_id: int = Path(..., ge=1),
    db: Session = Depends(get_db)
):
    """
    Download an asset file.
    
    Args:
        asset_id: Asset ID
        db: Database session
    
    Returns:
        Redirect to asset file
    """
    logger.info(f"Downloading asset with ID: {asset_id}")
    
    # Get asset
    asset = get_asset_by_id(db, asset_id)
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset with ID {asset_id} not found"
        )
    
    # Get presigned URL
    s3_client = get_s3_client()
    url = s3_client.get_presigned_url(asset.s3_key)
    
    if not url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL"
        )
    
    # Return redirect to presigned URL
    return JSONResponse(content={"url": url})


@router.get("/{guideline_id}/source", tags=["guidelines"])
async def download_guideline_source(
    guideline_id: int = Path(..., ge=1),
    db: Session = Depends(get_db)
):
    """
    Download the source document for a guideline.
    
    Args:
        guideline_id: Guideline ID
        db: Database session
    
    Returns:
        Redirect to source document file
    """
    logger.info(f"Downloading source document for guideline ID: {guideline_id}")
    
    # Get guideline
    guideline = get_guideline_by_id(db, guideline_id)
    if not guideline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Guideline with ID {guideline_id} not found"
        )
    
    # Check if guideline has a source document
    if not guideline.source_s3_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Guideline with ID {guideline_id} has no source document"
        )
    
    # Get presigned URL
    s3_client = get_s3_client()
    url = s3_client.get_presigned_url(guideline.source_s3_key)
    
    if not url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL"
        )
    
    # Return redirect to presigned URL
    return JSONResponse(content={"url": url})