"""
Enhanced retrieval functionality for guidelines integration.
This module implements the guidelines-specific retrieval functions.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.db.guidelines_db import (
    search_guidelines, 
    get_guideline_by_id, 
    get_guideline_assets,
    get_guideline_sections,
    get_section_assets,
    get_guideline_with_sections
)
from app.utils.s3_client import get_s3_client

logger = logging.getLogger(__name__)


def query_guidelines(
    db: Session, 
    query: str, 
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Query marketing guidelines based on the user's query.
    
    Args:
        db: Database session
        query: User's original query
        limit: Number of results to return
    
    Returns:
        List of relevant guidelines with their details
    """
    logger.info(f"Querying guidelines: '{query}'")
    
    try:
        # Search guidelines database
        guidelines = search_guidelines(db, query, limit=limit)
        
        # Format results
        results = []
        for guideline in guidelines:
            # Process guideline based on whether it's sectioned or not
            if guideline.is_sectioned and guideline.has_sections:
                # For sectioned guidelines, get all sections and combine them
                guideline_data = get_guideline_with_sections(db, guideline.id)
                if not guideline_data:
                    continue
                
                # Format the guideline with all its sections
                result = {
                    "content": format_sectioned_guideline_content(guideline_data),
                    "metadata": {
                        "id": guideline.id,
                        "title": guideline.title,
                        "department": guideline.department,
                        "subcategory": guideline.subcategory,
                        "source": guideline.source_url or f"/api/guidelines/{guideline.id}",
                        "type": "sectioned_guideline",
                        "has_sections": True
                    },
                    "relevance_score": calculate_relevance_score(query, guideline, sectioned=True),
                }
                
                results.append(result)
            else:
                # For regular guidelines, use the simpler formatting
                assets = []
                if guideline.has_assets:
                    db_assets = get_guideline_assets(db, guideline.id)
                    
                    s3_client = get_s3_client()
                    
                    for asset in db_assets:
                        asset_url = s3_client.get_presigned_url(asset.s3_key)
                        
                        # Parse metadata if available
                        metadata = {}
                        if asset.meta_data:
                            import json
                            try:
                                metadata = json.loads(asset.meta_data)
                            except json.JSONDecodeError:
                                pass
                        
                        assets.append({
                            "id": asset.id,
                            "asset_type": asset.asset_type,
                            "title": asset.title,
                            "description": asset.description,
                            "url": asset_url,
                            "content_type": asset.content_type,
                            "metadata": metadata
                        })
                
                # Format guideline result
                result = {
                    "content": format_guideline_content(guideline, assets),
                    "metadata": {
                        "id": guideline.id,
                        "title": guideline.title,
                        "department": guideline.department,
                        "subcategory": guideline.subcategory,
                        "source": guideline.source_url or f"/api/guidelines/{guideline.id}",
                        "type": "guideline",
                        "has_assets": guideline.has_assets
                    },
                    "relevance_score": calculate_relevance_score(query, guideline),
                    "assets": assets
                }
                
                results.append(result)
        
        logger.info(f"Retrieved {len(results)} guidelines")
        return results
        
    except Exception as e:
        logger.error(f"Error querying guidelines: {str(e)}")
        return []


def format_guideline_content(guideline, assets: List[Dict[str, Any]] = None) -> str:
    """
    Format a guideline and its assets into text content for retrieval.
    
    Args:
        guideline: Guideline database object
        assets: List of formatted assets
    
    Returns:
        Formatted content string
    """
    content_parts = []
    
    # Add guideline title and department info
    content_parts.append(f"# {guideline.title}")
    content_parts.append(f"Department: {guideline.department}")
    
    if guideline.subcategory:
        content_parts.append(f"Subcategory: {guideline.subcategory}")
    
    # Add description if available
    if guideline.description:
        content_parts.append(f"\n## Description\n{guideline.description}")
    
    # Add main content if available
    if guideline.content:
        content_parts.append(f"\n## Content\n{guideline.content}")
    
    # Add asset information if available
    if assets:
        content_parts.append("\n## Assets")
        
        # Group assets by type
        asset_types = {}
        for asset in assets:
            asset_type = asset.get("asset_type", "other")
            if asset_type not in asset_types:
                asset_types[asset_type] = []
            asset_types[asset_type].append(asset)
        
        # Format each asset type
        for asset_type, type_assets in asset_types.items():
            content_parts.append(f"\n### {asset_type.title()} Assets")
            
            for i, asset in enumerate(type_assets, 1):
                asset_title = asset.get("title", f"Asset {i}")
                asset_desc = asset.get("description", "")
                asset_url = asset.get("url", "")
                
                content_parts.append(f"- {asset_title}: {asset_desc}")
                if asset_url:
                    content_parts.append(f"  URL: {asset_url}")
    
    # Add source information
    if guideline.source_url:
        content_type_info = ""
        if guideline.source_content_type:
            content_type_info = f" ({guideline.source_content_type})"
        
        content_parts.append(f"\n## Source\nReference: {guideline.source_url}{content_type_info}")
        content_parts.append("Click the link above to access the full source document.")
    
    return "\n\n".join(content_parts)


def format_sectioned_guideline_content(guideline_data: Dict[str, Any]) -> str:
    """
    Format a sectioned guideline and all its sections into text content for retrieval.
    
    Args:
        guideline_data: Guideline data with sections from get_guideline_with_sections
    
    Returns:
        Formatted content string
    """
    content_parts = []
    
    # Add guideline title and department info
    content_parts.append(f"# {guideline_data['title']}")
    content_parts.append(f"Department: {guideline_data['department']}")
    
    if guideline_data.get("subcategory"):
        content_parts.append(f"Subcategory: {guideline_data['subcategory']}")
    
    # Add description if available
    if guideline_data.get("description"):
        content_parts.append(f"\n## Description\n{guideline_data['description']}")
    
    # Add main content if available
    if guideline_data.get("content"):
        content_parts.append(f"\n## General Content\n{guideline_data['content']}")
    
    # Add sections
    sections = guideline_data.get("sections", [])
    if sections:
        content_parts.append("\n## Sections")
        
        # Sort sections by order if available
        sorted_sections = sorted(sections, key=lambda s: s.get("order", 999))
        
        for section in sorted_sections:
            content_parts.append(f"\n### {section['title']}")
            
            if section.get("content"):
                content_parts.append(section["content"])
            
            # Add section assets if available
            section_assets = section.get("assets", [])
            if section_assets:
                content_parts.append("\n#### Assets")
                
                # Group assets by type
                asset_types = {}
                for asset in section_assets:
                    asset_type = asset.get("asset_type", "other")
                    if asset_type not in asset_types:
                        asset_types[asset_type] = []
                    asset_types[asset_type].append(asset)
                
                # Format each asset type
                for asset_type, type_assets in asset_types.items():
                    content_parts.append(f"\n##### {asset_type.title()} Assets")
                    
                    for i, asset in enumerate(type_assets, 1):
                        asset_title = asset.get("title", f"Asset {i}")
                        asset_desc = asset.get("description", "")
                        asset_url = asset.get("url", "")
                        
                        content_parts.append(f"- {asset_title}: {asset_desc}")
                        if asset_url:
                            content_parts.append(f"  URL: {asset_url}")
    
    # Add guideline assets at the end if any
    guideline_assets = guideline_data.get("assets", [])
    if guideline_assets:
        content_parts.append("\n## General Assets")
        
        # Group assets by type
        asset_types = {}
        for asset in guideline_assets:
            asset_type = asset.get("asset_type", "other")
            if asset_type not in asset_types:
                asset_types[asset_type] = []
            asset_types[asset_type].append(asset)
        
        # Format each asset type
        for asset_type, type_assets in asset_types.items():
            content_parts.append(f"\n### {asset_type.title()} Assets")
            
            for i, asset in enumerate(type_assets, 1):
                asset_title = asset.get("title", f"Asset {i}")
                asset_desc = asset.get("description", "")
                asset_url = asset.get("url", "")
                
                content_parts.append(f"- {asset_title}: {asset_desc}")
                if asset_url:
                    content_parts.append(f"  URL: {asset_url}")
    
    # Add source information
    if guideline_data.get("source_url"):
        content_type_info = ""
        if guideline_data.get("source_content_type"):
            content_type_info = f" ({guideline_data['source_content_type']})"
        
        content_parts.append(f"\n## Source\nReference: {guideline_data['source_url']}{content_type_info}")
        content_parts.append("Click the link above to access the full source document.")
    
    return "\n\n".join(content_parts)


def calculate_relevance_score(query: str, guideline, sectioned: bool = False) -> float:
    """
    Calculate a simple relevance score for a guideline against a query.
    This is a basic implementation that could be enhanced with NLP.
    
    Args:
        query: User's query
        guideline: Guideline object
        sectioned: Whether this is a sectioned guideline
    
    Returns:
        Relevance score (0-1)
    """
    query = query.lower()
    
    # Count query term occurrences in different fields
    count = 0
    
    # Check title (highest weight)
    if guideline.title:
        title_count = sum(query.count(term) for term in guideline.title.lower().split())
        count += title_count * 3
    
    # Check department and subcategory (high weight)
    if guideline.department:
        dept_count = sum(query.count(term) for term in guideline.department.lower().split())
        count += dept_count * 2
    
    if guideline.subcategory:
        subcat_count = sum(query.count(term) for term in guideline.subcategory.lower().split())
        count += subcat_count * 2
    
    # Check description and content (normal weight)
    if guideline.description:
        desc_count = sum(query.count(term) for term in guideline.description.lower().split())
        count += desc_count
    
    if guideline.content:
        content_count = sum(query.count(term) for term in guideline.content.lower().split())
        count += content_count
    
    # Boost score for sectioned guidelines slightly as they're more comprehensive
    if sectioned:
        count *= 1.1
    
    # Normalize to 0-1 range (simple approach)
    # This is a very basic calculation - in production, use more sophisticated methods
    score = min(count / 10, 1.0)
    
    return score