"""
Utility functions for processing guideline documents.
This module handles text extraction from source documents and indexing for text-only guidelines.
"""

import logging
import os
import tempfile
from typing import Optional, Tuple, List, Dict, Any

from app.config import get_settings
from app.utils.s3_client import get_s3_client
from app.db.guidelines_db import (
    get_guideline_by_id, 
    mark_guideline_processed,
    get_unprocessed_text_guidelines
)
from app.db.vector_store import ingest_documents

logger = logging.getLogger(__name__)


async def process_text_only_guideline(db, guideline_id: int) -> Tuple[bool, Optional[str]]:
    """
    Process a text-only guideline document by extracting its content 
    and ingesting it into the vector store.
    
    Args:
        db: Database session
        guideline_id: ID of the guideline to process
    
    Returns:
        Tuple of (success, error_message)
    """
    logger.info(f"Processing text-only guideline ID: {guideline_id}")
    
    # Get the guideline
    guideline = get_guideline_by_id(db, guideline_id)
    if not guideline:
        return False, f"Guideline with ID {guideline_id} not found"
    
    # Check if it's a text-only guideline
    if not guideline.is_text_only:
        return False, f"Guideline {guideline_id} is not a text-only guideline"
    
    # Check if the guideline has already been processed
    if guideline.is_processed:
        return True, None  # Already processed
    
    # Check if the guideline has a source document
    if not guideline.source_s3_key:
        return False, f"Guideline {guideline_id} has no source document"
    
    try:
        # Download the source document
        s3_client = get_s3_client()
        file_content = s3_client.download_file(guideline.source_s3_key)
        
        if not file_content:
            return False, f"Failed to download source document for guideline {guideline_id}"
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(guideline.source_s3_key)[1]) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            # Extract content from the document
            extracted_content = extract_text_from_document(temp_path, guideline.source_content_type)
            
            if not extracted_content:
                return False, f"Failed to extract content from document for guideline {guideline_id}"
            
            # Update the guideline with the extracted content
            success = mark_guideline_processed(db, guideline_id, extracted_content)
            
            if not success:
                return False, f"Failed to update guideline {guideline_id} with extracted content"
            
            # Create a document for ingestion
            processed_path = os.path.join(
                tempfile.gettempdir(), 
                f"guideline_{guideline_id}_processed.txt"
            )
            
            # Write the extracted content to a text file for ingestion
            with open(processed_path, 'w', encoding='utf-8') as f:
                # Add guideline metadata as header
                f.write(f"# {guideline.title}\n")
                f.write(f"Department: {guideline.department}\n")
                if guideline.subcategory:
                    f.write(f"Subcategory: {guideline.subcategory}\n")
                f.write("\n")
                
                # Write the actual content
                f.write(extracted_content)
            
            # Ingest the processed document into the vector store
            ingest_documents(
                document_paths=[processed_path],
                chunk_size=1000,
                chunk_overlap=200,
                qa_format=True,
                metadata={
                    "source": f"guideline_{guideline_id}",
                    "guideline_id": str(guideline_id),
                    "title": guideline.title,
                    "department": guideline.department,
                    "subcategory": guideline.subcategory
                }
            )
            
            # Clean up temporary files
            os.unlink(temp_path)
            os.unlink(processed_path)
            
            logger.info(f"Successfully processed guideline {guideline_id}")
            return True, None
            
        except Exception as e:
            logger.error(f"Error processing guideline {guideline_id}: {str(e)}")
            # Clean up temporary file
            os.unlink(temp_path)
            return False, str(e)
            
    except Exception as e:
        logger.error(f"Error processing guideline {guideline_id}: {str(e)}")
        return False, str(e)


def extract_text_from_document(file_path: str, content_type: Optional[str] = None) -> str:
    """
    Extract text content from a document file.
    
    Args:
        file_path: Path to the document file
        content_type: MIME type of the document
    
    Returns:
        Extracted text content
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Extract text based on file type
    if file_ext == '.txt':
        # Simple text file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    
    elif file_ext == '.md':
        # Markdown file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    
    elif file_ext == '.docx':
        # Word document
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception:
            # Fallback to docx2txt
            try:
                import docx2txt
                return docx2txt.process(file_path)
            except Exception as e:
                logger.error(f"Error extracting text from DOCX: {str(e)}")
                return ""
    
    elif file_ext == '.pdf':
        # PDF document
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            # Try another PDF library if available
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    return "\n\n".join([page.extract_text() or "" for page in pdf.pages])
            except Exception as e2:
                logger.error(f"Error extracting text from PDF with alternate method: {str(e2)}")
                return ""
    
    else:
        # Unsupported file type
        logger.warning(f"Unsupported file type for text extraction: {file_ext}")
        return ""


async def process_pending_guidelines(db) -> Dict[int, str]:
    """
    Process all pending text-only guidelines.
    
    Args:
        db: Database session
    
    Returns:
        Dictionary mapping guideline IDs to error messages (if any)
    """
    logger.info("Processing pending text-only guidelines")
    
    # Get unprocessed guidelines
    unprocessed = get_unprocessed_text_guidelines(db)
    
    if not unprocessed:
        logger.info("No unprocessed text-only guidelines found")
        return {}
    
    results = {}
    for guideline in unprocessed:
        success, error = await process_text_only_guideline(db, guideline.id)
        if not success and error:
            results[guideline.id] = error
    
    return results