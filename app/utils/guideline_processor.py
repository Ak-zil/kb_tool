"""
Utility functions for processing guideline documents.
This module handles text extraction from source documents and indexing for text-only guidelines.
"""

import logging
import os
import tempfile
from typing import Optional, Tuple, Dict

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
    
    temp_files = []  # Track all temporary files
    try:
        # Download the source document
        s3_client = get_s3_client()
        file_content = s3_client.download_file(guideline.source_s3_key)
        
        if not file_content:
            return False, f"Failed to download source document for guideline {guideline_id}"
        
        # Get file extension from the S3 key
        file_ext = os.path.splitext(guideline.source_s3_key)[1]
        if not file_ext:
            file_ext = get_file_extension_from_content_type(guideline.source_content_type)
        
        # Make the file extension lowercase
        file_ext = file_ext.lower()
        
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"guideline_{guideline_id}_source{file_ext}")
        
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(file_content)
        
        temp_files.append(temp_path)
        logger.info(f"Saved source document to temporary file: {temp_path}")
        
        # Extract content from the document
        extracted_content = extract_text_from_document(temp_path, guideline.source_content_type)
        
        if not extracted_content:
            return False, f"Failed to extract content from document for guideline {guideline_id}"
        
        # Update the guideline with the extracted content
        success = mark_guideline_processed(db, guideline_id, extracted_content)
        
        if not success:
            return False, f"Failed to update guideline {guideline_id} with extracted content"
        
        # Create a processed text file for ingestion
        processed_path = os.path.join(temp_dir, f"guideline_{guideline_id}_processed.txt")
        
        # Write the extracted content to a text file for ingestion with rich error handling
        try:
            with open(processed_path, 'w', encoding='utf-8') as f:
                # Add guideline metadata as header
                f.write(f"# {guideline.title}\n")
                f.write(f"Department: {guideline.department}\n")
                if guideline.subcategory:
                    f.write(f"Subcategory: {guideline.subcategory}\n")
                f.write("\n")
                
                # Write the actual content
                f.write(extracted_content)
            
            temp_files.append(processed_path)
            logger.info(f"Created processed file for ingestion: {processed_path}")
            
            # Verify that the file is readable
            with open(processed_path, 'r', encoding='utf-8') as f:
                sample = f.read(1024)  # Read a sample to verify
                if not sample:
                    logger.warning(f"Warning: Generated file is empty or unreadable: {processed_path}")
        
        except Exception as file_error:
            logger.error(f"Error creating processed file: {str(file_error)}")
            return False, f"Failed to create processed file: {str(file_error)}"
        
        # Ingest the processed document into the vector store
        try:
            doc_count = ingest_documents(
                document_paths=[processed_path],
                chunk_size=1000,
                chunk_overlap=200,
                qa_format=True
            )
            
            logger.info(f"Successfully ingested {doc_count} chunks from guideline {guideline_id}")
        except Exception as ingest_error:
            logger.error(f"Error ingesting document: {str(ingest_error)}")
            return False, f"Failed to ingest document: {str(ingest_error)}"
        
        logger.info(f"Successfully processed guideline {guideline_id}")
        return True, None
        
    except Exception as e:
        logger.error(f"Error processing guideline {guideline_id}: {str(e)}")
        return False, str(e)
    finally:
        # Clean up all temporary files
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.info(f"Removed temporary file: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to remove temporary file {file_path}: {str(cleanup_error)}")


def extract_text_from_document(file_path: str, content_type: Optional[str] = None) -> str:
    """
    Extract text content from a document file.
    
    Args:
        file_path: Path to the document file
        content_type: MIME type of the document
    
    Returns:
        Extracted text content
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return ""
    
    file_ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"Extracting text from {file_path} with extension {file_ext}")
    
    try:
        # Extract text based on file type
        if file_ext == '.txt':
            # Simple text file - handle various encodings
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
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
                return "\n\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
            except Exception as docx_error:
                logger.warning(f"Error extracting with python-docx: {str(docx_error)}")
                # Fallback to docx2txt
                try:
                    import docx2txt
                    return docx2txt.process(file_path)
                except Exception as docx2txt_error:
                    logger.error(f"Error extracting with docx2txt: {str(docx2txt_error)}")
                    return ""
        
        elif file_ext == '.pdf':
            # PDF document
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
                
                # Check if we got meaningful content
                if text.strip():
                    return text
                
                logger.warning("PDF text extraction yielded little content, trying alternative method")
                
                # Try another PDF library if available
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        alt_text = "\n\n".join([page.extract_text() or "" for page in pdf.pages])
                        return alt_text if alt_text.strip() else text
                except ImportError:
                    return text  # Return original text if pdfplumber is not available
                except Exception as pdfplumber_error:
                    logger.error(f"Error with pdfplumber: {str(pdfplumber_error)}")
                    return text  # Return original text if pdfplumber fails
                
            except Exception as pdf_error:
                logger.error(f"Error extracting text from PDF: {str(pdf_error)}")
                return ""
        
        else:
            # Try to handle as text for unsupported file types
            logger.warning(f"Unsupported file type {file_ext} for text extraction, trying as text")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            except Exception:
                # Last resort: try binary read and decode
                with open(file_path, 'rb') as f:
                    content = f.read()
                    return content.decode('utf-8', errors='replace')
    
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""


def get_file_extension_from_content_type(content_type: Optional[str]) -> str:
    """
    Get a file extension based on MIME type.
    
    Args:
        content_type: MIME type string
    
    Returns:
        File extension with leading dot
    """
    if not content_type:
        return ".txt"  # Default to text
    
    # Map common MIME types to extensions
    content_type = content_type.lower()
    mime_map = {
        "text/plain": ".txt",
        "text/markdown": ".md",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/msword": ".doc",
        "application/pdf": ".pdf",
        "text/html": ".html",
        "application/json": ".json"
    }
    
    return mime_map.get(content_type, ".txt")


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