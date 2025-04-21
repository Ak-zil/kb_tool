# """
# Admin API endpoints for managing the Marketing Knowledge Base.
# """

# import logging
# import os
# from typing import List, Dict, Any, Optional

# from fastapi import (
#     APIRouter, 
#     Depends, 
#     HTTPException, 
#     status, 
#     UploadFile, 
#     File, 
#     Form,
#     BackgroundTasks
# )
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field
# from sqlalchemy.orm import Session

# from app.config import get_settings
# from app.db.metrics_db import get_db, create_chat_summary
# from app.db.vector_store import (
#     ingest_documents, 
#     ingest_directory, 
#     query_documents, 
#     clear_vector_store
# )

# logger = logging.getLogger(__name__)

# router = APIRouter()


# # Pydantic models for request/response
# class DocumentUpload(BaseModel):
#     """Model for document upload response."""
#     filename: str = Field(..., description="Name of the uploaded file")
#     status: str = Field(..., description="Status of the upload")
#     file_size: int = Field(..., description="Size of the file in bytes")
#     document_id: Optional[str] = Field(None, description="ID of the document in the knowledge base")


# class IngestResponse(BaseModel):
#     """Model for document ingestion response."""
#     success: bool = Field(..., description="Whether the ingestion was successful")
#     message: str = Field(..., description="Status message")
#     documents_count: int = Field(..., description="Number of documents ingested")


# class ChatSummaryCreate(BaseModel):
#     """Model for creating a chat summary."""
#     title: str = Field(..., description="Title of the chat summary")
#     summary: str = Field(..., description="Text summary of the chat")
#     topics: Optional[List[str]] = Field(None, description="List of topics covered in the chat")
#     metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# class ChatSummaryResponse(BaseModel):
#     """Model for chat summary response."""
#     id: int = Field(..., description="ID of the chat summary")
#     title: str = Field(..., description="Title of the chat summary")
#     summary: str = Field(..., description="Text summary of the chat")
#     topics: Optional[List[str]] = Field(None, description="List of topics covered in the chat")
#     timestamp: str = Field(..., description="Timestamp when the summary was created")
#     metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# @router.post("/upload", response_model=DocumentUpload, tags=["admin", "knowledge-base"])
# async def upload_document(
#     file: UploadFile = File(...),
#     background_tasks: BackgroundTasks = None,
#     db: Session = Depends(get_db)
# ):
#     """
#     Upload a document to the knowledge base.
    
#     Args:
#         file: The file to upload
#         background_tasks: Background tasks manager
#         db: Database session
    
#     Returns:
#         Upload status information
#     """
#     logger.info(f"Uploading document: {file.filename}")
    
#     try:
#         # Get settings
#         settings = get_settings()
        
#         # Create knowledge base directory if it doesn't exist
#         os.makedirs(settings.knowledge_base_dir, exist_ok=True)
        
#         # Generate a file path
#         file_path = os.path.join(settings.knowledge_base_dir, file.filename)
        
#         # Check if file already exists
#         if os.path.exists(file_path):
#             # Add timestamp to filename to make it unique
#             import time
#             timestamp = int(time.time())
#             filename_parts = os.path.splitext(file.filename)
#             new_filename = f"{filename_parts[0]}_{timestamp}{filename_parts[1]}"
#             file_path = os.path.join(settings.knowledge_base_dir, new_filename)
        
#         # Save the file
#         with open(file_path, "wb") as f:
#             content = await file.read()
#             f.write(content)
#             file_size = len(content)
        
#         # Schedule ingestion in the background if background_tasks is provided
#         if background_tasks:
#             background_tasks.add_task(
#                 ingest_documents,
#                 document_paths=[file_path]
#             )
#             ingest_status = "scheduled"
#         else:
#             # Otherwise, ingest immediately
#             try:
#                 ingest_documents(document_paths=[file_path])
#                 ingest_status = "completed"
#             except Exception as e:
#                 logger.error(f"Error ingesting document: {str(e)}")
#                 ingest_status = "failed"
        
#         logger.info(f"Document uploaded successfully: {file_path}")
#         return {
#             "filename": os.path.basename(file_path),
#             "status": ingest_status,
#             "file_size": file_size,
#             "document_id": os.path.basename(file_path)
#         }
        
#     except Exception as e:
#         logger.error(f"Error uploading document: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to upload document: {str(e)}"
#         )


# @router.post("/ingest", response_model=IngestResponse, tags=["admin", "knowledge-base"])
# async def ingest_documents_endpoint(
#     files: List[str] = Form(...),
#     chunk_size: int = Form(1000),
#     chunk_overlap: int = Form(200),
#     db: Session = Depends(get_db)
# ):
#     """
#     Ingest documents into the knowledge base.
    
#     Args:
#         files: List of file paths relative to the knowledge base directory
#         chunk_size: Size of document chunks
#         chunk_overlap: Overlap between chunks
#         db: Database session
    
#     Returns:
#         Ingestion status
#     """
#     logger.info(f"Ingesting documents: {files}")
    
#     try:
#         # Get settings
#         settings = get_settings()
        
#         # Convert relative paths to absolute paths
#         absolute_paths = []
#         for file in files:
#             if os.path.isabs(file):
#                 absolute_paths.append(file)
#             else:
#                 absolute_paths.append(os.path.join(settings.knowledge_base_dir, file))
        
#         # Validate that the files exist
#         for path in absolute_paths:
#             if not os.path.exists(path) or not os.path.isfile(path):
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail=f"File does not exist: {path}"
#                 )
        
#         # Ingest the documents
#         doc_count = ingest_documents(
#             document_paths=absolute_paths,
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap
#         )
        
#         logger.info(f"Documents ingested successfully: {doc_count} chunks")
#         return {
#             "success": True,
#             "message": f"Successfully ingested {len(absolute_paths)} documents ({doc_count} chunks)",
#             "documents_count": doc_count
#         }
        
#     except Exception as e:
#         logger.error(f"Error ingesting documents: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to ingest documents: {str(e)}"
#         )


# @router.post("/ingest-directory", response_model=IngestResponse, tags=["admin", "knowledge-base"])
# async def ingest_directory_endpoint(
#     directory: str = Form(...),
#     glob_pattern: str = Form("**/*.txt"),
#     chunk_size: int = Form(1000),
#     chunk_overlap: int = Form(200),
#     db: Session = Depends(get_db)
# ):
#     """
#     Ingest all documents in a directory.
    
#     Args:
#         directory: Directory path relative to the knowledge base directory
#         glob_pattern: Pattern for matching files
#         chunk_size: Size of document chunks
#         chunk_overlap: Overlap between chunks
#         db: Database session
    
#     Returns:
#         Ingestion status
#     """
#     logger.info(f"Ingesting directory: {directory} with pattern {glob_pattern}")
    
#     try:
#         # Get settings
#         settings = get_settings()
        
#         # Convert relative path to absolute path
#         if os.path.isabs(directory):
#             directory_path = directory
#         else:
#             directory_path = os.path.join(settings.knowledge_base_dir, directory)
        
#         # Validate that the directory exists
#         if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Directory does not exist: {directory_path}"
#             )
        
#         # Ingest the directory
#         doc_count = ingest_directory(
#             directory_path=directory_path,
#             glob_pattern=glob_pattern,
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap
#         )
        
#         logger.info(f"Directory ingested successfully: {doc_count} chunks")
#         return {
#             "success": True,
#             "message": f"Successfully ingested directory {directory} ({doc_count} chunks)",
#             "documents_count": doc_count
#         }
        
#     except Exception as e:
#         logger.error(f"Error ingesting directory: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to ingest directory: {str(e)}"
#         )


# @router.post("/clear-knowledge-base", tags=["admin", "knowledge-base"])
# async def clear_knowledge_base_endpoint(db: Session = Depends(get_db)):
#     """
#     Clear the entire knowledge base.
#     Use with caution!
    
#     Args:
#         db: Database session
    
#     Returns:
#         Status message
#     """
#     logger.warning("Clearing knowledge base")
    
#     try:
#         # Clear the vector store
#         clear_vector_store()
        
#         logger.info("Knowledge base cleared successfully")
#         return {"message": "Knowledge base cleared successfully"}
        
#     except Exception as e:
#         logger.error(f"Error clearing knowledge base: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to clear knowledge base: {str(e)}"
#         )


# @router.post("/chat-summary", response_model=ChatSummaryResponse, tags=["admin", "chat-summaries"])
# async def create_chat_summary_endpoint(summary: ChatSummaryCreate, db: Session = Depends(get_db)):
#     """
#     Create a new chat summary.
    
#     Args:
#         summary: Chat summary data
#         db: Database session
    
#     Returns:
#         Created chat summary
#     """
#     logger.info(f"Creating chat summary: {summary.title}")
    
#     try:
#         # Create the chat summary
#         db_summary = create_chat_summary(db, summary.dict())
        
#         # Format response
#         topics = []
#         if db_summary.topics:
#             import json
#             try:
#                 topics = json.loads(db_summary.topics)
#             except json.JSONDecodeError:
#                 topics = []
        
#         metadata = {}
#         if db_summary.metadata:
#             try:
#                 metadata = json.loads(db_summary.metadata)
#             except json.JSONDecodeError:
#                 metadata = {}
        
#         logger.info(f"Chat summary created successfully: ID {db_summary.id}")
#         return {
#             "id": db_summary.id,
#             "title": db_summary.title,
#             "summary": db_summary.summary,
#             "topics": topics,
#             "timestamp": db_summary.timestamp.isoformat(),
#             "metadata": metadata
#         }
        
#     except Exception as e:
#         logger.error(f"Error creating chat summary: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to create chat summary: {str(e)}"
#         )


# @router.get("/test-query", tags=["admin", "knowledge-base"])
# async def test_query_endpoint(
#     query: str,
#     top_k: int = 5,
#     db: Session = Depends(get_db)
# ):
#     """
#     Test a query against the knowledge base.
    
#     Args:
#         query: Query string
#         top_k: Number of results to return
#         db: Database session
    
#     Returns:
#         Query results
#     """
#     logger.info(f"Testing query: '{query}'")
    
#     try:
#         # Query the knowledge base
#         results = query_documents(query, top_k=top_k)
        
#         logger.info(f"Query returned {len(results)} results")
#         return {
#             "query": query,
#             "results_count": len(results),
#             "results": results
#         }
        
#     except Exception as e:
#         logger.error(f"Error testing query: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to test query: {str(e)}"
#         )


"""
Admin API endpoints for managing the Marketing Knowledge Base.
"""

import logging
import os
import mimetypes
from typing import List, Dict, Any, Optional

from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    status, 
    UploadFile, 
    File, 
    Form,
    BackgroundTasks
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.metrics_db import get_db, create_chat_summary
from app.db.vector_store import (
    ingest_documents, 
    ingest_directory, 
    query_documents, 
    clear_vector_store
)
from app.utils.text_processing import preprocess_document_text

logger = logging.getLogger(__name__)

router = APIRouter()

# Ensure mimetypes are initialized
mimetypes.init()
# Add MIME types for common document formats if not already present
if not mimetypes.guess_extension('application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
    mimetypes.add_type('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx')

# Pydantic models for request/response
class DocumentUpload(BaseModel):
    """Model for document upload response."""
    filename: str = Field(..., description="Name of the uploaded file")
    status: str = Field(..., description="Status of the upload")
    file_size: int = Field(..., description="Size of the file in bytes")
    document_id: Optional[str] = Field(None, description="ID of the document in the knowledge base")
    file_type: str = Field(..., description="Type of the file")


class IngestResponse(BaseModel):
    """Model for document ingestion response."""
    success: bool = Field(..., description="Whether the ingestion was successful")
    message: str = Field(..., description="Status message")
    documents_count: int = Field(..., description="Number of documents ingested")


class ChunkingOptions(BaseModel):
    """Model for chunking options."""
    chunk_size: int = Field(1000, description="Size of document chunks")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    qa_format: bool = Field(False, description="Whether to detect and optimize Q&A format")


class ChatSummaryCreate(BaseModel):
    """Model for creating a chat summary."""
    title: str = Field(..., description="Title of the chat summary")
    summary: str = Field(..., description="Text summary of the chat")
    topics: Optional[List[str]] = Field(None, description="List of topics covered in the chat")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChatSummaryResponse(BaseModel):
    """Model for chat summary response."""
    id: int = Field(..., description="ID of the chat summary")
    title: str = Field(..., description="Title of the chat summary")
    summary: str = Field(..., description="Text summary of the chat")
    topics: Optional[List[str]] = Field(None, description="List of topics covered in the chat")
    timestamp: str = Field(..., description="Timestamp when the summary was created")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# Helper functions
def is_supported_file_type(filename: str) -> bool:
    """
    Check if the file type is supported for ingestion.
    
    Args:
        filename: Filename to check
    
    Returns:
        True if supported, False otherwise
    """
    supported_extensions = ['.txt', '.docx', '.doc', '.md', '.pdf']
    _, ext = os.path.splitext(filename.lower())
    return ext in supported_extensions


@router.post("/upload", response_model=DocumentUpload, tags=["admin", "knowledge-base"])
async def upload_document(
    file: UploadFile = File(...),
    qa_format: bool = Form(False),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Upload a document to the knowledge base.
    
    Args:
        file: The file to upload
        qa_format: Whether to detect and optimize Q&A format
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        background_tasks: Background tasks manager
        db: Database session
    
    Returns:
        Upload status information
    """
    logger.info(f"Uploading document: {file.filename}")
    
    try:
        # Check if file type is supported
        if not is_supported_file_type(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file.filename}. Supported types: .txt, .docx, .doc, .md, .pdf"
            )
        
        # Get settings
        settings = get_settings()
        
        # Create knowledge base directory if it doesn't exist
        os.makedirs(settings.knowledge_base_dir, exist_ok=True)
        
        # Determine file type and extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
        
        # Generate a file path
        file_path = os.path.join(settings.knowledge_base_dir, file.filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            # Add timestamp to filename to make it unique
            import time
            timestamp = int(time.time())
            filename_parts = os.path.splitext(file.filename)
            new_filename = f"{filename_parts[0]}_{timestamp}{filename_parts[1]}"
            file_path = os.path.join(settings.knowledge_base_dir, new_filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            file_size = len(content)
        
        # Schedule ingestion in the background if background_tasks is provided
        if background_tasks:
            background_tasks.add_task(
                ingest_documents,
                document_paths=[file_path],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            ingest_status = "scheduled"
        else:
            # Otherwise, ingest immediately
            try:
                ingest_documents(
                    document_paths=[file_path],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                ingest_status = "completed"
            except Exception as e:
                logger.error(f"Error ingesting document: {str(e)}")
                ingest_status = "failed"
        
        logger.info(f"Document uploaded successfully: {file_path}")
        return {
            "filename": os.path.basename(file_path),
            "status": ingest_status,
            "file_size": file_size,
            "document_id": os.path.basename(file_path),
            "file_type": content_type
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )
    

@router.post("/ingest", response_model=IngestResponse, tags=["admin", "knowledge-base"])
async def ingest_documents_endpoint(
    files: List[str] = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    qa_format: bool = Form(False),
    db: Session = Depends(get_db)
):
    """
    Ingest documents into the knowledge base.
    
    Args:
        files: List of file paths relative to the knowledge base directory
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        qa_format: Whether to detect and optimize Q&A format
        db: Database session
    
    Returns:
        Ingestion status
    """
    logger.info(f"Ingesting documents: {files}")
    
    try:
        # Get settings
        settings = get_settings()
        
        # Convert relative paths to absolute paths
        absolute_paths = []
        for file in files:
            if os.path.isabs(file):
                absolute_paths.append(file)
            else:
                absolute_paths.append(os.path.join(settings.knowledge_base_dir, file))
        
        # Validate that the files exist
        for path in absolute_paths:
            if not os.path.exists(path) or not os.path.isfile(path):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File does not exist: {path}"
                )
        
        # Ingest the documents
        doc_count = ingest_documents(
            document_paths=absolute_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            qa_format=qa_format
        )
        
        logger.info(f"Documents ingested successfully: {doc_count} chunks")
        return {
            "success": True,
            "message": f"Successfully ingested {len(absolute_paths)} documents ({doc_count} chunks)",
            "documents_count": doc_count
        }
        
    except Exception as e:
        logger.error(f"Error ingesting documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest documents: {str(e)}"
        )

@router.post("/ingest-directory", response_model=IngestResponse, tags=["admin", "knowledge-base"])
async def ingest_directory_endpoint(
    directory: str = Form(...),
    glob_pattern: str = Form("**/*.*"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    qa_format: bool = Form(False),
    db: Session = Depends(get_db)
):
    """
    Ingest all documents in a directory.
    
    Args:
        directory: Directory path relative to the knowledge base directory
        glob_pattern: Pattern for matching files
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        qa_format: Whether to detect and optimize Q&A format
        db: Database session
    
    Returns:
        Ingestion status
    """
    logger.info(f"Ingesting directory: {directory} with pattern {glob_pattern}")
    
    try:
        # Get settings
        settings = get_settings()
        
        # Convert relative path to absolute path
        if os.path.isabs(directory):
            directory_path = directory
        else:
            directory_path = os.path.join(settings.knowledge_base_dir, directory)
        
        # Validate that the directory exists
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Directory does not exist: {directory_path}"
            )
        
        # Ingest the directory
        doc_count = ingest_directory(
            directory_path=directory_path,
            glob_pattern=glob_pattern,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            qa_format=qa_format
        )
        
        logger.info(f"Directory ingested successfully: {doc_count} chunks")
        return {
            "success": True,
            "message": f"Successfully ingested directory {directory} ({doc_count} chunks)",
            "documents_count": doc_count
        }
        
    except Exception as e:
        logger.error(f"Error ingesting directory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest directory: {str(e)}"
        )
    





@router.post("/clear-knowledge-base", tags=["admin", "knowledge-base"])
async def clear_knowledge_base_endpoint(db: Session = Depends(get_db)):
    """
    Clear the entire knowledge base.
    Use with caution!
    
    Args:
        db: Database session
    
    Returns:
        Status message
    """
    logger.warning("Clearing knowledge base")
    
    try:
        # Clear the vector store
        clear_vector_store()
        
        logger.info("Knowledge base cleared successfully")
        return {"message": "Knowledge base cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {str(e)}"
        )


@router.post("/chat-summary", response_model=ChatSummaryResponse, tags=["admin", "chat-summaries"])
async def create_chat_summary_endpoint(summary: ChatSummaryCreate, db: Session = Depends(get_db)):
    """
    Create a new chat summary.
    
    Args:
        summary: Chat summary data
        db: Database session
    
    Returns:
        Created chat summary
    """
    logger.info(f"Creating chat summary: {summary.title}")
    
    try:
        # Create the chat summary
        db_summary = create_chat_summary(db, summary.dict())
        
        # Format response
        topics = []
        if db_summary.topics:
            import json
            try:
                topics = json.loads(db_summary.topics)
            except json.JSONDecodeError:
                topics = []
        
        metadata = {}
        if db_summary.metadata:
            try:
                metadata = json.loads(db_summary.metadata)
            except json.JSONDecodeError:
                metadata = {}
        
        logger.info(f"Chat summary created successfully: ID {db_summary.id}")
        return {
            "id": db_summary.id,
            "title": db_summary.title,
            "summary": db_summary.summary,
            "topics": topics,
            "timestamp": db_summary.timestamp.isoformat(),
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error creating chat summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chat summary: {str(e)}"
        )


@router.get("/test-query", tags=["admin", "knowledge-base"])
async def test_query_endpoint(
    query: str,
    top_k: int = 5,
    db: Session = Depends(get_db)
):
    """
    Test a query against the knowledge base.
    
    Args:
        query: Query string
        top_k: Number of results to return
        db: Database session
    
    Returns:
        Query results
    """
    logger.info(f"Testing query: '{query}'")
    
    try:
        # Query the knowledge base
        results = query_documents(query, top_k=top_k)
        
        logger.info(f"Query returned {len(results)} results")
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error testing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test query: {str(e)}"
        )