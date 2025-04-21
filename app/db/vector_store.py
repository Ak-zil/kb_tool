# """
# Vector database interface for storing and retrieving knowledge base documents.
# """

# import logging
# from typing import List, Dict, Any, Optional
# from functools import lru_cache

# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import TextLoader, DirectoryLoader, DocxLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.document_loaders.base import BaseLoader

# from app.config import get_settings

# logger = logging.getLogger(__name__)


# @lru_cache
# def get_embeddings():
#     """
#     Create and return an embedding model instance.
#     Uses LRU cache to prevent multiple instantiations.
#     """
#     settings = get_settings()
#     logger.info(f"Initializing embedding model: {settings.vector_db.embedding_model}")
    
#     return HuggingFaceEmbeddings(
#         model_name=settings.vector_db.embedding_model,
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )


# @lru_cache
# def get_vector_store():
#     """
#     Create and return a vector store instance.
#     Uses LRU cache to prevent multiple instantiations.
#     """
#     settings = get_settings()
#     embeddings = get_embeddings()
    
#     logger.info(f"Initializing vector store at: {settings.vector_db.persist_directory}")
    
#     return Chroma(
#         collection_name=settings.vector_db.collection_name,
#         embedding_function=embeddings,
#         persist_directory=settings.vector_db.persist_directory
#     )


# def ingest_documents(document_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200):
#     """
#     Ingest documents into the vector store.
    
#     Args:
#         document_paths: List of file paths to ingest
#         chunk_size: Size of text chunks
#         chunk_overlap: Overlap between chunks
    
#     Returns:
#         Number of documents ingested
#     """
#     logger.info(f"Ingesting {len(document_paths)} documents")
    
#     # Initialize text splitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#     )
    
#     # Load and split documents
#     documents = []
#     for path in document_paths:
#         try:
#             loader = TextLoader(path)
#             docs = loader.load()
#             split_docs = text_splitter.split_documents(docs)
#             documents.extend(split_docs)
#             logger.info(f"Processed {path}: {len(split_docs)} chunks")
#         except Exception as e:
#             logger.error(f"Error processing {path}: {str(e)}")
    
#     # Get vector store and add documents
#     vector_store = get_vector_store()
#     vector_store.add_documents(documents)
#     vector_store.persist()
    
#     logger.info(f"Successfully ingested {len(documents)} document chunks")
#     return len(documents)


# def ingest_directory(directory_path: str, glob_pattern: str = "**/*.txt", **kwargs):
#     """
#     Ingest all documents in a directory into the vector store.
    
#     Args:
#         directory_path: Path to directory
#         glob_pattern: Pattern for matching files
#         **kwargs: Additional arguments to pass to ingest_documents
    
#     Returns:
#         Number of documents ingested
#     """
#     logger.info(f"Ingesting directory: {directory_path} with pattern {glob_pattern}")
    
#     try:
#         loader = DirectoryLoader(
#             directory_path,
#             glob=glob_pattern,
#             loader_cls=TextLoader
#         )
#         docs = loader.load()
        
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=kwargs.get("chunk_size", 1000),
#             chunk_overlap=kwargs.get("chunk_overlap", 200),
#             length_function=len,
#         )
        
#         split_docs = text_splitter.split_documents(docs)
        
#         # Get vector store and add documents
#         vector_store = get_vector_store()
#         vector_store.add_documents(split_docs)
#         vector_store.persist()
        
#         logger.info(f"Successfully ingested {len(split_docs)} document chunks from directory")
#         return len(split_docs)
        
#     except Exception as e:
#         logger.error(f"Error ingesting directory {directory_path}: {str(e)}")
#         raise


# def query_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#     """
#     Query the vector store for relevant documents.
    
#     Args:
#         query: Query string
#         top_k: Number of results to return
    
#     Returns:
#         List of document chunks with their content and metadata
#     """
#     logger.info(f"Querying vector store: '{query}' (top_k={top_k})")
    
#     vector_store = get_vector_store()
#     results = vector_store.similarity_search_with_score(query, k=top_k)
    
#     # Format results
#     documents = []
#     for doc, score in results:
#         documents.append({
#             "content": doc.page_content,
#             "metadata": doc.metadata,
#             "relevance_score": float(score)  # Convert to float for JSON serialization
#         })
    
#     logger.info(f"Query returned {len(documents)} results")
#     return documents


# def clear_vector_store():
#     """
#     Clear all documents from the vector store.
#     Use with caution!
#     """
#     logger.warning("Clearing vector store")
    
#     vector_store = get_vector_store()
#     vector_store.delete_collection()
#     vector_store = get_vector_store()  # Recreate the collection
    
#     logger.info("Vector store cleared")
#     return True



"""
Vector database interface for storing and retrieving knowledge base documents.
"""

import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from app.db.document_loaders import get_document_loader, CustomDirectoryLoader
from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_embeddings():
    """
    Create and return an embedding model instance.
    Uses LRU cache to prevent multiple instantiations.
    """
    settings = get_settings()
    logger.info(f"Initializing embedding model: {settings.vector_db.embedding_model}")
    
    return HuggingFaceEmbeddings(
        model_name=settings.vector_db.embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@lru_cache
def get_vector_store():
    """
    Create and return a vector store instance.
    Uses LRU cache to prevent multiple instantiations.
    """
    settings = get_settings()
    embeddings = get_embeddings()
    
    logger.info(f"Initializing vector store at: {settings.vector_db.persist_directory}")
    
    return Chroma(
        collection_name=settings.vector_db.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.vector_db.persist_directory
    )


def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Create and return a text splitter with the specified parameters.
    
    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        Initialized text splitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Improved separators for better semantic chunking
        separators=["\n\n", "\n", ".", "?", "!", ";", ":", " ", ""]
    )


# def ingest_documents(document_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200):
#     """
#     Ingest documents into the vector store.
    
#     Args:
#         document_paths: List of file paths to ingest
#         chunk_size: Size of text chunks
#         chunk_overlap: Overlap between chunks
    
#     Returns:
#         Number of documents ingested
#     """
#     logger.info(f"Ingesting {len(document_paths)} documents")
    
#     # Initialize text splitter
#     text_splitter = get_text_splitter(chunk_size, chunk_overlap)
    
#     # Load and split documents
#     documents = []
#     for path in document_paths:
#         try:
#             loader = get_document_loader(path)
#             docs = loader.load()
#             split_docs = text_splitter.split_documents(docs)
#             documents.extend(split_docs)
#             logger.info(f"Processed {path}: {len(split_docs)} chunks")
#         except Exception as e:
#             logger.error(f"Error processing {path}: {str(e)}")
    
#     # Get vector store and add documents
#     vector_store = get_vector_store()
#     vector_store.add_documents(documents)
#     vector_store.persist()
    
#     logger.info(f"Successfully ingested {len(documents)} document chunks")
#     return len(documents)



def ingest_documents(document_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200, qa_format: bool = False):
    """
    Ingest documents into the vector store.
    
    Args:
        document_paths: List of file paths to ingest
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        qa_format: Whether to use Q&A-optimized processing
    
    Returns:
        Number of documents ingested
    """
    logger.info(f"Ingesting {len(document_paths)} documents")
    
    # Load documents
    documents = []
    for path in document_paths:
        try:
            loader = get_document_loader(path)
            docs = loader.load()
            
            # Preprocess document content if needed
            processed_docs = []
            for doc in docs:
                # Apply text preprocessing
                from app.utils.text_processing import preprocess_document_text
                processed_content = preprocess_document_text(doc.page_content)
                processed_docs.append(Document(page_content=processed_content, metadata=doc.metadata))
                
            documents.extend(processed_docs)
            logger.info(f"Loaded {path}: {len(processed_docs)} documents")
        except Exception as e:
            logger.error(f"Error loading {path}: {str(e)}")
    
    # Split documents
    split_docs = []
    if qa_format:
        # Use Q&A-optimized splitting
        from app.utils.qa_chunker import split_documents_qa_aware
        split_docs = split_documents_qa_aware(documents, chunk_size, chunk_overlap)
    else:
        # Use standard splitting
        text_splitter = get_text_splitter(chunk_size, chunk_overlap)
        split_docs = text_splitter.split_documents(documents)
    
    logger.info(f"Split into {len(split_docs)} chunks")
    
    # Get vector store and add documents
    if split_docs:
        vector_store = get_vector_store()
        vector_store.add_documents(split_docs)
        vector_store.persist()
        
        logger.info(f"Successfully ingested {len(split_docs)} document chunks")
        return len(split_docs)
    else:
        logger.warning("No document chunks created during ingestion")
        return 0


# def ingest_directory(directory_path: str, glob_pattern: str = "**/*.*", **kwargs):
#     """
#     Ingest all documents in a directory into the vector store.
    
#     Args:
#         directory_path: Path to directory
#         glob_pattern: Pattern for matching files
#         **kwargs: Additional arguments to pass to ingest_documents
    
#     Returns:
#         Number of documents ingested
#     """
#     logger.info(f"Ingesting directory: {directory_path} with pattern {glob_pattern}")
    
#     try:
#         loader = CustomDirectoryLoader(
#             path=directory_path,
#             glob=glob_pattern,
#         )
#         docs = loader.load()
        
#         text_splitter = get_text_splitter(
#             chunk_size=kwargs.get("chunk_size", 1000),
#             chunk_overlap=kwargs.get("chunk_overlap", 200),
#         )
        
#         split_docs = text_splitter.split_documents(docs)
        
#         # Get vector store and add documents
#         vector_store = get_vector_store()
#         vector_store.add_documents(split_docs)
#         vector_store.persist()
        
#         logger.info(f"Successfully ingested {len(split_docs)} document chunks from directory")
#         return len(split_docs)
        
#     except Exception as e:
#         logger.error(f"Error ingesting directory {directory_path}: {str(e)}")
#         raise


def ingest_directory(directory_path: str, glob_pattern: str = "**/*.*", **kwargs):
    """
    Ingest all documents in a directory into the vector store.
    
    Args:
        directory_path: Path to directory
        glob_pattern: Pattern for matching files
        **kwargs: Additional arguments to pass to ingest_documents
    
    Returns:
        Number of documents ingested
    """
    logger.info(f"Ingesting directory: {directory_path} with pattern {glob_pattern}")
    
    try:
        loader = CustomDirectoryLoader(
            path=directory_path,
            glob=glob_pattern,
        )
        docs = loader.load()
        
        # Process and preprocess documents
        processed_docs = []
        for doc in docs:
            # Apply text preprocessing
            from app.utils.text_processing import preprocess_document_text
            processed_content = preprocess_document_text(doc.page_content)
            processed_docs.append(Document(page_content=processed_content, metadata=doc.metadata))
        
        # Extract parameters
        chunk_size = kwargs.get("chunk_size", 1000)
        chunk_overlap = kwargs.get("chunk_overlap", 200)
        qa_format = kwargs.get("qa_format", False)
        
        # Split documents
        split_docs = []
        if qa_format:
            # Use Q&A-optimized splitting
            from app.utils.qa_chunker import split_documents_qa_aware
            split_docs = split_documents_qa_aware(processed_docs, chunk_size, chunk_overlap)
        else:
            # Use standard splitting
            text_splitter = get_text_splitter(chunk_size, chunk_overlap)
            split_docs = text_splitter.split_documents(processed_docs)
        
        # Get vector store and add documents
        if split_docs:
            vector_store = get_vector_store()
            vector_store.add_documents(split_docs)
            vector_store.persist()
            
            logger.info(f"Successfully ingested {len(split_docs)} document chunks from directory")
            return len(split_docs)
        else:
            logger.warning("No document chunks created during directory ingestion")
            return 0
        
    except Exception as e:
        logger.error(f"Error ingesting directory {directory_path}: {str(e)}")
        raise


def query_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the vector store for relevant documents.
    
    Args:
        query: Query string
        top_k: Number of results to return
    
    Returns:
        List of document chunks with their content and metadata
    """
    logger.info(f"Querying vector store: '{query}' (top_k={top_k})")
    
    vector_store = get_vector_store()
    
    # Use MMR (Maximum Marginal Relevance) for better diversity in results
    results = vector_store.max_marginal_relevance_search(query, k=top_k, fetch_k=top_k*2)
    
    # Format results
    documents = []
    for i, doc in enumerate(results):
        # For MMR search, we don't get scores directly, so calculate a synthetic score
        # based on position (higher positions = higher relevance)
        score = 1.0 - (i / (len(results) or 1))
        
        documents.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": score
        })
    
    logger.info(f"Query returned {len(documents)} results")
    return documents


def clear_vector_store():
    """
    Clear all documents from the vector store.
    Use with caution!
    """
    logger.warning("Clearing vector store")
    
    vector_store = get_vector_store()
    vector_store.delete_collection()
    vector_store = get_vector_store()  # Recreate the collection
    
    logger.info("Vector store cleared")
    return True