"""
Q&A-specific document chunking utilities.
"""

import logging
import re
from typing import List, Dict, Any, Optional

from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class QATextSplitter(RecursiveCharacterTextSplitter):
    """
    Text splitter optimized for Q&A content.
    Ensures questions and their answers stay together.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ):
        """Initialize with parameters specific to Q&A splitting."""
        # Q&A specific separators to ensure we don't break between a question and its answer
        separators = [
            "\n\n\nQuestion:", "\n\nQuestion:", "\nQuestion:", "Question:",
            "\n\n\nQ:", "\n\nQ:", "\nQ:", "Q:",
            "\n\n"
        ]
        
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **kwargs
        )
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks, ensuring Q&A pairs are kept together.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        # First, try to identify Q&A pairs
        qa_pattern = r'(?:Question:|Q:)\s*(.*?)(?=(?:(?:Question:|Q:)|\Z))'
        matches = list(re.finditer(qa_pattern, text, re.DOTALL))
        
        if matches:
            # If we found Q&A patterns, split by them
            chunks = []
            
            for i, match in enumerate(matches):
                qa_text = match.group(0).strip()
                
                # If the Q&A pair is too long, split it using the parent class method
                if len(qa_text) > self._chunk_size:
                    sub_chunks = super().split_text(qa_text)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(qa_text)
            
            return chunks
        else:
            # Fall back to regular splitting if no Q&A pattern is detected
            return super().split_text(text)

def process_qa_format(documents: List[Document]) -> List[Document]:
    """
    Process documents to optimize for Q&A format.
    
    Args:
        documents: List of documents to process
        
    Returns:
        Processed documents
    """
    processed_docs = []
    
    for doc in documents:
        content = doc.page_content
        
        # Standardize Q&A format
        # Replace various question formats with "Question:"
        content = re.sub(r'(?<!\w)Q:\s*', 'Question: ', content)
        content = re.sub(r'(?<!\w)A:\s*', 'Answer: ', content)
        
        # Update the document
        processed_docs.append(
            Document(page_content=content, metadata=doc.metadata)
        )
    
    return processed_docs

def split_documents_qa_aware(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents using Q&A-aware chunking.
    
    Args:
        documents: Documents to split
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of split documents
    """
    # Preprocess documents for Q&A format
    processed_docs = process_qa_format(documents)
    
    # Split using Q&A-aware splitter
    splitter = QATextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(processed_docs)