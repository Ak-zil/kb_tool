"""
Text processing utilities for the Marketing Knowledge Base.
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters that might affect parsing
    text = text.replace('\x00', '')
    
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text

def extract_qa_pairs(text: str) -> List[Dict[str, str]]:
    """
    Extract question-answer pairs from text.
    
    Args:
        text: Input text with potential Q&A format
    
    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    # Pattern for Q: and A: format
    pattern = r'Q:\s*(.*?)\s*(?=A:)A:\s*(.*?)(?=(?:Q:|\Z))'
    
    # Try to find pairs
    pairs = re.findall(pattern, text, re.DOTALL)
    
    if pairs:
        return [{'question': q.strip(), 'answer': a.strip()} for q, a in pairs]
    
    # Alternative format: Look for question marks followed by answers
    alt_pattern = r'(.*?\?)\s*(.*?)(?=(?:.*?\?|\Z))'
    alt_pairs = re.findall(alt_pattern, text, re.DOTALL)
    
    if alt_pairs:
        return [{'question': q.strip(), 'answer': a.strip()} for q, a in alt_pairs]
    
    return []

def preprocess_document_text(text: str) -> str:
    """
    Preprocess document text to improve chunking quality.
    
    Args:
        text: Raw document text
    
    Returns:
        Preprocessed text
    """
    # Clean the text
    text = clean_text(text)
    
    # Try to extract Q&A pairs and format them consistently
    qa_pairs = extract_qa_pairs(text)
    if qa_pairs:
        formatted_text = ""
        for pair in qa_pairs:
            formatted_text += f"Question: {pair['question']}\nAnswer: {pair['answer']}\n\n"
        return formatted_text
    
    return text