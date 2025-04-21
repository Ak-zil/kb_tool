"""
Custom document loaders for different file types.
"""

import logging
import os
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain.schema.document import Document

logger = logging.getLogger(__name__)

class DocxLoader(BaseLoader):
    """Loader for .docx files."""
    
    def __init__(self, file_path: str):
        """
        Initialize the DocxLoader.
        
        Args:
            file_path: Path to the .docx file
        """
        self.file_path = file_path
        
    def load(self) -> List[Document]:
        """
        Load the document content.
        
        Returns:
            List containing a single Document with the content of the file
        """
        try:
            # Try using python-docx first
            try:
                import docx
                doc = docx.Document(self.file_path)
                text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
            except Exception as e:
                logger.warning(f"Error using python-docx, falling back to docx2txt: {str(e)}")
                # Fall back to docx2txt
                import docx2txt
                text = docx2txt.process(self.file_path)
            
            metadata = {"source": self.file_path}
            return [Document(page_content=text, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Error loading .docx file {self.file_path}: {str(e)}")
            raise

def get_document_loader(file_path: str) -> BaseLoader:
    """
    Get the appropriate document loader based on file extension.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Document loader instance
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    if file_extension == '.docx':
        return DocxLoader(file_path)
    elif file_extension == '.txt':
        return TextLoader(file_path)
    else:
        # Default to TextLoader for other extensions
        logger.warning(f"No specific loader for {file_extension}, using TextLoader")
        return TextLoader(file_path)

class CustomDirectoryLoader(DirectoryLoader):
    """Custom directory loader that uses the appropriate loader for each file type."""
    
    def __init__(
        self,
        path: str,
        glob: str = "**/*",
        silent_errors: bool = False,
        load_hidden: bool = False,
    ):
        """Initialize with path."""
        super().__init__(
            path=path,
            glob=glob,
            loader_cls=None,  # We'll override the loading mechanism
            silent_errors=silent_errors,
            load_hidden=load_hidden,
        )
    
    def load(self) -> List[Document]:
        """
        Load documents from the directory, using the appropriate loader for each file.
        
        Returns:
            List of loaded documents
        """
        docs = []
        for file_path in self._get_filtered_paths():
            try:
                loader = get_document_loader(file_path)
                docs.extend(loader.load())
            except Exception as e:
                if self.silent_errors:
                    logger.warning(f"Error loading {file_path}: {str(e)}")
                    continue
                else:
                    raise
        return docs