#!/usr/bin/env python
"""
Script for ingesting documents into the knowledge base.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.vector_store import ingest_documents, ingest_directory
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the document ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Marketing Knowledge Base"
    )
    
    # Create subparsers for different ingestion modes
    subparsers = parser.add_subparsers(dest="command", help="Ingestion command")
    
    # Parser for ingesting individual files
    file_parser = subparsers.add_parser("files", help="Ingest specific files")
    file_parser.add_argument(
        "files", 
        nargs="+", 
        help="Paths to files to ingest (absolute or relative to knowledge base directory)"
    )
    file_parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000, 
        help="Size of document chunks"
    )
    file_parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=200, 
        help="Overlap between chunks"
    )
    
    # Parser for ingesting a directory
    dir_parser = subparsers.add_parser("directory", help="Ingest a directory of files")
    dir_parser.add_argument(
        "directory", 
        help="Path to directory to ingest (absolute or relative to knowledge base directory)"
    )
    dir_parser.add_argument(
        "--glob-pattern", 
        default="**/*.txt", 
        help="Glob pattern for matching files"
    )
    dir_parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000, 
        help="Size of document chunks"
    )
    dir_parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=200, 
        help="Overlap between chunks"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Get settings
    settings = get_settings()
    
    # Process command
    if args.command == "files":
        # Convert relative paths to absolute paths
        files = []
        for file in args.files:
            if os.path.isabs(file):
                files.append(file)
            else:
                files.append(os.path.join(settings.knowledge_base_dir, file))
        
        # Validate files
        for file in files:
            if not os.path.exists(file):
                logger.error(f"File does not exist: {file}")
                return
        
        # Ingest files
        logger.info(f"Ingesting {len(files)} files...")
        doc_count = ingest_documents(
            document_paths=files,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        logger.info(f"Successfully ingested {len(files)} files ({doc_count} chunks)")
        
    elif args.command == "directory":
        # Convert relative path to absolute path
        if os.path.isabs(args.directory):
            directory_path = args.directory
        else:
            directory_path = os.path.join(settings.knowledge_base_dir, args.directory)
        
        # Validate directory
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return
        
        # Ingest directory
        logger.info(f"Ingesting directory: {directory_path} with pattern {args.glob_pattern}...")
        doc_count = ingest_directory(
            directory_path=directory_path,
            glob_pattern=args.glob_pattern,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        logger.info(f"Successfully ingested directory ({doc_count} chunks)")


if __name__ == "__main__":
    main()