#!/usr/bin/env python
"""
Script for updating metrics in the database.
Can be used to import metrics from CSV or JSON files.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.metrics_db import init_db, create_metric, get_metrics
from app.config import get_settings
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_db_session():
    """Create and return a database session."""
    settings = get_settings()
    engine = create_engine(settings.metrics_db.url, echo=settings.metrics_db.echo)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Initialize the database if it doesn't exist
    init_db()
    
    return SessionLocal()


def import_from_csv(file_path: str, db: Session):
    """
    Import metrics from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        db: Database session
    
    Returns:
        Number of metrics imported
    """
    logger.info(f"Importing metrics from CSV: {file_path}")
    
    metrics_count = 0
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Convert value to float
                try:
                    value = float(row.get("value", 0))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value in row: {row}")
                    continue
                
                # Create metric data
                metric_data = {
                    "name": row.get("name", "").strip(),
                    "value": value,
                    "unit": row.get("unit", "").strip() or None,
                    "category": row.get("category", "").strip() or "General",
                    "subcategory": row.get("subcategory", "").strip() or None,
                    "description": row.get("description", "").strip() or None
                }
                
                # Handle metadata if present
                metadata_str = row.get("metadata", "").strip()
                if metadata_str:
                    try:
                        metric_data["metadata"] = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        # If not valid JSON, store as string
                        metric_data["metadata"] = {"text": metadata_str}
                
                # Validate required fields
                if not metric_data["name"]:
                    logger.warning(f"Skipping row with missing name: {row}")
                    continue
                
                # Create metric
                try:
                    create_metric(db, metric_data)
                    metrics_count += 1
                except Exception as e:
                    logger.error(f"Error creating metric {metric_data['name']}: {str(e)}")
        
        logger.info(f"Successfully imported {metrics_count} metrics from CSV")
        return metrics_count
        
    except Exception as e:
        logger.error(f"Error importing from CSV: {str(e)}")
        raise


def import_from_json(file_path: str, db: Session):
    """
    Import metrics from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        db: Database session
    
    Returns:
        Number of metrics imported
    """
    logger.info(f"Importing metrics from JSON: {file_path}")
    
    metrics_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
            
            # Handle different JSON formats
            metrics_list = []
            
            if isinstance(data, list):
                # List of metrics
                metrics_list = data
            elif isinstance(data, dict) and "metrics" in data:
                # Object with metrics array
                metrics_list = data["metrics"]
            elif isinstance(data, dict):
                # Single metric
                metrics_list = [data]
            
            for metric_data in metrics_list:
                # Convert value to float
                try:
                    value = float(metric_data.get("value", 0))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value in metric: {metric_data}")
                    continue
                
                # Create metric data
                processed_data = {
                    "name": metric_data.get("name", "").strip(),
                    "value": value,
                    "unit": metric_data.get("unit", "").strip() or None,
                    "category": metric_data.get("category", "").strip() or "General",
                    "subcategory": metric_data.get("subcategory", "").strip() or None,
                    "description": metric_data.get("description", "").strip() or None
                }
                
                # Handle metadata if present
                if "metadata" in metric_data:
                    processed_data["metadata"] = metric_data["metadata"]
                
                # Validate required fields
                if not processed_data["name"]:
                    logger.warning(f"Skipping metric with missing name: {metric_data}")
                    continue
                
                # Create metric
                try:
                    create_metric(db, processed_data)
                    metrics_count += 1
                except Exception as e:
                    logger.error(f"Error creating metric {processed_data['name']}: {str(e)}")
        
        logger.info(f"Successfully imported {metrics_count} metrics from JSON")
        return metrics_count
        
    except Exception as e:
        logger.error(f"Error importing from JSON: {str(e)}")
        raise


def export_to_json(file_path: str, db: Session, category: str = None):
    """
    Export metrics to a JSON file.
    
    Args:
        file_path: Path to the output JSON file
        db: Database session
        category: Optional category filter
    
    Returns:
        Number of metrics exported
    """
    logger.info(f"Exporting metrics to JSON: {file_path}")
    
    try:
        # Get metrics from database
        metrics = get_metrics(db, category=category)
        
        # Format metrics for export
        metrics_data = []
        for metric in metrics:
            # Parse metadata if available
            metadata = {}
            if metric.metadata:
                try:
                    metadata = json.loads(metric.metadata)
                except json.JSONDecodeError:
                    metadata = {}
            
            metrics_data.append({
                "id": metric.id,
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "category": metric.category,
                "subcategory": metric.subcategory,
                "timestamp": metric.timestamp.isoformat(),
                "description": metric.description,
                "metadata": metadata
            })
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump({"metrics": metrics_data}, jsonfile, indent=2)
        
        logger.info(f"Successfully exported {len(metrics_data)} metrics to JSON")
        return len(metrics_data)
        
    except Exception as e:
        logger.error(f"Error exporting to JSON: {str(e)}")
        raise


def main():
    """Main entry point for the metrics update script."""
    parser = argparse.ArgumentParser(
        description="Update metrics in the Marketing Knowledge Base"
    )
    
    # Create subparsers for different operations
    subparsers = parser.add_subparsers(dest="command", help="Metrics command")
    
    # Parser for importing from CSV
    csv_parser = subparsers.add_parser("import-csv", help="Import metrics from CSV file")
    csv_parser.add_argument(
        "file", 
        help="Path to CSV file to import"
    )
    
    # Parser for importing from JSON
    json_parser = subparsers.add_parser("import-json", help="Import metrics from JSON file")
    json_parser.add_argument(
        "file", 
        help="Path to JSON file to import"
    )
    
    # Parser for exporting to JSON
    export_parser = subparsers.add_parser("export", help="Export metrics to JSON file")
    export_parser.add_argument(
        "file", 
        help="Path to output JSON file"
    )
    export_parser.add_argument(
        "--category", 
        help="Filter metrics by category"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Get database session
    db = get_db_session()
    
    try:
        # Process command
        if args.command == "import-csv":
            import_from_csv(args.file, db)
            
        elif args.command == "import-json":
            import_from_json(args.file, db)
            
        elif args.command == "export":
            export_to_json(args.file, db, category=args.category)
            
    finally:
        db.close()


if __name__ == "__main__":
    main()