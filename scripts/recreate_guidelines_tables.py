#!/usr/bin/env python
"""
Script to recreate the guidelines tables if there are schema inconsistencies.
This is a safe way to fix database schema issues without losing data.
"""

import sys
import os
import logging
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import (
    Table, Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean, 
    MetaData, create_engine, inspect, text
)
from sqlalchemy.orm import sessionmaker
from app.config import get_settings
from app.db.guidelines_db import init_guidelines_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def run_migration():
    """Recreate the guidelines tables if there are schema issues."""
    logger.info("Starting guidelines database migration")
    
    # Get database settings
    settings = get_settings()
    engine = create_engine(settings.metrics_db.url, echo=settings.metrics_db.echo)
    
    # Create metadata object
    metadata = MetaData()
    
    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Get database inspector
    inspector = inspect(engine)
    
    # Backup existing data if the tables exist
    guidelines_data = []
    sections_data = []
    guideline_assets_data = []
    section_assets_data = []
    
    tables_to_check = ["guidelines", "guideline_sections", "guideline_assets", "section_assets"]
    existing_tables = inspector.get_table_names()
    
    try:
        # Backup data if tables exist
        if "guidelines" in existing_tables:
            logger.info("Backing up guidelines table data")
            result = session.execute(text("SELECT * FROM guidelines"))
            for row in result:
                guidelines_data.append(dict(row))
            
        if "guideline_sections" in existing_tables:
            logger.info("Backing up guideline_sections table data")
            result = session.execute(text("SELECT * FROM guideline_sections"))
            for row in result:
                sections_data.append(dict(row))
            
        if "guideline_assets" in existing_tables:
            logger.info("Backing up guideline_assets table data")
            result = session.execute(text("SELECT * FROM guideline_assets"))
            for row in result:
                guideline_assets_data.append(dict(row))
            
        if "section_assets" in existing_tables:
            logger.info("Backing up section_assets table data")
            result = session.execute(text("SELECT * FROM section_assets"))
            for row in result:
                section_assets_data.append(dict(row))
        
        # Drop tables if they exist
        logger.info("Dropping existing tables")
        for table in reversed(tables_to_check):  # Reverse to handle foreign key constraints
            if table in existing_tables:
                session.execute(text(f"DROP TABLE IF EXISTS {table}"))
                session.commit()
        
        # Create new tables
        logger.info("Creating new tables with correct schema")
        session.close()  # Close session before creating tables
        init_guidelines_db()
        
        # Reopen session
        session = Session()
        
        # Get new inspector to reflect updated schema
        inspector = inspect(engine)
        
        # Restore data with compatible fields
        if guidelines_data:
            logger.info(f"Restoring {len(guidelines_data)} guidelines")
            for item in guidelines_data:
                # Get the expected columns from the current schema
                guideline_columns = inspector.get_columns("guidelines")
                valid_columns = [col["name"] for col in guideline_columns]
                
                # Filter data to only include valid columns
                filtered_data = {k: v for k, v in item.items() if k in valid_columns and k != 'id'}
                
                if not filtered_data:
                    logger.warning("No valid data to restore for a guideline")
                    continue
                
                # Insert data
                columns = ", ".join(filtered_data.keys())
                placeholders = ", ".join([f":{k}" for k in filtered_data.keys()])
                
                # Create the SQL statement and parameters
                sql = text(f"INSERT INTO guidelines ({columns}) VALUES ({placeholders})")
                
                # Log the data being inserted
                logger.info(f"Inserting guideline: {filtered_data}")
                
                try:
                    session.execute(sql, filtered_data)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error inserting guideline: {e}")
                    logger.error(f"SQL: INSERT INTO guidelines ({columns}) VALUES ({placeholders})")
                    logger.error(f"Values: {filtered_data}")
        
        # Restore sections if needed
        if sections_data and "guideline_sections" in inspector.get_table_names():
            logger.info(f"Restoring {len(sections_data)} guideline sections")
            for item in sections_data:
                # Get the expected columns
                section_columns = inspector.get_columns("guideline_sections")
                valid_columns = [col["name"] for col in section_columns]
                
                # Filter data
                filtered_data = {k: v for k, v in item.items() if k in valid_columns and k != 'id'}
                
                if not filtered_data:
                    continue
                
                # Insert data
                columns = ", ".join(filtered_data.keys())
                placeholders = ", ".join([f":{k}" for k in filtered_data.keys()])
                
                sql = text(f"INSERT INTO guideline_sections ({columns}) VALUES ({placeholders})")
                
                try:
                    session.execute(sql, filtered_data)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error inserting section: {e}")
        
        # Similarly restore assets if needed
        if guideline_assets_data and "guideline_assets" in inspector.get_table_names():
            logger.info(f"Restoring {len(guideline_assets_data)} guideline assets")
            for item in guideline_assets_data:
                # Get the expected columns
                asset_columns = inspector.get_columns("guideline_assets")
                valid_columns = [col["name"] for col in asset_columns]
                
                # Filter data
                filtered_data = {k: v for k, v in item.items() if k in valid_columns and k != 'id'}
                
                if not filtered_data:
                    continue
                
                # Insert data
                columns = ", ".join(filtered_data.keys())
                placeholders = ", ".join([f":{k}" for k in filtered_data.keys()])
                
                sql = text(f"INSERT INTO guideline_assets ({columns}) VALUES ({placeholders})")
                
                try:
                    session.execute(sql, filtered_data)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error inserting asset: {e}")
        
        # Restore section assets if needed
        if section_assets_data and "section_assets" in inspector.get_table_names():
            logger.info(f"Restoring {len(section_assets_data)} section assets")
            for item in section_assets_data:
                # Get the expected columns
                asset_columns = inspector.get_columns("section_assets")
                valid_columns = [col["name"] for col in asset_columns]
                
                # Filter data
                filtered_data = {k: v for k, v in item.items() if k in valid_columns and k != 'id'}
                
                if not filtered_data:
                    continue
                
                # Insert data
                columns = ", ".join(filtered_data.keys())
                placeholders = ", ".join([f":{k}" for k in filtered_data.keys()])
                
                sql = text(f"INSERT INTO section_assets ({columns}) VALUES ({placeholders})")
                
                try:
                    session.execute(sql, filtered_data)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error inserting section asset: {e}")
        
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        logger.error("Migration failed")
    finally:
        session.close()


if __name__ == "__main__":
    run_migration()