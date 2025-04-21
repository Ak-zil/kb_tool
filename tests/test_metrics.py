"""
Tests for the metrics database and API.
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.metrics_db import (
    init_db, 
    create_metric, 
    get_metrics, 
    get_metric_by_id,
    update_metric,
    delete_metric
)
from app.core.metrics_engine import format_metrics_for_context
from app.main import app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="function")
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Initialize database
    from app.db.metrics_db import Base
    Base.metadata.create_all(bind=engine)
    
    # Create session
    session = SessionLocal()
    
    yield session
    
    # Cleanup
    session.close()


@pytest.fixture(scope="function")
def client():
    """Create a test client."""
    return TestClient(app)


def test_create_metric(db_session):
    """Test creating a metric in the database."""
    # Create a test metric
    metric_data = {
        "name": "Customer Acquisition Cost",
        "value": 50.25,
        "unit": "$",
        "category": "Marketing",
        "subcategory": "Acquisition",
        "description": "Average cost to acquire a new customer"
    }
    
    db_metric = create_metric(db_session, metric_data)
    
    # Verify the metric was created
    assert db_metric.id is not None
    assert db_metric.name == metric_data["name"]
    assert db_metric.value == metric_data["value"]
    assert db_metric.unit == metric_data["unit"]
    assert db_metric.category == metric_data["category"]
    assert db_metric.subcategory == metric_data["subcategory"]
    assert db_metric.description == metric_data["description"]
    assert isinstance(db_metric.timestamp, datetime)


def test_get_metrics(db_session):
    """Test retrieving metrics from the database."""
    # Create test metrics
    metrics = [
        {
            "name": "Customer Acquisition Cost",
            "value": 50.25,
            "unit": "$",
            "category": "Marketing",
            "subcategory": "Acquisition"
        },
        {
            "name": "Churn Rate",
            "value": 5.2,
            "unit": "%",
            "category": "Customer",
            "subcategory": "Retention"
        },
        {
            "name": "Monthly Active Users",
            "value": 12500,
            "category": "Product",
            "subcategory": "Engagement"
        }
    ]
    
    for metric in metrics:
        create_metric(db_session, metric)
    
    # Test getting all metrics
    all_metrics = get_metrics(db_session)
    assert len(all_metrics) == 3
    
    # Test filtering by category
    marketing_metrics = get_metrics(db_session, category="Marketing")
    assert len(marketing_metrics) == 1
    assert marketing_metrics[0].name == "Customer Acquisition Cost"
    
    # Test filtering by subcategory
    retention_metrics = get_metrics(db_session, subcategory="Retention")
    assert len(retention_metrics) == 1
    assert retention_metrics[0].name == "Churn Rate"
    
    # Test pagination
    paginated_metrics = get_metrics(db_session, limit=2, skip=1)
    assert len(paginated_metrics) == 2


def test_update_metric(db_session):
    """Test updating a metric in the database."""
    # Create a test metric
    metric_data = {
        "name": "Revenue",
        "value": 100000,
        "unit": "$",
        "category": "Financial"
    }
    
    db_metric = create_metric(db_session, metric_data)
    metric_id = db_metric.id
    
    # Update the metric
    update_data = {
        "value": 120000,
        "description": "Monthly revenue"
    }
    
    updated_metric = update_metric(db_session, metric_id, update_data)
    
    # Verify the update
    assert updated_metric.id == metric_id
    assert updated_metric.name == metric_data["name"]  # Unchanged
    assert updated_metric.value == update_data["value"]  # Changed
    assert updated_metric.description == update_data["description"]  # Added


def test_delete_metric(db_session):
    """Test deleting a metric from the database."""
    # Create a test metric
    metric_data = {
        "name": "Test Metric",
        "value": 42,
        "category": "Test"
    }
    
    db_metric = create_metric(db_session, metric_data)
    metric_id = db_metric.id
    
    # Delete the metric
    result = delete_metric(db_session, metric_id)
    assert result is True
    
    # Verify the metric was deleted
    deleted_metric = get_metric_by_id(db_session, metric_id)
    assert deleted_metric is None
    
    # Try to delete a non-existent metric
    result = delete_metric(db_session, 9999)
    assert result is False


def test_metrics_api_endpoints(client):
    """Test the metrics API endpoints."""
    # Create a test metric
    create_response = client.post(
        "/api/metrics/",
        json={
            "name": "API Test Metric",
            "value": 75.5,
            "unit": "%",
            "category": "API Test",
            "description": "A test metric created via API"
        }
    )
    
    assert create_response.status_code == 201
    metric_data = create_response.json()
    metric_id = metric_data["id"]
    
    # Get all metrics
    get_response = client.get("/api/metrics/")
    assert get_response.status_code == 200
    metrics = get_response.json()
    assert len(metrics) >= 1
    
    # Get a specific metric
    get_one_response = client.get(f"/api/metrics/{metric_id}")
    assert get_one_response.status_code == 200
    assert get_one_response.json()["id"] == metric_id
    
    # Update a metric
    update_response = client.put(
        f"/api/metrics/{metric_id}",
        json={
            "value": 80.0,
            "description": "Updated description"
        }
    )
    assert update_response.status_code == 200
    updated_data = update_response.json()
    assert updated_data["value"] == 80.0
    assert updated_data["description"] == "Updated description"
    
    # Get formatted metrics
    format_response = client.get("/api/metrics/formatted")
    assert format_response.status_code == 200
    assert "formatted_text" in format_response.json()
    assert len(format_response.json()["formatted_text"]) > 0
    
    # Get metrics summary
    summary_response = client.get("/api/metrics/summary")
    assert summary_response.status_code == 200
    assert "summary" in summary_response.json()
    assert "metrics_count" in summary_response.json()
    assert summary_response.json()["metrics_count"] >= 1
    
    # Delete the metric
    delete_response = client.delete(f"/api/metrics/{metric_id}")
    assert delete_response.status_code == 204


def test_format_metrics_for_context():
    """Test formatting metrics for context."""
    # Test metrics
    metrics = [
        {
            "id": 1,
            "name": "Customer Acquisition Cost",
            "value": 50.25,
            "unit": "$",
            "category": "Marketing",
            "subcategory": "Acquisition",
            "timestamp": datetime.utcnow().isoformat(),
            "description": "Average cost to acquire a new customer"
        },
        {
            "id": 2,
            "name": "Churn Rate",
            "value": 5.2,
            "unit": "%",
            "category": "Customer",
            "subcategory": "Retention",
            "timestamp": datetime.utcnow().isoformat(),
            "description": "Monthly customer churn rate"
        }
    ]
    
    # Format metrics
    formatted_text = format_metrics_for_context(metrics)
    
    # Check that the formatted text contains expected elements
    assert "MARKETING" in formatted_text
    assert "CUSTOMER" in formatted_text
    assert "Customer Acquisition Cost" in formatted_text
    assert "Churn Rate" in formatted_text
    assert "$50.25" in formatted_text or "$50.25" in formatted_text
    assert "5.20%" in formatted_text