# """
# Main FastAPI application entry point.
# """

# import logging
# from contextlib import asynccontextmanager

# from fastapi import FastAPI, Depends
# from fastapi.middleware.cors import CORSMiddleware

# from app.config import get_settings, AppConfig
# from app.api.chat import router as chat_router
# from app.api.admin import router as admin_router
# from app.api.metrics import router as metrics_router
# from app.db.vector_store import get_vector_store
# from app.db.metrics_db import init_db, get_db

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Lifespan context manager for FastAPI application.
#     Handles startup and shutdown events.
#     """
#     # Startup: Initialize databases and connections
#     logger.info("Initializing application...")
    
#     # Initialize metrics database
#     init_db()
    
#     # Initialize vector store
#     vector_store = get_vector_store()
#     logger.info(f"Vector store initialized: {vector_store.__class__.__name__}")
    
#     logger.info("Application startup complete")
    
#     yield
    
#     # Shutdown: Close connections and cleanup
#     logger.info("Shutting down application...")
#     logger.info("Application shutdown complete")


# def create_application() -> FastAPI:
#     """Create and configure the FastAPI application."""
#     settings = get_settings()
    
#     app = FastAPI(
#         title=settings.app_name,
#         description="A marketing research knowledge base tool for accessing company information",
#         version="0.1.0",
#         lifespan=lifespan,
#         debug=settings.debug
#     )
    
#     # Configure CORS
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],  # In production, this should be restricted
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )
    
#     # Include API routers
#     app.include_router(
#         chat_router,
#         prefix=f"{settings.api_prefix}/chat",
#         tags=["chat"]
#     )
#     app.include_router(
#         admin_router,
#         prefix=f"{settings.api_prefix}/admin",
#         tags=["admin"]
#     )
#     app.include_router(
#         metrics_router,
#         prefix=f"{settings.api_prefix}/metrics",
#         tags=["metrics"]
#     )
    
#     @app.get("/", tags=["health"])
#     async def health_check():
#         """Health check endpoint."""
#         return {"status": "healthy", "app_name": settings.app_name}
    
#     return app


# app = create_application()


# if __name__ == "__main__":
#     import uvicorn
    
#     uvicorn.run(
#         "app.main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )





"""
Updated main.py with static file serving configuration
"""

import logging
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Import StaticFiles
from fastapi.responses import FileResponse    # Import FileResponse for serving index.html

from app.config import get_settings, AppConfig
from app.api.chat import router as chat_router
from app.api.admin import router as admin_router
from app.api.metrics import router as metrics_router
from app.db.vector_store import get_vector_store
from app.db.metrics_db import init_db, get_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup: Initialize databases and connections
    logger.info("Initializing application...")
    
    # Initialize metrics database
    init_db()
    
    # Initialize vector store
    vector_store = get_vector_store()
    logger.info(f"Vector store initialized: {vector_store.__class__.__name__}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown: Close connections and cleanup
    logger.info("Shutting down application...")
    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="A marketing research knowledge base tool for accessing company information",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, this should be restricted
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(
        chat_router,
        prefix=f"{settings.api_prefix}/chat",
        tags=["chat"]
    )
    app.include_router(
        admin_router,
        prefix=f"{settings.api_prefix}/admin",
        tags=["admin"]
    )
    app.include_router(
        metrics_router,
        prefix=f"{settings.api_prefix}/metrics",
        tags=["metrics"]
    )
    
    # Get the base directory
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Define the static directory path
    static_dir = os.path.join(BASE_DIR, "static")
    
    # Add route for serving index.html at the root URL
    @app.get("/", tags=["frontend"])
    async def serve_frontend():
        """Serve the frontend application."""
        index_path = os.path.join(static_dir, "index.html")
        return FileResponse(index_path)
    
    # Mount the static directory
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "app_name": settings.app_name}
    
    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )