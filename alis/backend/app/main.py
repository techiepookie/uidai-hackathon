# backend/app/main.py
"""
ALIS - Aadhaar Lifecycle Intelligence System
Main FastAPI application entry point.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import time
from datetime import datetime
from pathlib import Path
from loguru import logger
import os

from app.config import settings
from app.database import engine, Base, init_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting ALIS application...")
    init_database()
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down ALIS application...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Predictive intelligence system for Aadhaar lifecycle management",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Frontend directory as static files
# We need to go up from backend/app/main.py -> backend/app -> backend -> alis -> Frontend
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "Frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    logger.info(f"Mounted static files from {FRONTEND_DIR}")
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}")


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.debug(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


# Import and register routers
from app.routers import pincodes, analytics, predictions, anomalies

app.include_router(pincodes.router, prefix=settings.API_V1_PREFIX)
app.include_router(analytics.router, prefix=settings.API_V1_PREFIX)
app.include_router(predictions.router, prefix=settings.API_V1_PREFIX)
app.include_router(anomalies.router, prefix=settings.API_V1_PREFIX)


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    System health check endpoint.
    Used for monitoring and load balancer health checks.
    """
    from sqlalchemy import text
    from app.database import SessionLocal
    
    checks = {
        "database": False,
        "models_loaded": True,
        "api_responsive": True
    }
    
    # Check database connection
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        checks["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    
    all_healthy = all(checks.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION,
        "checks": checks
    }


# Root redirect to docs
@app.get("/")
async def root():
    """Root endpoint - redirect to API documentation."""
    return {
        "message": "Welcome to ALIS - Aadhaar Lifecycle Intelligence System",
        "version": settings.APP_VERSION,
        "docs": "/api/docs",
        "health": "/health"
    }


# API info endpoint
@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "endpoints": {
            "pincodes": f"{settings.API_V1_PREFIX}/pincodes",
            "analytics": f"{settings.API_V1_PREFIX}/analytics",
            "predictions": f"{settings.API_V1_PREFIX}/predictions",
            "anomalies": f"{settings.API_V1_PREFIX}/anomalies"
        },
        "documentation": "/api/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )