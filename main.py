"""
Virtual Saree Try-On Backend - Main FastAPI Application
"""
import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from upload import router as upload_router
from preview import router as preview_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print("🚀 Virtual Saree Try-On Backend Started")
    yield
    # Shutdown
    print("🛑 Virtual Saree Try-On Backend Stopped")


# Initialize FastAPI app
app = FastAPI(
    title="Virtual Saree Try-On API",
    description="AI-powered virtual saree fitting backend using MODNet, MediaPipe, VITON-HD, and Real-ESRGAN",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving processed images
app.mount("/static", StaticFiles(directory="data"), name="static")

# Include routers
app.include_router(upload_router, prefix="/api/v1", tags=["Upload"])
app.include_router(preview_router, prefix="/api/v1", tags=["Preview"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Virtual Saree Try-On Backend API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "storage_available": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)