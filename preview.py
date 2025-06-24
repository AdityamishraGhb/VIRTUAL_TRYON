"""
Preview and Results Handler
"""
import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict, List

from enhance import create_thumbnail


router = APIRouter()


@router.get("/preview/{job_id}")
async def get_preview(job_id: str):
    """Get processing results and preview for a job"""
    job_folder = os.path.join("data", job_id)
    
    if not os.path.exists(job_folder):
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check job status
    status_file = os.path.join(job_folder, "status.txt")
    if not os.path.exists(status_file):
        return {"job_id": job_id, "status": "unknown"}
    
    with open(status_file, "r") as f:
        status = f.read().strip()
    
    response = {
        "job_id": job_id,
        "status": status,
        "files": {}
    }
    
    # List available files
    if os.path.exists(job_folder):
        for filename in os.listdir(job_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                response["files"][filename] = f"/static/{job_id}/{filename}"
    
    # Add pose data if available
    pose_file = os.path.join(job_folder, "pose_keypoints.json")
    if os.path.exists(pose_file):
        try:
            with open(pose_file, 'r') as f:
                response["pose_data"] = json.load(f)
        except Exception:
            pass
    
    # Add processing metadata
    if status == "completed":
        response["final_output"] = f"/static/{job_id}/final_output.jpg"
        response["download_url"] = f"/api/v1/download/{job_id}"
        
        # Create thumbnail if it doesn't exist
        thumbnail_path = os.path.join(job_folder, "thumbnail.jpg")
        if not os.path.exists(thumbnail_path):
            final_output_path = os.path.join(job_folder, "final_output.jpg")
            if os.path.exists(final_output_path):
                create_thumbnail(final_output_path, thumbnail_path)
        
        if os.path.exists(thumbnail_path):
            response["thumbnail"] = f"/static/{job_id}/thumbnail.jpg"
    
    return response


@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download final processed image"""
    job_folder = os.path.join("data", job_id)
    final_output = os.path.join(job_folder, "final_output.jpg")
    
    if not os.path.exists(final_output):
        raise HTTPException(status_code=404, detail="Final output not found")
    
    return FileResponse(
        final_output,
        media_type="image/jpeg",
        filename=f"saree_tryon_{job_id}.jpg"
    )


@router.get("/jobs")
async def list_jobs(limit: int = 10, offset: int = 0):
    """List all processing jobs"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return {"jobs": [], "total": 0}
    
    # Get all job directories
    job_dirs = [d for d in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, d))]
    
    # Sort by creation time (newest first)
    job_dirs.sort(key=lambda x: os.path.getctime(os.path.join(data_dir, x)), reverse=True)
    
    # Apply pagination
    total = len(job_dirs)
    paginated_jobs = job_dirs[offset:offset + limit]
    
    jobs = []
    for job_id in paginated_jobs:
        job_folder = os.path.join(data_dir, job_id)
        status_file = os.path.join(job_folder, "status.txt")
        
        # Get status
        status = "unknown"
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status = f.read().strip()
        
        # Get creation time
        created_at = os.path.getctime(job_folder)
        
        job_info = {
            "job_id": job_id,
            "status": status,
            "created_at": created_at,
            "preview_url": f"/api/v1/preview/{job_id}"
        }
        
        # Add thumbnail if available
        thumbnail_path = os.path.join(job_folder, "thumbnail.jpg")
        if os.path.exists(thumbnail_path):
            job_info["thumbnail"] = f"/static/{job_id}/thumbnail.jpg"
        
        jobs.append(job_info)
    
    return {
        "jobs": jobs,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a processing job and all its files"""
    job_folder = os.path.join("data", job_id)
    
    if not os.path.exists(job_folder):
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        import shutil
        shutil.rmtree(job_folder)
        return {"message": f"Job {job_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")


@router.get("/stats")
async def get_stats():
    """Get processing statistics"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return {
            "total_jobs": 0,
            "completed": 0,
            "processing": 0,
            "failed": 0
        }
    
    job_dirs = [d for d in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, d))]
    
    stats = {
        "total_jobs": len(job_dirs),
        "completed": 0,
        "processing": 0,
        "failed": 0,
        "unknown": 0
    }
    
    for job_id in job_dirs:
        status_file = os.path.join(data_dir, job_id, "status.txt")
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status = f.read().strip()
                if status == "completed":
                    stats["completed"] += 1
                elif status == "processing":
                    stats["processing"] += 1
                elif status.startswith("failed"):
                    stats["failed"] += 1
                else:
                    stats["unknown"] += 1
        else:
            stats["unknown"] += 1
    
    return stats