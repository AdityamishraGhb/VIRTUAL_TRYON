"""
Upload Handler - Manages saree image uploads and processing pipeline
"""
import os
import uuid
import asyncio
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil

from remove_bg import remove_background
from pose_detect import detect_pose
from warp import warp_saree_onto_model
from enhance import enhance_image


router = APIRouter()


async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """Save uploaded file to destination"""
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        upload_file.file.close()
        return destination
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        # No additional code is needed here. The save_upload_file function is already complete.


async def process_saree_images(job_id: str, job_folder: str):
    """Background task to process saree images through AI pipeline"""
    try:
        # File paths
        body_path = os.path.join(job_folder, "body.jpg")
        pallu_path = os.path.join(job_folder, "pallu.jpg")
        blouse_path = os.path.join(job_folder, "blouse.jpg")
        
        # Step 1: Remove backgrounds
        print(f"🔄 [{job_id}] Removing backgrounds...")
        body_nobg = await remove_background(body_path, os.path.join(job_folder, "body_nobg.jpg"))
        pallu_nobg = await remove_background(pallu_path, os.path.join(job_folder, "pallu_nobg.jpg"))
        blouse_nobg = await remove_background(blouse_path, os.path.join(job_folder, "blouse_nobg.jpg"))
        
        # Step 2: Detect pose from body image
        print(f"🔄 [{job_id}] Detecting pose...")
        pose_data = await detect_pose(body_nobg, os.path.join(job_folder, "pose_keypoints.json"))
        
        # Step 3: Warp saree components onto virtual model
        print(f"🔄 [{job_id}] Warping saree onto model...")
        warped_image = await warp_saree_onto_model(
            body_nobg, pallu_nobg, blouse_nobg, pose_data,
            os.path.join(job_folder, "warped_output.jpg")
        )
        
        # Step 4: Enhance final image
        print(f"🔄 [{job_id}] Enhancing final image...")
        final_output = await enhance_image(
            warped_image, 
            os.path.join(job_folder, "final_output.jpg")
        )
        
        # Mark job as completed
        with open(os.path.join(job_folder, "status.txt"), "w") as f:
            f.write("completed")
            
        print(f"✅ [{job_id}] Processing completed successfully!")
        
    except Exception as e:
        # Mark job as failed
        with open(os.path.join(job_folder, "status.txt"), "w") as f:
            f.write(f"failed: {str(e)}")
        print(f"❌ [{job_id}] Processing failed: {str(e)}")


@router.post("/upload")
async def upload_saree_images(
    background_tasks: BackgroundTasks,
    body: UploadFile = File(..., description="Body/model image"),
    pallu: UploadFile = File(..., description="Saree pallu image"),
    blouse: UploadFile = File(..., description="Blouse image")
):
    """
    Upload 3 saree images and start AI processing pipeline
    
    Returns job_id for tracking processing status
    """
    
    # Validate file types
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    for file in [body, pallu, blouse]:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.content_type}. Only JPEG/PNG allowed."
            )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    job_folder = os.path.join(data_dir, job_id)
    os.makedirs(job_folder, exist_ok=True)
    
    try:
        # Save uploaded files
        await save_upload_file(body, os.path.join(job_folder, "body.jpg"))
        await save_upload_file(pallu, os.path.join(job_folder, "pallu.jpg"))
        await save_upload_file(blouse, os.path.join(job_folder, "blouse.jpg"))
        
        # Mark job as processing
        with open(os.path.join(job_folder, "status.txt"), "w") as f:
            f.write("processing")
        
        # Start background processing
        background_tasks.add_task(process_saree_images, job_id, job_folder)
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "Images uploaded successfully. Processing started.",
                "job_id": job_id,
                "status": "processing",
                "estimated_time": "2-3 minutes"
            }
        )
        
    except Exception as e:
        # Cleanup on failure
        if os.path.exists(job_folder):
            shutil.rmtree(job_folder)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing status for a job"""
    job_folder = os.path.join("data", job_id)
    status_file = os.path.join(job_folder, "status.txt")
    
    if not os.path.exists(job_folder):
        raise HTTPException(status_code=404, detail="Job not found")
    
    if not os.path.exists(status_file):
        return {"job_id": job_id, "status": "unknown"}
    
    with open(status_file, "r") as f:
        status = f.read().strip()
    
    response = {"job_id": job_id, "status": status}
    
    # Add output URL if completed
    if status == "completed":
        response["output_url"] = f"/static/{job_id}/final_output.jpg"
        response["preview_url"] = f"/api/v1/preview/{job_id}"
    
    return response