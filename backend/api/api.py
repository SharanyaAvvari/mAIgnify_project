"""
FastAPI Backend for mAIstro
Connects frontend to the multi-agentic system
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import json
import uuid
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path

 #sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import MAIstroSystem
try:
    from backend.main import MAIstroSystem

    MAISTRO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import MAIstroSystem: {e}")
    print("📝 Running in demo mode")
    MAISTRO_AVAILABLE = False

# Create FastAPI app
app = FastAPI(title="mAIstro API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system
maistro_system = None

# Storage paths
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("outputs")
JOBS_DB = Path("jobs.json")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Jobs storage
jobs_store = {}

if JOBS_DB.exists():
    try:
        with open(JOBS_DB, 'r') as f:
            jobs_store = json.load(f)
    except:
        jobs_store = {}


# Pydantic models
class PromptRequest(BaseModel):
    prompt: str
    user_id: str
    file_ids: Optional[List[str]] = []


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


def save_jobs_db():
    """Save jobs to disk"""
    try:
        with open(JOBS_DB, 'w') as f:
            json.dump(jobs_store, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving jobs: {e}")


def initialize_system():
    """Initialize mAIstro system"""
    global maistro_system
    if not MAISTRO_AVAILABLE:
        print("⚠️  mAIstro system not available - running in demo mode")
        return False
    
    if maistro_system is None:
        try:
            maistro_system = MAIstroSystem()
            return True
        except Exception as e:
            print(f"Error initializing mAIstro: {e}")
            return False
    return True


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("\n" + "="*60)
    print("🚀 Starting mAIstro API Server")
    print("="*60)
    
    if initialize_system():
        print("✓ mAIstro system initialized")
    else:
        print("⚠️  Running in demo mode (mAIstro not fully initialized)")
    
    print("✓ Server ready")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "mAIstro API",
        "version": "1.0.0",
        "message": "Welcome to mAIstro API"
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    system_ready = maistro_system is not None
    return {
        "status": "healthy" if system_ready else "demo_mode",
        "system_initialized": system_ready,
        "active_jobs": len([j for j in jobs_store.values() if j.get('status') == 'running']),
        "total_jobs": len(jobs_store)
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file"""
    try:
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/submit", response_model=JobResponse)
async def submit_prompt(request: PromptRequest, background_tasks: BackgroundTasks):
    """Submit a prompt for processing"""
    
    job_id = str(uuid.uuid4())
    
    job = {
        "job_id": job_id,
        "user_id": request.user_id,
        "prompt": request.prompt,
        "file_ids": request.file_ids,
        "status": "queued",
        "progress": 0,
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    jobs_store[job_id] = job
    save_jobs_db()
    
    # Add to background tasks
    background_tasks.add_task(process_job, job_id, request.prompt, request.file_ids)
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message="Job submitted successfully"
    )


async def process_job(job_id: str, prompt: str, file_ids: List[str]):
    """Process a job in background"""
    try:
        # Update status
        jobs_store[job_id]["status"] = "running"
        jobs_store[job_id]["progress"] = 10
        jobs_store[job_id]["updated_at"] = datetime.now().isoformat()
        save_jobs_db()
        
        # Create output directory
        job_output_dir = RESULTS_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)
        
        # Progress update
        jobs_store[job_id]["progress"] = 30
        save_jobs_db()
        
        # Execute (demo mode if system not available)
        if maistro_system:
            result = maistro_system.execute_query(prompt, verbose=True)
        else:
            # Demo mode - simulate processing
            import time
            time.sleep(2)
            result = {
                'status': 'success',
                'message': 'Demo mode - job completed'
            }
            
            # Create demo output file
            demo_file = job_output_dir / 'demo_result.txt'
            with open(demo_file, 'w') as f:
                f.write(f"Query: {prompt}\n\n")
                f.write("This is a demo result.\n")
                f.write("In full mode, actual AI processing would occur here.\n")
        
        # Progress update
        jobs_store[job_id]["progress"] = 90
        save_jobs_db()
        
        # Gather output files
        output_files = list(job_output_dir.rglob("*.*"))
        output_info = [
            {
                "name": f.name,
                "path": str(f.relative_to(RESULTS_DIR)),
                "size": f.stat().st_size,
                "type": f.suffix
            }
            for f in output_files if f.is_file()
        ]
        
        # Update with success
        jobs_store[job_id].update({
            "status": "completed",
            "progress": 100,
            "result": {
                "status": result.get("status", "success"),
                "output_files": output_info,
                "output_dir": str(job_output_dir.relative_to(RESULTS_DIR)),
                "summary": str(result.get("message", "Job completed"))[:500]
            },
            "updated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        jobs_store[job_id].update({
            "status": "failed",
            "progress": 0,
            "error": str(e),
            "updated_at": datetime.now().isoformat()
        })
    
    finally:
        save_jobs_db()


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**jobs_store[job_id])


@app.get("/api/jobs/user/{user_id}")
async def get_user_jobs(user_id: str):
    """Get all jobs for a user"""
    user_jobs = [
        job for job in jobs_store.values()
        if job.get("user_id") == user_id
    ]
    
    user_jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "user_id": user_id,
        "total_jobs": len(user_jobs),
        "jobs": user_jobs
    }


@app.get("/api/results/{job_id}/{filename}")
async def download_result(job_id: str, filename: str):
    """Download result file"""
    file_path = RESULTS_DIR / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job"""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete files
    job_output_dir = RESULTS_DIR / job_id
    if job_output_dir.exists():
        import shutil
        shutil.rmtree(job_output_dir)
    
    del jobs_store[job_id]
    save_jobs_db()
    
    return {"message": "Job deleted successfully"}


@app.get("/api/examples")
async def get_examples():
    """Get example prompts"""
    return {
        "examples": [
            {
                "title": "Exploratory Data Analysis",
                "prompt": "Perform comprehensive EDA on my dataset and save visualizations",
                "category": "Data Analysis"
            },
            {
                "title": "Feature Selection",
                "prompt": "Analyze feature importance and save top 10 features",
                "category": "Feature Engineering"
            },
            {
                "title": "Classification Model",
                "prompt": "Train a classification model on my data",
                "category": "Machine Learning"
            },
            {
                "title": "Simple Query",
                "prompt": "What can you do?",
                "category": "Information"
            }
        ]
    }


# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)