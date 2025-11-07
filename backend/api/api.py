"""
API.py - Alternative Backend (Gemini REMOVED)

⚠️ NOTE: It's recommended to use main.py instead of this file.
This file has been fixed to remove Gemini dependency.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(title="mAIstro API (Legacy)", version="0.5.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== REMOVED GEMINI ====================
# ALL Gemini code has been removed
# NO external API dependencies
# ======================================================

class TaskRequest(BaseModel):
    prompt: str
    file_ids: List[str] = []

@app.get("/")
async def root():
    return {
        "message": "mAIstro Legacy API",
        "warning": "This is the old API file. Please use main.py instead",
        "recommendation": "Run: python main.py"
    }

@app.post("/api/process")
async def process_task(request: TaskRequest):
    """
    Process task - LOCAL ONLY (no external APIs)
    """
    return {
        "status": "processed",
        "message": "Task completed locally",
        "agent": "local_agent",
        "response": f"Processed: {request.prompt}",
        "note": "Using local processing only - no external APIs"
    }

@app.get("/api/status")
async def get_status():
    """Check API status"""
    return {
        "status": "online",
        "version": "0.5.0",
        "external_apis": "None (all local)",
        "recommendation": "Use main.py for full functionality"
    }

@app.get("/api/available")
async def available_agents():
    """List available agents"""
    return {
        "agents": ["local_eda", "local_classifier"],
        "note": "For full agent system, use main.py"
    }

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup():
    print("\n" + "="*60)
    print("⚠️  Running LEGACY api.py")
    print("="*60)
    print("❌ Gemini code has been removed")
    print("✅ No external API dependencies")
    print("💡 RECOMMENDED: Use main.py instead")
    print("   Run: python main.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    print("⚠️  Starting legacy API (api.py)")
    print("💡 Recommendation: Use main.py instead for full features")
    uvicorn.run(app, host="0.0.0.0", port=8000)