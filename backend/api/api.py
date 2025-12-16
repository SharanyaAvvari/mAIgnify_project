"""
api.py - Compatibility Backend for Frontend

✔ Redirects legacy /api/process calls to the real /predict logic
✔ No Gemini
✔ Uses same ML pipeline as main.py
✔ Prevents 501 frontend errors
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid
import shutil
import os

# Import predict logic from main.py
from main import predict  # 👈 VERY IMPORTANT

app = FastAPI(
    title="mAIgnify API (Compatibility Layer)",
    version="1.0.0"
)

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ROOT ====================
@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "mAIgnify API compatibility layer",
        "forwarding": "/api/process → /predict",
        "timestamp": datetime.now().isoformat()
    }

# ==================== FRONTEND COMPAT API ====================
@app.post("/api/process")
async def process_legacy(file: UploadFile = File(...)):
    """
    Legacy endpoint used by frontend.
    Internally forwards request to /predict
    """

    try:
        # Call main.py predict directly
        response = await predict(file)

        return {
            "status": "success",
            "source": "legacy_api",
            "data": response
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Legacy API failed: {str(e)}"
        )

# ==================== STATUS ====================
@app.get("/api/status")
async def api_status():
    return {
        "status": "online",
        "mode": "compatibility",
        "real_endpoint": "/predict",
        "timestamp": datetime.now().isoformat()
    }

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup():
    print("\n" + "=" * 60)
    print("✅ mAIgnify API compatibility layer started")
    print("🔁 /api/process → /predict")
    print("🚫 Gemini: REMOVED")
    print("✔ Model handled by main.py")
    print("=" * 60 + "\n")

# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
