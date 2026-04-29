import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from diarization.hybrid_pipeline import HybridPipeline
from dotenv import load_dotenv
import time
import torch

load_dotenv()

app = FastAPI(title="Rojak Diarization Service", description="Hybrid Neural-Classic Diarization API")

# Configuration
CHECKPOINT_PATH = "models/checkpoints/diarizer_finetuned.pt"

# Initialize Pipeline with Automatic Resource Scaling
print("Initializing Pipeline with Automatic Resource Detection...")
pipeline = HybridPipeline(
    model_path=CHECKPOINT_PATH, 
    n_speakers=None # Default to automatic detection
)

TEMP_DIR = "data/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/diarize")
async def diarize_audio(
    file: UploadFile = File(...), 
    threshold: float = 0.3,
    num_speakers: int = None
):
    """
    Hybrid Diarization: Neural Voice Activity + Classic Speaker Clustering.
    """
    allowed_extensions = ('.wav', '.mp3', '.flac')
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Unsupported format.")

    file_id = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_DIR, f"{file_id}_{file.filename}")
    start_time = time.time()

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process using Hybrid logic
        segments = pipeline.process(temp_path, threshold=threshold, num_speakers=num_speakers)
        
        processing_time = time.time() - start_time
        unique_speakers = sorted(list(set(s['speaker'] for s in segments)))

        return {
            "status": "success",
            "pipeline_type": "hybrid_neural_classic",
            "metadata": {
                "file_name": file.filename,
                "num_speakers_detected": len(unique_speakers),
                "speakers": unique_speakers,
                "processing_time_sec": round(processing_time, 2),
                "device": pipeline.device
            },
            "segments": segments
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": pipeline.device}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
