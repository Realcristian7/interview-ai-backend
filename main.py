import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import uvicorn

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI()

# Load Whisper model (CPU-safe for Railway free tier)
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

# -----------------------------
# Health check (IMPORTANT)
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Interview AI backend is running"}

# -----------------------------
# Audio interview endpoint
# -----------------------------
@app.post("/interview")
async def interview(audio: UploadFile = File(...)):
    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path)
        text = " ".join(segment.text for segment in segments)

        return {
            "language": info.language,
            "transcript": text
        }

    finally:
        os.remove(tmp_path)

# -----------------------------
# Railway / Production entrypoint
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
