from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel
import requests
import os

app = FastAPI()

model = WhisperModel("base", compute_type="int8")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

@app.post("/interview")
async def interview(audio: UploadFile):
    temp_path = f"/tmp/{audio.filename}"

    with open(temp_path, "wb") as f:
        f.write(await audio.read())

    segments, _ = model.transcribe(temp_path)
    question = " ".join([s.text for s in segments])

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You answer interview questions clearly and briefly."},
            {"role": "user", "content": question}
        ]
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )

    answer = response.json()["choices"][0]["message"]["content"]

    return {"question": question, "answer": answer}
