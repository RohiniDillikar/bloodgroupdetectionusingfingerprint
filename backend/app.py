import io
import sys
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Make backend folder importable
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from predict import model, transform, idx_to_class, device
import torch


app = FastAPI(
    title="Fingerprint Blood Group Detection API",
    description="AI-based fingerprint blood group prediction system",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://bloodgroupdetectionusingfingerprint-sigma.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Fingerprint Blood Group Detection API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check file type
    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/webp"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:
        # Read uploaded image
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # Preprocess
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Prediction
        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_idx = torch.max(
                probabilities,
                dim=1
            )

        predicted_idx = predicted_idx.item()
        confidence = confidence.item()

        blood_group = idx_to_class[predicted_idx]

        return {
            "success": True,
            "blood_group": blood_group,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )