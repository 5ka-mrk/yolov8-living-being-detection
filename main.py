from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid

app = FastAPI()

# Load model
model = YOLO("yolov8n.pt")  # Or use custom trained model

# Serve static frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(image)[0]

    names = model.names
    counts = {"person": 0, "cat": 0, "dog": 0}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        if label in counts:
            counts[label] += 1

    return JSONResponse(content=counts)
