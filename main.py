from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid

app = FastAPI()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can replace this with your custom model path

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the frontend HTML page
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# Detection endpoint
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
        conf = float(box.conf[0])
        if label in counts:
            counts[label] += 1
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = f"static/{filename}"
    cv2.imwrite(save_path, image)

    return JSONResponse(content={
        "counts": counts,
        "image_url": f"/static/{filename}"
    })
