# --- Imports ---
import os, uuid, json, traceback, io
from datetime import datetime
from typing import Optional, Tuple, Dict
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import logging
from logging.handlers import RotatingFileHandler
from pymongo import MongoClient
import time
from config import *
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
import pytz
from geopy.geocoders import Nominatim
from fastapi.encoders import jsonable_encoder

# --- Setup Logging ---
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Setup geolocator ---
geolocator = Nominatim(user_agent="chicken_disease_app")

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://barrojohnnems01:cddapiendpoint@cdd.gg9azyr.mongodb.net/?retryWrites=true&w=majority&appName=CDD")
MONGO_DB = os.getenv("MONGO_DB", "chicken_app")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "scan_results")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")

try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DB]
    mongo_collection = mongo_db[MONGO_COLLECTION]
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.error(f"MongoDB connection error: {str(e)}")
    mongo_client = None
    mongo_collection = None

# --- FastAPI Setup ---
app = FastAPI(title="Chicken Disease Detection API", version="1.0.0")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate Limit Middleware ---
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if current_time - t < RATE_LIMIT_WINDOW
        ]
        if len(self.requests[client_ip]) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})
        self.requests[client_ip].append(current_time)
        return await call_next(request)

app.add_middleware(RateLimitMiddleware)

# --- Load TFLite Model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "new-model-aug.tflite")

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TFLite model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

# --- Class Labels ---
label_map = {
    0: "Coccidiosis",
    1: "Newcastle Disease",
    2: "Healthy",
    3: "NonFecal",
    4: "Unknown"
}

# --- Preprocessing for ResNet50 ---
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Prediction ---
def run_prediction(image_bytes: bytes) -> Tuple[str, float, str, Dict[str, float]]:
    try:
        input_array = preprocess_image(image_bytes)
        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        class_probs = {label_map[i]: float(conf) for i, conf in enumerate(output_data)}
        sorted_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))

        top_class = max(class_probs, key=class_probs.get)
        confidence = class_probs[top_class]
        severity = "High" if confidence > 0.8 else "Moderate" if confidence > 0.4 else "Low"

        return top_class, confidence, severity, sorted_probs
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise RuntimeError(f"Prediction failed: {str(e)}")

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    latitude: Optional[str] = Form(None),
    longitude: Optional[str] = Form(None),
):
    try:
        if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            raise HTTPException(status_code=400, detail="Invalid image format")

        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="File too large")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        filename = f"{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
        upload_path = os.path.join(UPLOAD_DIR, filename)

        with open(upload_path, "wb") as buffer:
            buffer.write(content)

        prediction, confidence, severity, probabilities = run_prediction(content)
        image_url = f"{API_BASE_URL}/static/uploads/{filename}"

        location_name = "Unknown Location"
        try:
            if latitude and longitude:
                location = geolocator.reverse(f"{latitude}, {longitude}", language='en')
                if location and location.address:
                    location_name = location.address
        except Exception as geo_err:
            logger.warning(f"Reverse geocoding failed: {str(geo_err)}")

        tz = pytz.timezone("Asia/Manila")
        scanned_at = datetime.now(tz).isoformat()

        scan_data = {
            "result": prediction,
            "confidence": round(confidence, 3),
            "severity": severity,
            "probabilities": probabilities,
            "image_url": image_url,
            "location_name": location_name,
            "lat": latitude,
            "lon": longitude,
            "scanned_at": scanned_at,
        }

        db_id = None
        if mongo_collection is not None:
            try:
                result = mongo_collection.insert_one(scan_data)
                db_id = str(result.inserted_id)
                logger.info(f"Saved to MongoDB with ID: {db_id}")
            except Exception as db_error:
                logger.error(f"MongoDB insert failed: {str(db_error)}")
                logger.error(traceback.format_exc())

        scan_data.pop('_id', None)
        return JSONResponse({**scan_data, "id": db_id, "saved_to_db": db_id is not None})

    except HTTPException as he:
        raise he
    except RuntimeError as re:
        logger.error(f"Runtime error: {str(re)}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- Recent Scan Retrieval ---
@app.get("/scans")
async def get_recent_scans(limit: int = 100):
    try:
        if mongo_collection is None:
            raise HTTPException(status_code=500, detail="MongoDB not initialized")
        results = mongo_collection.find().sort("scanned_at", -1).limit(limit)
        scans = []
        for doc in results:
            doc["_id"] = str(doc["_id"])
            scans.append(jsonable_encoder(doc))
        return scans
    except Exception as e:
        logger.error(f"Failed to fetch recent scans: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch scan history")

# --- Serve Uploaded Files ---
app.mount("/static", StaticFiles(directory="static"), name="static")
