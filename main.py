# --- Imports ---
import os, uuid, json, traceback
from datetime import datetime
from typing import Optional, Tuple, Dict
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import platform
import logging
from logging.handlers import RotatingFileHandler
from pymongo import MongoClient
import time
from config import *
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
import pytz
from geopy.geocoders import Nominatim

try:
    if platform.system() == "Windows":
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    else:
        from tflite_runtime.interpreter import Interpreter
except ImportError as e:
    raise ImportError("Neither tflite_runtime nor tensorflow is available.") from e

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
app = FastAPI(
    title="Chicken Disease Detection API",
    description="API for detecting chicken diseases from images",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None
)

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
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < RATE_LIMIT_WINDOW
        ]

        if len(self.requests[client_ip]) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )

        self.requests[client_ip].append(current_time)
        return await call_next(request)

app.add_middleware(RateLimitMiddleware)

# --- Load Model and Labels ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "chicken_disease_efficientnetb4_model.tflite")
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "label_map.json")

try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(LABEL_MAP_PATH, "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    logger.info("Model and label map loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model or labels: {str(e)}")
    raise

# --- Health Check ---
@app.get("/test")
async def test():
    try:
        db_status = "connected" if mongo_collection else "not initialized"
        return {
            "status": "ok",
            "message": "Server is running",
            "model_loaded": True,
            "label_map": label_map,
            "database_status": db_status,
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# --- Prediction Core ---
def run_prediction(image_path: str) -> Tuple[str, float, str, Dict[str, float]]:
    try:
        image = Image.open(image_path).convert("RGB")
        input_shape = input_details[0]['shape']
        target_size = (input_shape[2], input_shape[1])
        image = image.resize(target_size)

        input_array = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_array = (input_array - mean) / std
        input_array = np.expand_dims(input_array, axis=0)

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

        prediction, confidence, severity, probabilities = run_prediction(upload_path)
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
        else:
            logger.warning("Skipping database save - MongoDB not initialized")

        response_data = {
            **scan_data,
            "id": db_id,
            "saved_to_db": db_id is not None,
        }

        return JSONResponse(response_data)

    except HTTPException as he:
        raise he
    except RuntimeError as re:
        logger.error(f"Runtime error: {str(re)}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- Serve uploaded images ---
app.mount("/static", StaticFiles(directory="static"), name="static")
