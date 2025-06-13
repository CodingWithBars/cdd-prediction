# --- Imports ---
import os, uuid, json, traceback
from datetime import datetime
from typing import Optional, Tuple, Dict
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from supabase import create_client, Client
from PIL import Image
import numpy as np
import tensorflow as tf
import logging
from logging.handlers import RotatingFileHandler
import time
from config import *
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict

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

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nkgpheijeocinaulhgfr.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5rZ3BoZWlqZW9jaW5hdWxoZ2ZyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDgzNDMwNjEsImV4cCI6MjA2MzkxOTA2MX0.MmKsCmkkYtRgO9cx-E3_zMqc0Vs1OClPl3QVosK5z0I")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://192.168.2.7:8080")

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Supabase client: {str(e)}")
    supabase = None

# --- FastAPI Setup ---
app = FastAPI(
    title="Chicken Disease Detection API",
    description="API for detecting chicken diseases from images",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key Security ---
api_key_header = APIKeyHeader(name=API_KEY_HEADER)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is missing")
    return api_key

# --- Rate Limiting Middleware ---
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

# --- Load Model and Label Map ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "chicken_disease_efficientnetb4_model.tflite")
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "label_map.json")

try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    logger.info(f"Loading label map from: {LABEL_MAP_PATH}")

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(LABEL_MAP_PATH, "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    logger.info("Model loaded successfully")
    logger.info(f"Input details: {input_details}")
    logger.info(f"Output details: {output_details}")
    logger.info(f"Label map loaded: {label_map}")
except Exception as e:
    logger.error(f"Error loading model or label map: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# --- Health Check Endpoint ---
@app.get("/test")
async def test():
    try:
        if supabase:
            test_query = supabase.table("scan_results").select("*").limit(1).execute()
            db_status = "connected" if not test_query.error else "error"
        else:
            db_status = "not initialized"

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

# --- Prediction Logic diri---
def run_prediction(image_path: str) -> Tuple[str, float, str, Dict[str, float]]:
    try:
        image = Image.open(image_path).convert("RGB")
        target_size = (input_details[0]['shape'][2], input_details[0]['shape'][1])
        image = image.resize(target_size)

        input_array = np.array(image, dtype=np.float32) / 255.0
        input_array = np.expand_dims(input_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        class_probs = {label_map[i]: float(conf) for i, conf in enumerate(output_data)}
        sorted_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))

        top_class = max(class_probs, key=class_probs.get)
        confidence = class_probs[top_class]
        risk_level = (
            "High" if confidence > 0.8 else
            "Moderate" if confidence > 0.4 else
            "Low"
        )

        return top_class, confidence, risk_level, sorted_probs
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise RuntimeError(f"Prediction failed: {str(e)}")

# --- Predict Endpoint to save unta sa db for production ---
@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    latitude: Optional[str] = Form(None),
    longitude: Optional[str] = Form(None)
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
        scan_data = {
            "result": prediction,
            "confidence": round(confidence, 3),
            "severity": severity,
            "probabilities": json.dumps(probabilities),
            "image_url": image_url,
            "location_name": "Your Farm",
            "lat": latitude,
            "lon": longitude,
            "scanned_at": datetime.utcnow().isoformat(),
        }

        db_id = None
        if supabase:
            try:
                logger.info(f"Inserting scan data: {json.dumps(scan_data, indent=2)}")
                sb_res = supabase.table("scan_results").insert([scan_data]).execute()

                if hasattr(sb_res, 'error') and sb_res.error:
                    logger.error(f"Supabase error: {sb_res.error}")
                    raise HTTPException(status_code=500, detail=f"Database error: {sb_res.error}")

                if sb_res.data:
                    db_id = sb_res.data[0]["id"]
                    logger.info(f"Saved to database with ID: {db_id}")
                else:
                    logger.warning("No data returned from Supabase insert")
            except Exception as db_error:
                logger.error(f"Supabase insert failed: {str(db_error)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            logger.warning("Skipping database save - Supabase not initialized")

        response_data = {
            **scan_data,
            "id": db_id,
            "saved_to_db": db_id is not None,
            "probabilities": probabilities
        }

        return JSONResponse(response_data)

    except HTTPException as he:
        raise he
    except RuntimeError as re:
        logger.error(f"Runtime error: {str(re)}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- Static Files to save prediction locally --- gamit AsyncStorage
app.mount("/static", StaticFiles(directory="static"), name="static")
