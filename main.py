import os
import logging
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from io import BytesIO

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("NSFW-Detector")

# GPU and Resource Logging
logger.info("Checking system resources...")
gpu_available = torch.cuda.is_available()
logger.info(f"PyTorch CUDA Available: {gpu_available}")
if gpu_available:
    logger.info(f"Using CUDA Device: {torch.cuda.get_device_name(0)}")
else:
    logger.info("CUDA not available. Using CPU.")

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_DIR = "./nsfw_image_detection"

# Device setup
device = "cuda" if gpu_available else "cpu"

# Load model and processor
logger.info("Loading model and processor...")
try:
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(device)
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    logger.info("Model and processor loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load model or processor: {e}", exc_info=True)
    raise RuntimeError("Model or processor loading failed.")

# Helper function: Image classification
def classify_image(image: Image.Image):
    """Classify the image as NSFW or Safe."""
    try:
        logger.info("Preparing image for inference...")
        inputs = processor(images=image, return_tensors="pt").to(device)

        logger.info("Performing inference...")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get predictions
        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        confidence = torch.nn.functional.softmax(logits, dim=-1).max().item()
        logger.info(f"Inference complete: Label={label}, Confidence={confidence:.2f}")
        return label, confidence
    except Exception as e:
        logger.error(f"Error during classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error during classification.")

# Helper function: Validate image file
def validate_image(file: UploadFile):
    """Validate uploaded image file type."""
    logger.info(f"Validating file: {file.filename}...")
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning(f"Unsupported file format: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPEG and PNG are supported.")
    logger.info(f"File {file.filename} validated successfully.")

# Root endpoint
@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "NSFW Detection API is running."}

# Image classification endpoint
@app.post("/classify_image/")
async def classify_image_endpoint(file: UploadFile = File(...)):
    """Endpoint to classify NSFW content."""
    try:
        # Validate image
        validate_image(file)

        # Read and process image
        logger.info("Reading uploaded file...")
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        logger.info(f"Image {file.filename} loaded successfully.")

        # Perform classification
        label, confidence = classify_image(image)

        # Construct response
        logger.info("Constructing response...")
        response = {
            "NSFW Content": label == "nsfw",
            "Confidence Percentage": round(confidence * 100, 2)
        }
        logger.info("Response constructed successfully.")
        return JSONResponse(content=response, status_code=200)

    except HTTPException as e:
        logger.warning(f"HTTPException: {e.detail}")
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

# 404 error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    logger.warning(f"HTTPException at {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

# Catch-all error handler
@app.exception_handler(Exception)
async def exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception at {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."}
    )
