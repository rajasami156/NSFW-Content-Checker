import os
import logging
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("uvicorn")

# CORS configuration (allowing requests from any origin in this example)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, adjust as per your use case
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Path to model directory
model_directory = './nsfw_image_detection'

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load the model and processor once, to optimize performance
model = AutoModelForImageClassification.from_pretrained(model_directory).to(device)
processor = ViTImageProcessor.from_pretrained(model_directory)

# Log the model device configuration
logger.info(f"Model is on device: {model.device}")

# Helper function to predict NSFW content
def predict_nsfw(image: Image.Image):
    try:
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Perform the classification
        logger.info(f"Running inference on device: {device}")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted label
        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]

        # Confidence score (optional)
        confidence = torch.nn.functional.softmax(logits, dim=-1).max().item()

        return label, confidence
    except Exception as e:
        logger.error(f"Error in NSFW prediction: {e}")
        raise HTTPException(status_code=500, detail="Error in NSFW prediction")

# Helper function to validate the file (no size validation here as it's handled by Nginx)
def validate_image(file: UploadFile):
    # Check file type (only accept JPEG/PNG)
    if file.content_type not in ['image/jpeg', 'image/png']:
        logger.warning(f"Invalid file type received: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed.")
    
    logger.info(f"Received file: {file.filename} of type {file.content_type}")

# POST endpoint to classify the image
@app.post("/classify_image/")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Validate the uploaded image file
        validate_image(file)

        # Read the image data and process it
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))

        # Log the processing start
        logger.info(f"Processing image: {file.filename}")

        # Predict NSFW content
        label, confidence = predict_nsfw(image)

        # Log the result
        logger.info(f"Prediction for image {file.filename}: {label}, Confidence: {confidence * 100:.2f}%")

        # Return the response based on classification
        if label == 'nsfw':
            return JSONResponse(content={
                "NSFW Content": True,
                "NSFW Content Percentage": round(confidence * 100, 2)
            })
        else:
            return JSONResponse(content={
                "NSFW Content": False,
                "NSFW Content Percentage": 0
            })

    except HTTPException as e:
        # Log known exceptions (like file validation errors)
        logger.warning(f"HTTPException: {e.detail}")
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Error handler for 404 (Resource not found)
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    logger.warning(f"HTTPException occurred: {exc.detail} at {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

# Catch-all error handler for unexpected errors
@app.exception_handler(Exception)
async def exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception occurred: {str(exc)} at {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."},
    )
