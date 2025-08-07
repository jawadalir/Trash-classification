from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
import os
from contextlib import asynccontextmanager
import numpy as np
import cv2
import base64

# Initialize model as None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not load_model():
        print("Warning: Model could not be loaded. The application will not be able to make predictions.")
    yield
    # Shutdown
    if model is not None:
        del model

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Define class labels
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def process_image(image):
    # Convert PIL Image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Edge Detection (Canny)
    edges = cv2.Canny(cv_image, 100, 200)
    
    # Contour Detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # Color Analysis
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(color_hist, color_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    color_vis = np.zeros((100, 256, 3), dtype=np.uint8)
    for i in range(180):
        for j in range(256):
            color_vis[:, j] = [i, 255, 255]
    color_vis = cv2.cvtColor(color_vis, cv2.COLOR_HSV2BGR)
    
    # Convert processed images to base64 for sending to frontend
    def image_to_base64(img):
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')
    
    return {
        'original': image_to_base64(cv_image),
        'edges': image_to_base64(edges),
        'contours': image_to_base64(contour_image),
        'color_analysis': image_to_base64(color_vis)
    }

def load_model():
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Not a dict")
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            from torchvision.models import resnet50
            model_instance = resnet50(weights=None)
            model_instance.fc = torch.nn.Linear(model_instance.fc.in_features, len(CLASSES))
            model_instance.load_state_dict(checkpoint['model_state_dict'])
            model = model_instance
        else:
            model = checkpoint
            
        model.eval()
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image for computer vision tasks
        processed_images = process_image(image)
        
        # Prepare image for model prediction
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get top 3 predictions
            top3_prob, top3_catid = torch.topk(probabilities, 3)
            
            predictions = []
            for i in range(3):
                predictions.append({
                    "class": CLASSES[top3_catid[i].item()],
                    "confidence": f"{top3_prob[i].item():.2%}"
                })
        
        return JSONResponse({
            "predictions": predictions,
            "processed_images": processed_images
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import socket
    
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    port = find_free_port()
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port) 