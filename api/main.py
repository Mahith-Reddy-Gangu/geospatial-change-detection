from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from inference import GeoChangePredictor
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile
import os
import io
import base64
import math
import urllib.request

app = FastAPI(title="GeoChange API")

# --- Esri Public Export Fetcher ---
def get_esri_export(lat, lon):
    # Create ~500m bounding box
    delta_lat = 0.00225
    delta_lon = 0.00225 / math.cos(math.radians(lat))
    xmin = lon - delta_lon
    xmax = lon + delta_lon
    ymin = lat - delta_lat
    ymax = lat + delta_lat
    
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={xmin},{ymin},{xmax},{ymax}&bboxSR=4326&imageSR=4326&size=512,512&format=png&f=image"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        return response.read()
# --------------------------------

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
# Look for weights one directory up if needed, or in current dir
predictor = GeoChangePredictor(model_path='../siamese_unet_weights.pth')

@app.post("/infer")
async def infer(
    img2: UploadFile = File(...),
    img1: UploadFile = File(None),
    lat: float = Form(None),
    lon: float = Form(None)
):
    # Save uploaded files temporarily
    path1 = None
    
    if lat is not None and lon is not None:
        # User provided coordinates! Fetch from Esri Export.
        tile_bytes = get_esri_export(lat, lon)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1:
            f1.write(tile_bytes)
            path1 = f1.name
    elif img1 is not None:
        # User uploaded Image 1 manually (or pressed Demo button)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1:
            f1.write(await img1.read())
            path1 = f1.name
    else:
        return {"error": "Must provide either img1 OR (lat and lon)"}
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
        f2.write(await img2.read())
        path2 = f2.name
        
    try:
        # Generate overlay image (RGBA PNG) and analytics
        prediction_result = predictor.predict(path1, path2)
        overlay_img = prediction_result["overlay_image"]
        analytics = prediction_result["analytics"]
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        overlay_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Convert to base64
        base64_encoded = base64.b64encode(img_byte_arr).decode('utf-8')
        
        return {
            "image_base64": base64_encoded,
            "analytics": analytics
        }
        
    finally:
        # Clean up temp files
        os.remove(path1)
        os.remove(path2)

# Mount the frontend directory so it serves the HTML and sample images locally
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
