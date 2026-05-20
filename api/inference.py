import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import sys
import os

# Ensure we can import model from kaggle_training
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kaggle_training')))
from model import SiameseUNet

try:
    from skimage.exposure import match_histograms
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

class GeoChangePredictor:
    def __init__(self, model_path='../siamese_unet_weights.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseUNet().to(self.device)
        
        # Load weights if they exist
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded weights from {model_path}")
        else:
            print(f"Warning: {model_path} not found. Running with untrained weights for demonstration.")
            
        self.model.eval()
        
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, img1_path, img2_path, apply_histogram_matching=True):
        imgA = Image.open(img1_path).convert('RGB')
        imgB = Image.open(img2_path).convert('RGB')
        
        # Save original size to resize the mask back
        orig_size = imgA.size # (width, height)
        
        # --- Domain Shift Mitigation: Histogram Matching ---
        # Forces Image 2 (Google Maps) to adopt the exact color, brightness, 
        # and contrast of Image 1 (Esri Baseline) to trick the neural network.
        if apply_histogram_matching and HAS_SKIMAGE:
            npA = np.array(imgA)
            npB = np.array(imgB)
            # Match imgB to imgA using the last axis as the color channel
            matched_B = match_histograms(npB, npA, channel_axis=-1)
            imgB = Image.fromarray(matched_B.astype(np.uint8))
        # --------------------------------------------------
        
        tA = self.transform(imgA).unsqueeze(0).to(self.device)
        tB = self.transform(imgB).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tA, tB)
            # Apply sigmoid to get probabilities
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
        # Binarize
        mask = (prob > 0.5).astype(np.uint8) * 255
        
        # Resize mask back to original image size
        mask_img = Image.fromarray(mask).resize(orig_size, Image.NEAREST)
        mask_np = np.array(mask_img)
        
        # Calculate Analytics
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out extremely small noise (e.g., less than 10 pixels)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
        num_changes = len(valid_contours)
        
        # --- Advanced Area Estimation ---
        # Since the Siamese U-Net might only detect *parts* of a changed building, 
        # calculating the area of a "Convex Hull" (a polygon wrapped tightly around the 
        # detected parts) gives a much more accurate estimate of the total building footprint.
        changed_area_pixels = 0
        for contour in valid_contours:
            hull = cv2.convexHull(contour)
            changed_area_pixels += int(cv2.contourArea(hull))
        
        # Generate Neon Pink Outlines
        overlay_img = self._generate_outline_overlay(mask_np, orig_size, valid_contours)
        
        return {
            "overlay_image": overlay_img,
            "analytics": {
                "distinct_changes": num_changes,
                "changed_area_pixels": changed_area_pixels
            }
        }

    def _generate_outline_overlay(self, mask_np, size, contours):
        # mask_np is 255 for change, 0 for background
        # Create an empty transparent RGBA image
        overlay = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        
        # Draw neon pink outlines (R=255, G=20, B=147, A=255)
        neon_pink = (255, 20, 147, 255)
        # Draw slightly thicker line for glow effect
        cv2.drawContours(overlay, contours, -1, neon_pink, thickness=2)
        
        # Add a semi-transparent fill so it's easier to see the whole area
        fill_pink = (255, 20, 147, 80)
        cv2.drawContours(overlay, contours, -1, fill_pink, thickness=-1)
        
        return Image.fromarray(overlay, 'RGBA')
