from flask import Flask, request, jsonify
import os
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from ultralytics import YOLO
from flask_cors import CORS

import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
CROPPED_FOLDER = 'static/cropped_faces'
RESULTS_FOLDER = 'static/results'
CORS(app)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load YOLO model for face detection
yolo_model = YOLO("./face-detection-yolov8/yolov8n-face.pt")

# Siamese Neural Network definition
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # Pass in both images and obtain both vectors
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Load the Siamese model
model = SiameseNetwork()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Image transformation for the Siamese network
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# Load reference images for comparison
import torchvision.datasets as datasets
folder_dataset = datasets.ImageFolder(root="./Training/Training")
dataset_imgs = folder_dataset.imgs  # list of (filepath, class)
@app.route("/detect_and_ideantify", methods=["GET", "POST"])
def detect_and_identify():

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    # Save the uploaded group photo
    input_filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_filepath)
    
    # Load the image for processing
    img = cv2.imread(input_filepath)
    if img is None:
        return jsonify({"error": "Could not read the uploaded image"}), 400
    
    # Original image dimensions for reference
    original_img = img.copy()
    img_height, img_width = img.shape[:2]
    
    # Run face detection with YOLO
    results = yolo_model(input_filepath, conf=0.5)
    
    # Check if any faces were detected
    if len(results) == 0 or not hasattr(results[0], 'boxes') or len(results[0].boxes) == 0:
        return jsonify({"error": "No faces detected in the image"}), 400
    
    # Get bounding boxes
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    
    # Sort boxes by x-coordinate (left to right)
    boxes_with_conf = [(box, conf) for box, conf in zip(boxes, confidences)]
    boxes_with_conf.sort(key=lambda x: x[0][0])  # Sort by x1 coordinate
    
    # Create a result image with annotations
    result_img = original_img.copy()
    
    # Process each detected face
    identified_faces = []
    
    for i, (box, conf) in enumerate(boxes_with_conf):
        x1, y1, x2, y2 = map(int, box)
        
        # Ensure box coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        # Crop the face from the image
        face_img = img[y1:y2, x1:x2]
        
        # Save the cropped face
        cropped_path = os.path.join(CROPPED_FOLDER, f"face_{i+1}.jpg")
        cv2.imwrite(cropped_path, face_img)
        
        # Convert to grayscale for Siamese network
        pil_face = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).convert("L")
        transformed_face = transform(pil_face).unsqueeze(0)
        
        # Compare with dataset using Siamese model
        scores = []
        for img_path, class_id in dataset_imgs:
            # Load reference image
            ref_img = Image.open(img_path).convert("L")
            ref_img_tensor = transform(ref_img).unsqueeze(0)
            
            # Forward pass through the model
            with torch.no_grad():
                output1, output2 = model(transformed_face, ref_img_tensor)
                distance = F.pairwise_distance(output1, output2).item()
                scores.append((distance, img_path, class_id))
        
        # Find the best match
        best_match = min(scores, key=lambda x: x[0])
        best_score, best_img_path, class_id = best_match
        
        # Determine if the face is "Unknown" based on threshold
        threshold = 0.5  # Updated threshold for dissimilarity
        person_name = os.path.basename(os.path.dirname(best_img_path)) if best_score < threshold else "Unknown"
        
        # Draw rectangle and label on the result image
        color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        # Add name and confidence score
        label = f"{person_name} ({best_score:.2f})"
        cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add to results
        identified_faces.append({
            "face_id": i + 1,
            "cropped_face": cropped_path,
            "person_name": person_name,
            "dissimilarity_score": float(best_score),
            "confidence": float(conf),

        })
      
    
    result_path = os.path.join(RESULTS_FOLDER, f"result_{os.path.basename(input_filepath)}")
    cv2.imwrite(result_path, result_img)
    
    # Return the results as an HTML page
 
  
    return jsonify({
        "original_image": input_filepath,
        "detected_faces": identified_faces
    })
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)