import os
import time
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# Configuration
# -----------------------------

TEST_VIDEO_DIR = "./test_videos"
OUTPUT_DIR = "./outputs"

DETECTION_MODEL_PATH = "./models/detection.pt"
CLASSIFICATION_MODEL_PATH = "./models/best_classification_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# classes for classification model
CLASSES = ["animal", "human"]

# -----------------------------
# Load Detection Model
# -----------------------------

detection_model = YOLO(DETECTION_MODEL_PATH)

# -----------------------------
# Load Classification Model
# -----------------------------

from torchvision import models
import torch.nn as nn

classification_model = models.resnet50(pretrained=False)
classification_model.fc = nn.Linear(classification_model.fc.in_features, 2)

classification_model.load_state_dict(
    torch.load(CLASSIFICATION_MODEL_PATH, map_location=DEVICE)
)

classification_model.to(DEVICE)
classification_model.eval()

# -----------------------------
# Image Transform for Classifier
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Classification Function
# -----------------------------

def classify_crop(crop):

    # Convert BGR to RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(crop)
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = classification_model(image)
        _, predicted = torch.max(outputs, 1)

    return CLASSES[predicted.item()]

# -----------------------------
# Video Processing Function
# -----------------------------

def process_video(video_path):

    video_name = os.path.basename(video_path)
    output_path = os.path.join(OUTPUT_DIR, f"output_{video_name}")

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {video_name}...")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # -----------------------------
        # Object Detection
        # -----------------------------

        results = detection_model(frame)[0]

        for box in results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # crop detected object
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # -----------------------------
            # Classification
            # -----------------------------

            label = classify_crop(crop)

            # draw bounding box
            color = (0,255,0) if label == "human" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        out.write(frame)

    cap.release()
    out.release()

    print(f"Saved output → {output_path}")


# -----------------------------
# Directory Monitoring
# -----------------------------

def monitor_directory():

    print("Watching for new videos in ./test_videos ...")

    processed = set()

    while True:

        videos = os.listdir(TEST_VIDEO_DIR)

        for video in videos:

            video_path = os.path.join(TEST_VIDEO_DIR, video)

            if video_path not in processed:

                process_video(video_path)
                processed.add(video_path)

        time.sleep(5)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    monitor_directory()