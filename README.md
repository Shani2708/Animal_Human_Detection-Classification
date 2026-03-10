## Setup Instructions

To set up and test the project, follow the steps below.

### 1. Clone the Project from GitHub

```bash
git clone https://github.com/<your-username>/Human_Animal_Classification.git

cd Human_Animal_Classification
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**On Windows**

```bash
venv\Scripts\activate
```

**On macOS / Linux**

```bash
source venv/bin/activate
```

### 4. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 5. Ensure Models Are Available

Make sure the trained models are placed in the `models/` directory.

```
models/
├── detection.pt                   # YOLO detection model
└── best_classification_model.pth  # ResNet classification model
```

### 6. Run the Inference Script

```bash
python inference.py
```

### 7. Add a Test Video

Place any video file inside the `test_videos/` folder.

```
test_videos/
    sample_video.mp4
```

The system will automatically detect and process the video and save the annotated result in the `outputs/` folder.

```
outputs/
    output_sample_video.mp4
```

The inference script monitors the `test_videos/` directory and processes new videos automatically.