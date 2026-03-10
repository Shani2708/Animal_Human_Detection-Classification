"""
===========================================================
ANIMAL AND HUMAN DETECTION SYSTEM
===========================================================

This project implements a two-stage computer vision pipeline:

1️⃣ Object Detection Model
   Detects objects in the frame and produces bounding boxes.

2️⃣ Classification Model
   Classifies each detected object as:
        - Human
        - Animal

The pipeline processes videos automatically from the folder:

        ./test_videos/

and outputs annotated videos to:

        ./outputs/

"""

# ==========================================================
# 1. DATASET SELECTION AND PREPROCESSING
# ==========================================================

"""
Dataset Source
----------------------------------------

The dataset used for this project was an open-source dataset
downloaded from the internet that was already formatted in
YOLO object detection format.

The dataset originally contained:

    • ~9000 images
    • 58 object classes
    • YOLO formatted annotations
    • Separate train / validation splits

Each image had a corresponding label file containing
bounding box annotations in the YOLO format:

    class_id  x_center  y_center  width  height


Why This Dataset Was Chosen
----------------------------------------

This dataset was selected because:

1️⃣ YOLO Format Availability

The dataset was already provided in YOLO annotation format,
which made it directly compatible with YOLO-based detection
models without requiring complex annotation conversion.

2️⃣ Bounding Box Annotations

The dataset contained high-quality bounding box annotations
necessary for training an object detection model.

3️⃣ Multiple Human and Animal Classes

The dataset contained several human-related and animal-related
categories, allowing us to construct a custom dataset
focused specifically on the task of detecting humans
and animals.

4️⃣ Dataset Size

With approximately 9000 images, the dataset was large enough
to train a robust detection model while remaining manageable
for experimentation and training within limited computational
resources.

5️⃣ Real-world Diversity

Images in the dataset contain objects in various environments,
lighting conditions, and viewpoints, helping the model learn
generalizable visual features.


Dataset Filtering for the Task
----------------------------------------

The original dataset contained 58 classes including many
irrelevant objects such as vehicles, furniture, tools,
and other everyday objects.

However, the goal of this project is to detect only:

        • Humans
        • Animals

Therefore, the dataset was filtered to retain only images
containing relevant classes.


Human Classes Selected
----------------------------------------

The following classes were mapped to the HUMAN category:

    Pedestrians
    Persona
    Pessoa
    people
    person
    persons

These classes represent humans under different labels
present in the dataset.


Animal Classes Selected
----------------------------------------

The following classes were mapped to the ANIMAL category:

    bird
    cat
    cow
    dog
    fox
    goat
    horse
    racoon
    sheep


Class Remapping Strategy
----------------------------------------

After identifying the relevant classes, all selected
classes were mapped into two final categories:

        0 → Human
        1 → Animal


For example:

Original Class → New Class

    person      → 0 (Human)
    people      → 0 (Human)
    dog         → 1 (Animal)
    cat         → 1 (Animal)
    horse       → 1 (Animal)

All other classes were ignored.


Dataset Filtering Process
----------------------------------------

The dataset preprocessing pipeline involved the following steps:

Step 1
Load the dataset configuration file (data.yaml)
to obtain the list of all 58 class names.

Step 2
Identify class IDs corresponding to human-related
and animal-related classes.

Step 3
Iterate through all label files in the dataset.

Step 4
For each annotation in a label file:

    • If the class belongs to the human class list
      → change label to class 0

    • If the class belongs to the animal class list
      → change label to class 1

    • If the class belongs to any other category
      → ignore the annotation

Step 5
If an image contains at least one valid human
or animal annotation, copy the image and
filtered label file to the new dataset.


Final Detection Dataset
----------------------------------------

After filtering, the dataset was reduced to
only two classes:

        0 → Human
        1 → Animal


Final dataset distribution used for training:

    Training Images      ≈ 4200
    Validation Images    ≈ 1200


Final Dataset Structure
----------------------------------------

dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
└── valid/
    ├── images/
    └── labels/


Each image contains YOLO bounding box annotations
for either human or animal objects.

This filtered dataset was used to train the
YOLOv8 detection model.
"""


# ==========================================================
# 2. OBJECT DETECTION MODEL
# ==========================================================

"""
Model Used:
----------------------------------------
YOLOv8 (Ultralytics)

Model Variant:
        YOLOv8n (nano version)

Reason for choosing YOLOv8:

1️⃣ State-of-the-art real-time detection
   YOLOv8 provides excellent performance for object detection.

2️⃣ Fast inference
   Suitable for real-time video processing.

3️⃣ Easy fine-tuning
   Ultralytics provides simple training pipelines.

4️⃣ Lightweight models available
   YOLOv8n is small and efficient while still accurate.

5️⃣ Strong community support
   Widely used in industry applications.

Detection Model Task:
----------------------------------------

Input:
        Full image or video frame

Output:
        Bounding boxes around objects

Example:

Frame
 ↓
YOLO Detection
 ↓
Bounding Boxes
    Person
    Dog
    Cat
    etc


Training Details:
----------------------------------------

Pretrained weights:
        yolov8n.pt

Image size:
        640

Epochs:
        80

Optimizer:
        Default YOLO optimizer

Device:
        GPU


Metrics Logged:
----------------------------------------

Training Loss
Validation Loss
Mean Average Precision (mAP)
Precision
Recall


Logging Tool:
----------------------------------------

Weights & Biases (wandb)

Reason for using wandb:

1️⃣ Easy experiment tracking
2️⃣ Visualization of metrics
3️⃣ Compare training runs
4️⃣ Industry standard monitoring tool
"""


# ==========================================================
# 3. CLASSIFICATION MODEL
# ==========================================================

"""
Model Used:
----------------------------------------
ResNet50

Framework:
        PyTorch


Reason for choosing ResNet50:

1️⃣ Proven CNN architecture
   ResNet models are widely used for image classification.

2️⃣ Residual connections
   Helps train deeper networks effectively.

3️⃣ Pretrained on ImageNet
   Allows transfer learning.

4️⃣ Good balance between speed and accuracy.

5️⃣ Strong feature extraction capability.


Classification Task:
----------------------------------------

After detection, each detected object
is cropped from the image and passed to
the classification model.

The classifier predicts:

        Human
        Animal


Example pipeline:

Frame
 ↓
YOLO Detection
 ↓
Crop bounding box
 ↓
ResNet50 classifier
 ↓
Human / Animal


Dataset Creation for Classification:
----------------------------------------

The detection dataset contains full images.

To train the classifier:

1️⃣ Each bounding box is cropped.

2️⃣ The cropped object is saved as an image.

3️⃣ Images are placed into folders:

classification_dataset/
│
├── train/
│   ├── human/
│   └── animal/
│
└── val/
    ├── human/
    └── animal/


Dataset Size:

Train:
        2000 human images
        2000 animal images

Validation:
        200 human images
        200 animal images


Image Preprocessing:
----------------------------------------

Resize:
        224 x 224

Normalization:
        ImageNet mean/std

Augmentations (optional):
        Flip
        Rotation
        Color jitter


Training Details:
----------------------------------------

Batch Size:
        32

Optimizer:
        Adam

Learning Rate:
        0.0001

Loss Function:
        CrossEntropyLoss

Epochs:
        15


Metrics Logged:
----------------------------------------

Training Loss
Validation Loss
Training Accuracy
Validation Accuracy

All metrics logged using wandb.
"""


# ==========================================================
# 4. MODEL SAVING
# ==========================================================

"""
After training completes:

Detection Model Saved As:

models/
    detection.pt


Classification Model Saved As:

models/
    best_classification_model.pth


Saving best model based on:

        Highest validation accuracy
        Lowest validation loss
"""


# ==========================================================
# 5. INFERENCE PIPELINE
# ==========================================================

"""
The inference pipeline automatically processes
videos uploaded by the user.

Input Folder:
----------------------------------------

./test_videos/

User only needs to place a video inside this folder.


Output Folder:
----------------------------------------

./outputs/

The system generates:

        Annotated video


Inference Steps:
----------------------------------------

Step 1
Monitor the directory:

        ./test_videos/


Step 2
When a new video is detected:

Load the video using OpenCV.


Step 3
Process the video frame-by-frame.


Step 4
Run YOLO detection on each frame.

Output:
        Bounding boxes


Step 5
For each detected bounding box:

Crop the object region.


Step 6
Resize cropped image to:

        224 x 224


Step 7
Pass cropped image to classification model.


Step 8
Classifier predicts:

        Human
        Animal


Step 9
Draw bounding box on frame.

Example label:

        Human
        Animal


Step 10
Save annotated frame.


Step 11
Combine frames and save output video.

Saved to:

        ./outputs/


Example Output:

Original Video
 ↓
Detection + Classification
 ↓
Annotated Video


Example annotation:

[Human]
[Animal]
"""


# ==========================================================
# 6. CHALLENGES FACED
# ==========================================================

"""
1️⃣ Dataset Noise

Open Images contains many classes.
Filtering and mapping classes to only
human and animal required careful preprocessing.


2️⃣ Class Imbalance

Some animal categories had fewer samples.
Balancing the dataset was necessary.


3️⃣ Bounding Box Quality

Some bounding boxes were small or inaccurate.
Very small crops were removed during
classification dataset generation.


4️⃣ Two-Stage Pipeline Complexity

Detection and classification models must
work together correctly.

Proper cropping and resizing was required.


5️⃣ Training Time

Training detection models on large datasets
requires significant GPU resources.
"""


# ==========================================================
# 7. POSSIBLE IMPROVEMENTS
# ==========================================================

"""
1️⃣ Use Larger Detection Models

YOLOv8m or YOLOv8l could improve accuracy.


2️⃣ Multi-class Animal Detection

Instead of a single "animal" class,
different animals could be classified separately.


3️⃣ End-to-End Architecture

Use a single model capable of detecting
and classifying objects simultaneously.


4️⃣ Data Augmentation

Advanced augmentations such as:

MixUp
Mosaic
CutMix


5️⃣ Real-Time Deployment

The system could be deployed as:

- Web application
- Surveillance monitoring tool
- Wildlife monitoring system


6️⃣ Object Tracking

Object tracking could be integrated to maintain the
identity of detected objects across multiple frames.

Algorithms such as DeepSORT or ByteTrack could be used
to assign a unique tracking ID to each detected object
and follow its movement across frames.

This would improve video stability, reduce flickering
detections, and enable advanced features such as
object counting and movement analysis.
"""
