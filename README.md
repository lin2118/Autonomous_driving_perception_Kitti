# Autonomous Perception Pipeline using YOLOv8 and KITTI Dataset

## Overview

This project demonstrates a complete computer vision pipeline for 2D object detection in autonomous driving scenarios using the YOLOv8 architecture and the KITTI Vision Benchmark Suite dataset.

The objective was to build a custom object detection model capable of identifying road users and vehicles from front-facing camera images, starting from raw KITTI annotations through dataset preparation, training, validation, and inference.

The project was developed and trained on a Linux-based workstation using GPU acceleration and serves as a foundation for future extensions toward multi-object tracking, sensor fusion, and 3D object detection.

---

## Model and Dataset

### Object Detection Model

* YOLOv8s (You Only Look Once Version 8 - Small)
* Pretrained weights: `yolov8s.pt`
* Framework: Ultralytics YOLO
* Task: 2D Object Detection

### Dataset

* KITTI Object Detection Benchmark
* Total annotated images: 7,481
* Image source: Front-left RGB camera (`image_2`)
* Annotation format: KITTI native label format

### Object Classes

The model was trained to detect the following 9 classes:

1. Car
2. Pedestrian
3. Cyclist
4. Van
5. Truck
6. Tram
7. Person_sitting
8. Misc
9. DontCare

---

## Dataset Preparation

One of the key challenges of this project was converting the original KITTI annotation format into the format required by YOLO.

The raw KITTI labels contain object information in the following structure:

```text
Class Truncated Occluded Alpha x1 y1 x2 y2 ...
```

Example:

```text
Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 ...
```

The bounding box coordinates:

```text
x1 = 712.40
y1 = 143.00
x2 = 810.73
y2 = 307.92
```

were extracted and converted into YOLO format:

```text
class_id x_center y_center width height
```

normalized to image dimensions.

Custom Python scripts were developed to:

* Parse KITTI label files
* Extract bounding box information
* Convert annotations into YOLO format
* Maintain image-label correspondence
* Generate train and validation splits

---

## Train / Validation Split

The dataset was randomly split while preserving image-label pairing.

| Split      | Images |
| ---------- | ------ |
| Training   | 5,984  |
| Validation | 1,497  |
| Total      | 7,481  |

A significant portion of the implementation effort was dedicated to ensuring that every image remained correctly associated with its corresponding annotation file after the conversion and split process.

---

## Training Environment

### Hardware

* NVIDIA Quadro RTX 8000
* 48 GB VRAM
* CUDA Accelerated Training

### Software

* Ubuntu Linux
* Python 3.10
* PyTorch 2.12
* CUDA 13
* Ultralytics YOLOv8

The training pipeline successfully utilized GPU acceleration throughout the training process.

Example training output:

```text
CUDA:0 (Quadro RTX 8000, 48380MiB)
```

---

## Training Hyperparameters

| Parameter  | Value    |
| ---------- | -------- |
| Model      | YOLOv8s  |
| Epochs     | 150      |
| Batch Size | 16       |
| Image Size | 640      |
| Optimizer  | AdamW    |
| Device     | CUDA GPU |
| Classes    | 9        |

Training duration:

```text
150 epochs completed in 2.323 hours
```

---

## Results

### Overall Performance

| Metric    | Score |
| --------- | ----- |
| Precision | 0.870 |
| Recall    | 0.844 |
| mAP@50    | 0.871 |
| mAP@50-95 | 0.664 |

### Class-wise Highlights

| Class      | mAP@50 |
| ---------- | ------ |
| Car        | 0.982  |
| Van        | 0.974  |
| Truck      | 0.971  |
| Tram       | 0.950  |
| Cyclist    | 0.930  |
| Pedestrian | 0.860  |

These results demonstrate strong detection performance across multiple road-user categories while maintaining real-time inference capability.

### Inference Performance

```text
Average Inference Time: ~5 ms per image
```

This corresponds to approximately:

```text
~200 FPS
```

on the NVIDIA Quadro RTX 8000 GPU.

---

## Inference Demonstration

After training, the best-performing model (`best.pt`) was evaluated on an unseen driving sequence from the KITTI Raw Dataset.

### Evaluation Scene

* KITTI Raw Dataset
* Drive Sequence: `2011_09_26_drive_0106_sync`
* Camera Stream: `image_02`
* Left RGB Camera

The inference results were exported as a video and can be found in:

```text
kitti_drive_0106_yolo_demo.mp4
```

This demonstration showcases the model detecting vehicles, pedestrians, cyclists, and other road users in a realistic urban driving environment.

---


## Future Work

Potential extensions of this project include:

* Multi-object tracking
* Object tracking using ByteTrack / BoT-SORT
* Depth estimation using stereo cameras
* 3D object detection
* LiDAR-camera sensor fusion
* Integration into autonomous driving stacks using ROS 2
* Real-time deployment on edge computing platforms

---

## Key Learning Outcomes

* End-to-end object detection pipeline development
* Dataset preprocessing and annotation conversion
* GPU-accelerated deep learning workflows
* YOLOv8 training and hyperparameter tuning
* Autonomous driving perception fundamentals
* Validation and inference on unseen driving sequences
* Dataset management and experiment reproducibility
