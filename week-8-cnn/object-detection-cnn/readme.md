# Object Detection + Transfer Learning | [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640)

- **YOLO Overview:**
  - YOLO is a single-shot object detection algorithm that processes an entire image in one pass, making it computationally efficient and suitable for real-time applications.
  - It frames object detection as a regression problem, predicting bounding boxes and class probabilities simultaneously.
  - It uses a single convolutional network to process the entire image, making it extremely fast and efficient.

- **Key Features:**
  1. **Speed:** Processes images in real-time at 45 frames per second (Fast YOLO achieves 155 fps).
  2. **Unified Architecture:** Combines feature extraction, bounding box prediction, and classification into a single network.
  3. **Generalization:** Performs well on new domains and unexpected inputs, making it robust for real-world applications.

- **Performance Metrics:**
  - **Intersection over Union (IoU):** Measures localization accuracy by calculating the overlap between predicted and ground truth bounding boxes.
  - **Average Precision (AP):** Area under the precision-recall curve, with mean Average Precision (mAP) used for overall performance evaluation.

- **Model Architecture:**
  - 24 convolutional layers followed by 2 fully connected layers.
  - Uses 1x1 reduction layers and 3x3 convolutional layers.
  - Final output is a 7x7x30 tensor of predictions.
  - Divides the input image into an S × S grid, where each grid cell predicts bounding boxes and confidence scores.
  - Uses anchor boxes to handle objects of varying sizes and aspect ratios (introduced in YOLO v2).
  - Feature Pyramid Networks (FPN) in YOLO v3 and later versions improve detection of small objects.

- **Training Details:**
  - Pretrained on ImageNet for feature extraction.
  - Fine-tuned for object detection with additional convolutional and fully connected layers.
  - Loss function includes localization, confidence, and classification errors, with improvements like GHM loss (YOLO v4) and CIoU loss (YOLO v5).

- **Limitations:**
  - Struggles with small objects in groups.
  - Spatial constraints(lighting or env cond) limit the number of nearby objects that can be detected.
  - Coarse features due to multiple downsampling layers.
  - Computationally intensive for resource-constrained devices.

- **Performance:**
  - Achieves 63.4 mAP on PASCAL VOC 2007 with a speed of 45 fps.
  - Outperforms traditional methods like DPM and R-CNN in speed and accuracy.

- **Applications:**
  - Real-time object detection for autonomous driving, surveillance, and robotics.
  - General-purpose object detection across various domains, including artwork and natural images.

- **Evolution of YOLO Versions:**
  - **YOLO v1:** Introduced in 2016, focused on speed and simplicity.
  - **YOLO v2 (YOLO9000):** Added anchor boxes, batch normalization, and multi-scale training.
  - **YOLO v3:** Introduced Darknet-53 and feature pyramid networks for multi-scale detection.
  - **YOLO v4:** Improved with CSPNet, k-means clustering for anchor boxes, and GHM loss.
  - **YOLO v5:** Added dynamic anchor boxes, spatial pyramid pooling, and CIoU loss.
  - **YOLO v6:** Introduced EfficientNet-L2 architecture and dense anchor boxes.
  - **YOLO v7:** Enhanced with focal loss, higher resolution, and improved layer aggregation for better feature learning.
  - **YOLO v8:** Latest version with a new API for easier training and inference, supporting previous YOLO versions.

### Key Observation

YOLO is a state-of-the-art object detection algorithm that uses a Convolutional Neural Network (CNN) to detect objects in an image. It works by:

1. **Object Detection with CNN**: YOLO processes the entire image in a single pass through the CNN, dividing it into a grid. Each grid cell predicts bounding boxes and class probabilities for objects within its area.

2. **Bounding Boxes**: The algorithm predicts multiple bounding boxes for each grid cell. Each box is defined by its center coordinates, width, height, and a confidence score (probability of containing an object).

3. **Improving Classification**:
   - **Thresholding**: Boxes with low confidence scores are filtered out.
   - **Non-Max Suppression (NMS)**: When multiple boxes overlap and detect the same object, NMS keeps only the box with the highest confidence score and removes the rest.

This process ensures that YOLO outputs only the most accurate bounding boxes and class predictions for the objects in the image.

[Implementation](https://nbviewer.org/github/amanchadha/coursera-deep-learning-specialization/blob/master/C4%20-%20Convolutional%20Neural%20Networks/Week%203/Car%20detection%20for%20Autonomous%20Driving/Autonomous_driving_application_Car_detection.ipynb)