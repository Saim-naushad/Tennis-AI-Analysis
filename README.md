# Tennis-AI-Analysis

This repository implements a deep learning-based solution for analyzing tennis matches through computer vision. The focus is on detecting tennis court keypoints, tracking players, and detecting ball movement. The results are visualized on a miniaturized court for enhanced analysis.

## Features

- **Court Keypoint Detection**: Detects and maps 14 key points on the tennis court.
- **Player Tracking**: Tracks player movements across frames.
- **Ball Tracking**: Detects and tracks the tennis ball.
- **Mini Court Mapping**: Visualizes player and ball positions on a scaled-down tennis court.
- **Post-Processing**: Refines ball and keypoint predictions using computer vision techniques.

## Model Training

 **Keypoint Detection**

This model predicts 14 keypoints representing the tennis court's main features. It’s based on a ResNet-50 architecture, with the last layer adjusted to output 28 values (x, y coordinates for each of the 14 keypoints). Training involves a dataset of tennis court images with annotated keypoints.
To train the keypoint detection model, follow the steps in the keypoint_training.ipynb notebook. It walks you through dataset preparation, model training, and saving the weights.
