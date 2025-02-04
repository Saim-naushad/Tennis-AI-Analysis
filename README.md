# Tennis-AI-Analysis

This repository implements a deep learning-based solution for analyzing tennis matches through computer vision. The focus is on detecting tennis court keypoints, tracking players, and detecting ball movement. The results are visualized on a miniaturized court for enhanced analysis.

## ✨ Features

- **Court Keypoint Detection**: Detects and maps 14 key points on the tennis court.
- **Player Tracking**: Tracks player movements across frames.
- **Ball Tracking**: Detects and tracks the tennis ball.
- **Mini Court Mapping**: Visualizes player and ball positions on a scaled-down tennis court.
- **Post-Processing**: Refines ball and keypoint predictions using computer vision techniques.

## 🎓 Model Training

 ### Keypoint Detection

This model predicts 14 keypoints representing the tennis court's main features. It’s based on a ResNet-50 architecture, with the last layer adjusted to output 28 values (x, y coordinates for each of the 14 keypoints). Training involves a dataset of tennis court images with annotated keypoints.
To train the keypoint detection model, follow the steps in the `keypoint_training.ipynb notebook`. It walks you through dataset preparation, model training, and saving the weights.

### Ball Detection

The Ball Detection model leverages YOLOv5, for detecting and tracking the tennis ball in video frames. The model is trained on a custom dataset of annotated tennis ball images, enabling it to accurately detect the ball across frames. Training steps are outlined in the `ball_detector_training.ipynb notebook`, which guides you through dataset preparation, model training, and deployment. 

## 🚀 Usage

To run the program on your chosen input video, use the following command:

<pre>
<code>
python main.py path_to_input_video.mp4 --output_video output_videos/output_result.avi
</code>
</pre>

- `path_to_input_video.mp4`: Replace this with the path to the video you want to analyze.
 
- `--output_video output_videos/output_result.avi`: This flag specifies where the processed video (with annotations and visualizations) will be saved.

- If you prefer to use a different set of intermediate files or run the process from scratch, you can specify a custom path using the `--stub_path` flag or disable stubs entirely.

  ## 👀 Here's a glimpse...
  <br><br>
<p align="center">
  <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXNjanhxNGl5d2tuaTM0ZjFmanIwbmh4dDA5dHZoZmsxNmdxbHJyaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/9yl0AGKWrQR6cPsEQK/giphy.gif"   width="700px">
</p>
<br><br>

## Requirements 📋

To run the project, you'll need the following dependencies:

- Python 3.8
- PyTorch (with CUDA support if using GPU)
- OpenCV
- NumPy
- Pillow
- Torchvision
- Roboflow
- Ultralytics (for YOLOv5 training)
- scikit-learn (for evaluation and preprocessing)
- Matplotlib (for visualizations)

You can install these dependencies using `pip`:
<pre>
<code>
pip install -r requirements.txt
</code>
</pre>

## 🌱 Future Work 

1. **Player Detection Enhancement**  
   Improve the robustness of player detection, particularly in occluded or crowded scenes, by exploring advanced techniques such as pose estimation or multi-task learning.

2. **Real-Time Performance Optimization**  
   Focus on improving the system’s efficiency for real-time analysis on resource-constrained devices, using approaches like model quantization and edge computing.

3. **Event Detection and Analysis**  
   Implement automated detection of key match events (e.g., aces, faults) to provide actionable insights on player performance during the game.

  

