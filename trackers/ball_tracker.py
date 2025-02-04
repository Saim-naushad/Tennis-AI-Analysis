from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.conf_threshold = 0.2

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold)[0]
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        if read_from_stub and stub_path:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate(method='cubic').bfill()

        df_ball_positions['delta_x'] = df_ball_positions['x1'].diff().abs()
        df_ball_positions['delta_y'] = df_ball_positions['y1'].diff().abs()

        ball_positions = [{1: row[['x1', 'y1', 'x2', 'y2']].tolist()} for _, row in df_ball_positions.iterrows()]
        return ball_positions

    def draw_ball_path(self, video_frames, ball_detections):
        ball_positions = []
        max_trail_length = 4
        base_size = 9
        min_size = 3
        color = (111, 32, 243)

        output_video_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections):
            if 1 in ball_dict:
                x1, y1, x2, y2 = ball_dict[1]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                ball_positions.append((cx, cy))

            if len(ball_positions) > max_trail_length:
                ball_positions.pop(0)

            for i, (px, py) in enumerate(ball_positions):
                alpha = (i + 1) / max_trail_length
                size = int(base_size - (base_size - min_size) * (i / max_trail_length))

                overlay = frame.copy()
                cv2.circle(overlay, (px, py), size, color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            output_video_frames.append(frame)

        return output_video_frames
