from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2

def main():
    # Read video frames
    video_frames = read_video("input_videos/input_video.mp4")

    # Initialize trackers
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/ball_best.pt')

    # Track players and ball
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect court lines
    court_line_detector = CourtLineDetector("models/final_keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose and filter players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Initialize mini court
    mini_court = MiniCourt(video_frames[0])

    # Convert positions to mini court coordinates
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints
    )

    # Draw annotations
    output_video_frames = player_tracker.draw_annotations(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_ball_path(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections, color=(0, 0, 255))
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0,255,255))

    # Save output video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
