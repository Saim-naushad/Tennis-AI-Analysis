import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance,
)

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
        self.previous_ball_position = None  # For smoothing ball mapping
        self.smoothing_factor = 0.7  # Adjust this factor for smoother transitions

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0] * 28
        # Setting key points based on court dimensions
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1]
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3]
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7]
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
        drawing_key_points[25] = drawing_key_points[17]
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (12, 13),
            (2, 3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self, frame):

        # Drawing Lines (court)
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)  # White lines for court

        # Drawing Net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 255, 255), 2)  # White for the net

        return frame

    def draw_background_rectangle(self, frame):
        # Create a blue background
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (223, 134, 85), cv2.FILLED)  # Blue color
        # Simply replace the frame's region with the blue rectangle
        out = frame.copy()
        out[self.start_y:self.end_y, self.start_x:self.end_x] = shapes[self.start_y:self.end_y, self.start_x:self.end_x]
        return out



    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point,
                                   closest_key_point_index,
                                   player_height_in_pixels,
                                   player_height_in_meters):
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Convert pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels)
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels)
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_court_keypoint = (self.drawing_key_points[closest_key_point_index * 2],
                                       self.drawing_key_points[closest_key_point_index * 2 + 1])

        mini_court_player_position = (closest_mini_court_keypoint[0] + mini_court_x_distance_pixels,
                                      closest_mini_court_keypoint[1] + mini_court_y_distance_pixels)

        return mini_court_player_position

    def smooth_ball_position(self, ball_position, smoothing_factor=0.7):
        # Initialize the previous_ball_position if it doesn't exist or is None
        if not hasattr(self, 'previous_ball_position') or self.previous_ball_position is None:
            self.previous_ball_position = ball_position
            return ball_position

        # Smooth the ball position using a weighted average
        smoothed_position = (
            smoothing_factor * ball_position[0] + (1 - smoothing_factor) * self.previous_ball_position[0],
            smoothing_factor * ball_position[1] + (1 - smoothing_factor) * self.previous_ball_position[1]
        )

        # Update previous ball position for the next frame
        self.previous_ball_position = smoothed_position

        return smoothed_position
        
    def calculate_weighted_mini_court_position(self, object_position, court_key_points, mini_court_key_points, previous_object_y=None, frame_time=None):
        object_x, object_y = object_position

        # Extract x and y coordinates from court keypoints
        court_xs = [court_key_points[i * 2] for i in range(len(court_key_points) // 2)]
        court_ys = [court_key_points[i * 2 + 1] for i in range(len(court_key_points) // 2)]

        # Find the top and bottom indices based on the y-coordinate
        valid_top_indices = [i for i in range(len(court_ys)) if court_ys[i] <= object_y]
        top_idx = max(valid_top_indices) if valid_top_indices else 0

        valid_bottom_indices = [i for i in range(len(court_ys)) if court_ys[i] > object_y]
        bottom_idx = min(valid_bottom_indices) if valid_bottom_indices else len(court_ys) - 1

        # Key points for vertical interpolation (top and bottom)
        top_key_point = (court_key_points[top_idx * 2], court_key_points[top_idx * 2 + 1])
        bottom_key_point = (court_key_points[bottom_idx * 2], court_key_points[bottom_idx * 2 + 1])

        # Mini court key points for vertical interpolation
        top_mini_point = (mini_court_key_points[top_idx * 2], mini_court_key_points[top_idx * 2 + 1])
        bottom_mini_point = (mini_court_key_points[bottom_idx * 2], mini_court_key_points[bottom_idx * 2 + 1])

        # Proportional progression along the court
        court_height = court_ys[-1] - court_ys[0]  # Total height of the court
        proportional_progress = (object_y - court_ys[0]) / (court_height + 1e-6)

        # Direction detection (if available)
        if previous_object_y is not None:
            direction = "near-to-far" if object_y < previous_object_y else "far-to-near"
        else:
            direction = "unknown"

        # Dynamic scaling factor for vertical interpolation (slowed down even more)
        net_proximity_factor = abs(object_y - court_ys[len(court_ys) // 2]) / (court_height / 2 + 1e-6)
        scaling_factor = 0.1 + 0.3 * (1 - net_proximity_factor)  # Even less aggressive scaling

        # Apply scaled interpolation for vertical position (slower)
        if top_idx != bottom_idx:
            vertical_alpha = (object_y - top_key_point[1]) / (bottom_key_point[1] - top_key_point[1] + 1e-6)

            # Adjust with dynamic scaling and proportional progress (slower)
            scaled_vertical_alpha = (1 - scaling_factor) * vertical_alpha + scaling_factor * proportional_progress

            # Directional adjustments (slower)
            if direction == "near-to-far":
                scaled_vertical_alpha += 0.005  # Smaller forward correction for near-to-far
            elif direction == "far-to-near":
                scaled_vertical_alpha -= 0.005  # Smaller backward correction for far-to-near

            # Apply an additional dynamic correction factor for more precise mapping at the far end
            if object_y > court_ys[-1] - 5:
                scaled_vertical_alpha += 0.01  # Slight correction to avoid early mapping

            mini_court_y = (1 - scaled_vertical_alpha) * top_mini_point[1] + scaled_vertical_alpha * bottom_mini_point[1]
        else:
            mini_court_y = top_mini_point[1]  # Handle edge case where top_idx == bottom_idx

        # Horizontal interpolation remains unchanged
        valid_left_indices = [i for i in range(len(court_xs)) if court_xs[i] <= object_x]
        left_idx = max(valid_left_indices) if valid_left_indices else 0

        valid_right_indices = [i for i in range(len(court_xs)) if court_xs[i] > object_x]
        right_idx = min(valid_right_indices) if valid_right_indices else len(court_xs) - 1

        left_key_point = (court_key_points[left_idx * 2], court_key_points[left_idx * 2 + 1])
        right_key_point = (court_key_points[right_idx * 2], court_key_points[right_idx * 2 + 1])

        left_mini_point = (mini_court_key_points[left_idx * 2], mini_court_key_points[left_idx * 2 + 1])
        right_mini_point = (mini_court_key_points[right_idx * 2], mini_court_key_points[right_idx * 2 + 1])

        if left_idx != right_idx:
            horizontal_alpha = (object_x - left_key_point[0]) / (right_key_point[0] - left_key_point[0] + 1e-6)
            mini_court_x = (1 - horizontal_alpha) * left_mini_point[0] + horizontal_alpha * right_mini_point[0]
        else:
            mini_court_x = left_mini_point[0]  # Handle edge case where left_idx == right_idx

        # Ensure the ball position stays within the mini court boundaries
        mini_court_x = max(self.court_start_x, min(self.court_end_x, mini_court_x))
        mini_court_y = max(self.court_start_y, min(self.court_end_y, mini_court_y))

        # Return final position
        return mini_court_x, mini_court_y

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)

            # Process Player and Ball Detection in the current frame
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get closest keypoint for player
                closest_key_point_index = get_closest_keypoint_index(foot_position, original_court_key_points, [0, 2, 12, 13])
                closest_key_point = (original_court_key_points[closest_key_point_index * 2],
                                    original_court_key_points[closest_key_point_index * 2 + 1])

                # Get Player height in pixels
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point,
                                                                            closest_key_point_index,
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )

                output_player_bboxes_dict[player_id] = mini_court_player_position

                # Handle Ball Mapping: Weighted Mapping on Mini-Court
                if closest_player_id_to_ball == player_id:
                    smoothed_ball_position = self.calculate_weighted_mini_court_position(
                        ball_position, original_court_key_points, self.drawing_key_points
                    )
                    # Apply optional smoothing
                    smoothed_ball_position = self.smooth_ball_position(smoothed_ball_position)

                    output_ball_boxes.append({1: smoothed_ball_position})

            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                if isinstance(position, tuple) and len(position) == 2:
                    mini_court_x, mini_court_y = position
                    x, y = int(mini_court_x), int(mini_court_y)
                    cv2.circle(frame, (x, y), 5, color, -1)
        return frames


