def get_center_of_bbox(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_foot_position(bbox):
    return (int((bbox[0] + bbox[2]) / 2), bbox[3])

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_indx in keypoint_indices:
        keypoint = (keypoints[keypoint_indx * 2], keypoints[keypoint_indx * 2 + 1])
        distance = abs(point[1] - keypoint[1])
        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_indx
    return key_point_ind

def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]

def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
