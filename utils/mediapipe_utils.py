"""
MediaPipe Utility Functions
"""
import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional


# MediaPipe pose landmark indices
POSE_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

# Pose connections for skeleton visualization
POSE_CONNECTIONS = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear')
]


def normalize_pose_landmarks(landmarks: List[Dict], image_width: int, image_height: int) -> List[Dict]:
    """
    Normalize pose landmarks to image coordinates
    
    Args:
        landmarks: List of landmark dictionaries
        image_width: Image width
        image_height: Image height
        
    Returns:
        Normalized landmarks
    """
    normalized = []
    for landmark in landmarks:
        normalized_landmark = landmark.copy()
        normalized_landmark['x'] = landmark['x'] * image_width
        normalized_landmark['y'] = landmark['y'] * image_height
        normalized.append(normalized_landmark)
    
    return normalized


def filter_pose_landmarks(landmarks: List[Dict], confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Filter pose landmarks by confidence threshold
    
    Args:
        landmarks: List of landmark dictionaries
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Filtered landmarks
    """
    return [lm for lm in landmarks if lm.get('confidence', 0) >= confidence_threshold]


def get_body_bounding_box(landmarks: List[Dict]) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box around body landmarks
    
    Args:
        landmarks: List of landmark dictionaries
        
    Returns:
        Bounding box (x, y, width, height)
    """
    if not landmarks:
        return (0, 0, 0, 0)
    
    x_coords = [lm['x'] for lm in landmarks]
    y_coords = [lm['y'] for lm in landmarks]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Add padding
    padding = 20
    x = max(0, int(min_x - padding))
    y = max(0, int(min_y - padding))
    width = int(max_x - min_x + 2 * padding)
    height = int(max_y - min_y + 2 * padding)
    
    return (x, y, width, height)


def get_torso_region(landmarks: List[Dict]) -> Optional[np.ndarray]:
    """
    Extract torso region coordinates from pose landmarks
    
    Args:
        landmarks: List of landmark dictionaries
        
    Returns:
        Torso region points as numpy array
    """
    landmark_dict = {lm['name']: (lm['x'], lm['y']) for lm in landmarks}
    
    required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    if not all(point in landmark_dict for point in required_points):
        return None
    
    # Define torso quadrilateral
    torso_points = np.array([
        landmark_dict['left_shoulder'],
        landmark_dict['right_shoulder'],
        landmark_dict['right_hip'],
        landmark_dict['left_hip']
    ], dtype=np.float32)
    
    return torso_points


def get_arm_regions(landmarks: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract left and right arm regions from pose landmarks
    
    Args:
        landmarks: List of landmark dictionaries
        
    Returns:
        Tuple of (left_arm_points, right_arm_points)
    """
    landmark_dict = {lm['name']: (lm['x'], lm['y']) for lm in landmarks}
    
    left_arm_points = None
    right_arm_points = None
    
    # Left arm
    left_arm_landmarks = ['left_shoulder', 'left_elbow', 'left_wrist']
    if all(point in landmark_dict for point in left_arm_landmarks):
        left_arm_points = np.array([landmark_dict[point] for point in left_arm_landmarks], dtype=np.float32)
    
    # Right arm
    right_arm_landmarks = ['right_shoulder', 'right_elbow', 'right_wrist']
    if all(point in landmark_dict for point in right_arm_landmarks):
        right_arm_points = np.array([landmark_dict[point] for point in right_arm_landmarks], dtype=np.float32)
    
    return left_arm_points, right_arm_points


def calculate_body_orientation(landmarks: List[Dict]) -> float:
    """
    Calculate body orientation angle from pose landmarks
    
    Args:
        landmarks: List of landmark dictionaries
        
    Returns:
        Body orientation angle in degrees
    """
    landmark_dict = {lm['name']: (lm['x'], lm['y']) for lm in landmarks}
    
    if 'left_shoulder' not in landmark_dict or 'right_shoulder' not in landmark_dict:
        return 0.0
    
    left_shoulder = np.array(landmark_dict['left_shoulder'])
    right_shoulder = np.array(landmark_dict['right_shoulder'])
    
    # Calculate shoulder line angle
    shoulder_vector = right_shoulder - left_shoulder
    angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
    
    return np.degrees(angle)


def create_pose_mask(landmarks: List[Dict], image_shape: Tuple[int, int], 
                    region: str = 'torso') -> np.ndarray:
    """
    Create binary mask for specific body region
    
    Args:
        landmarks: List of landmark dictionaries
        image_shape: Image shape (height, width)
        region: Body region ('torso', 'full_body', 'upper_body')
        
    Returns:
        Binary mask
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    landmark_dict = {lm['name']: (int(lm['x']), int(lm['y'])) for lm in landmarks}
    
    if region == 'torso':
        required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        if all(point in landmark_dict for point in required_points):
            points = np.array([landmark_dict[point] for point in required_points], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
    
    elif region == 'upper_body':
        # Include torso and arms
        torso_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        if all(point in landmark_dict for point in torso_points):
            # Torso
            points = np.array([landmark_dict[point] for point in torso_points], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            
            # Arms (if available)
            for side in ['left', 'right']:
                arm_points = [f'{side}_shoulder', f'{side}_elbow', f'{side}_wrist']
                if all(point in landmark_dict for point in arm_points):
                    # Create arm region
                    shoulder = landmark_dict[f'{side}_shoulder']
                    elbow = landmark_dict[f'{side}_elbow']
                    wrist = landmark_dict[f'{side}_wrist']
                    
                    # Draw thick line for arm
                    cv2.line(mask, shoulder, elbow, 255, 30)
                    cv2.line(mask, elbow, wrist, 255, 25)
    
    elif region == 'full_body':
        # Create mask for entire visible body
        all_points = [(int(lm['x']), int(lm['y'])) for lm in landmarks 
                     if lm.get('confidence', 0) > 0.5]
        if all_points:
            # Create convex hull around all points
            points_array = np.array(all_points, dtype=np.int32)
            hull = cv2.convexHull(points_array)
            cv2.fillPoly(mask, [hull], 255)
    
    return mask


def smooth_pose_sequence(pose_sequence: List[List[Dict]], window_size: int = 5) -> List[List[Dict]]:
    """
    Smooth pose landmarks across a sequence of frames
    
    Args:
        pose_sequence: List of pose landmark lists for each frame
        window_size: Smoothing window size
        
    Returns:
        Smoothed pose sequence
    """
    if len(pose_sequence) < window_size:
        return pose_sequence
    
    smoothed_sequence = []
    half_window = window_size // 2
    
    for i in range(len(pose_sequence)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(pose_sequence), i + half_window + 1)
        
        # Average landmarks across window
        window_poses = pose_sequence[start_idx:end_idx]
        smoothed_landmarks = []
        
        if window_poses and window_poses[0]:
            for j in range(len(window_poses[0])):
                landmark_name = window_poses[0][j]['name']
                
                # Average coordinates
                x_coords = [pose[j]['x'] for pose in window_poses if j < len(pose)]
                y_coords = [pose[j]['y'] for pose in window_poses if j < len(pose)]
                confidences = [pose[j].get('confidence', 0) for pose in window_poses if j < len(pose)]
                
                smoothed_landmark = {
                    'name': landmark_name,
                    'x': np.mean(x_coords),
                    'y': np.mean(y_coords),
                    'confidence': np.mean(confidences)
                }
                smoothed_landmarks.append(smoothed_landmark)
        
        smoothed_sequence.append(smoothed_landmarks)
    
    return smoothed_sequence


def export_pose_to_openpose_format(landmarks: List[Dict], image_width: int, image_height: int) -> Dict:
    """
    Convert MediaPipe pose to OpenPose format
    
    Args:
        landmarks: MediaPipe landmarks
        image_width: Image width
        image_height: Image height
        
    Returns:
        OpenPose format dictionary
    """
    # OpenPose keypoint order (COCO format)
    openpose_order = [
        'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
        'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip',
        'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
        'right_eye', 'left_eye', 'right_ear', 'left_ear'
    ]
    
    landmark_dict = {lm['name']: lm for lm in landmarks}
    
    # Calculate neck position (midpoint between shoulders)
    neck_pos = None
    if 'left_shoulder' in landmark_dict and 'right_shoulder' in landmark_dict:
        left_shoulder = landmark_dict['left_shoulder']
        right_shoulder = landmark_dict['right_shoulder']
        neck_pos = {
            'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
            'y': (left_shoulder['y'] + right_shoulder['y']) / 2,
            'confidence': min(left_shoulder.get('confidence', 0), right_shoulder.get('confidence', 0))
        }
    
    openpose_keypoints = []
    for keypoint_name in openpose_order:
        if keypoint_name == 'neck' and neck_pos:
            kp = neck_pos
        elif keypoint_name in landmark_dict:
            kp = landmark_dict[keypoint_name]
        else:
            kp = {'x': 0, 'y': 0, 'confidence': 0}
        
        # OpenPose format: [x, y, confidence]
        openpose_keypoints.extend([kp['x'], kp['y'], kp.get('confidence', 0)])
    
    return {
        'version': 1.3,
        'people': [{
            'person_id': [-1],
            'pose_keypoints_2d': openpose_keypoints,
            'face_keypoints_2d': [],
            'hand_left_keypoints_2d': [],
            'hand_right_keypoints_2d': [],
            'pose_keypoints_3d': [],
            'face_keypoints_3d': [],
            'hand_left_keypoints_3d': [],
            'hand_right_keypoints_3d': []
        }]
    }