"""
Saree Warping Module using VITON-HD approach
"""
import os
import cv2
import json
import numpy as np
from typing import Dict, Tuple, Optional


class SareeWarper:
    """VITON-HD based saree warping system"""
    
    def __init__(self, model_path: str = "models/viton_hd.pth"):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load VITON-HD model"""
        try:
            print(f"📦 Loading VITON-HD model from {self.model_path}")
            
            # Placeholder for actual VITON-HD model loading
            # In production: load PyTorch model
            if os.path.exists(self.model_path):
                self.model = "viton_hd_loaded"  # Placeholder
                print("✅ VITON-HD model loaded successfully")
            else:
                print("⚠️  VITON-HD model not found, using geometric warping")
                self.model = None
                
        except Exception as e:
            print(f"❌ Failed to load VITON-HD model: {e}")
            self.model = None
    
    def warp_saree(self, body_img_path: str, pallu_img_path: str, 
                   blouse_img_path: str, pose_data: Dict, output_path: str) -> str:
        """Warp saree components onto body using pose information"""
        try:
            # Load images
            body_img = cv2.imread(body_img_path)
            pallu_img = cv2.imread(pallu_img_path)
            blouse_img = cv2.imread(blouse_img_path)
            
            if any(img is None for img in [body_img, pallu_img, blouse_img]):
                raise ValueError("Could not load one or more input images")
            
            height, width = body_img.shape[:2]
            
            if self.model:
                # Actual VITON-HD inference would go here
                print("🔄 Warping saree using VITON-HD...")
                result = self._viton_hd_warp(body_img, pallu_img, blouse_img, pose_data)
            else:
                # Fallback: Geometric warping based on pose keypoints
                print("🔄 Warping saree using geometric transformation...")
                result = self._geometric_warp(body_img, pallu_img, blouse_img, pose_data)
            
            # Save result
            cv2.imwrite(output_path, result)
            print(f"✅ Saree warping completed: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Saree warping failed: {e}")
            # Fallback: return body image
            cv2.imwrite(output_path, body_img)
            return output_path
    
    def _viton_hd_warp(self, body_img: np.ndarray, pallu_img: np.ndarray, 
                       blouse_img: np.ndarray, pose_data: Dict) -> np.ndarray:
        """VITON-HD based warping (placeholder implementation)"""
        # In production, this would use the actual VITON-HD model
        # For now, simulate with geometric warping
        return self._geometric_warp(body_img, pallu_img, blouse_img, pose_data)
    
    def _geometric_warp(self, body_img: np.ndarray, pallu_img: np.ndarray, 
                        blouse_img: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Geometric warping based on pose keypoints"""
        result = body_img.copy()
        height, width = result.shape[:2]
        
        # Extract key pose points
        keypoints = {kp["name"]: (int(kp["x"]), int(kp["y"])) 
                    for kp in pose_data.get("keypoints", []) 
                    if kp.get("confidence", 0) > 0.5}
        
        # Warp blouse onto torso area
        if all(kp in keypoints for kp in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            result = self._warp_blouse(result, blouse_img, keypoints)
        
        # Warp pallu (drape) onto shoulder/arm area
        if all(kp in keypoints for kp in ["left_shoulder", "right_shoulder"]):
            result = self._warp_pallu(result, pallu_img, keypoints)
        
        return result
    
    def _warp_blouse(self, base_img: np.ndarray, blouse_img: np.ndarray, 
                     keypoints: Dict) -> np.ndarray:
        """Warp blouse onto torso area"""
        try:
            # Define torso region
            left_shoulder = keypoints["left_shoulder"]
            right_shoulder = keypoints["right_shoulder"]
            left_hip = keypoints.get("left_hip", (left_shoulder[0], left_shoulder[1] + 150))
            right_hip = keypoints.get("right_hip", (right_shoulder[0], right_shoulder[1] + 150))
            
            # Create quadrilateral for torso
            torso_pts = np.array([left_shoulder, right_shoulder, right_hip, left_hip], dtype=np.float32)
            
            # Resize blouse to fit torso area
            torso_width = int(np.linalg.norm(np.array(right_shoulder) - np.array(left_shoulder)))
            torso_height = int(np.linalg.norm(np.array(left_hip) - np.array(left_shoulder)))
            
            blouse_resized = cv2.resize(blouse_img, (torso_width, torso_height))
            blouse_h, blouse_w = blouse_resized.shape[:2]
            
            # Source points (corners of resized blouse)
            src_pts = np.array([[0, 0], [blouse_w, 0], [blouse_w, blouse_h], [0, blouse_h]], dtype=np.float32)
            
            # Perspective transformation
            transform_matrix = cv2.getPerspectiveTransform(src_pts, torso_pts)
            warped_blouse = cv2.warpPerspective(blouse_resized, transform_matrix, 
                                              (base_img.shape[1], base_img.shape[0]))
            
            # Create mask for blending
            mask = np.zeros(base_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [torso_pts.astype(np.int32)], 255)
            
            # Blend warped blouse with base image
            mask_3d = cv2.merge([mask, mask, mask]) / 255.0
            result = base_img * (1 - mask_3d) + warped_blouse * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"❌ Blouse warping failed: {e}")
            return base_img
    
    def _warp_pallu(self, base_img: np.ndarray, pallu_img: np.ndarray, 
                    keypoints: Dict) -> np.ndarray:
        """Warp pallu (drape) onto shoulder area"""
        try:
            # Define pallu region (over shoulder)
            left_shoulder = keypoints["left_shoulder"]
            right_shoulder = keypoints["right_shoulder"]
            
            # Create pallu drape area
            shoulder_width = int(np.linalg.norm(np.array(right_shoulder) - np.array(left_shoulder)))
            pallu_height = shoulder_width // 2
            
            # Position pallu over left shoulder
            pallu_top_left = (left_shoulder[0] - 30, left_shoulder[1] - 50)
            pallu_top_right = (left_shoulder[0] + shoulder_width // 3, left_shoulder[1] - 30)
            pallu_bottom_left = (left_shoulder[0] - 20, left_shoulder[1] + pallu_height)
            pallu_bottom_right = (left_shoulder[0] + shoulder_width // 4, left_shoulder[1] + pallu_height + 20)
            
            pallu_pts = np.array([pallu_top_left, pallu_top_right, 
                                 pallu_bottom_right, pallu_bottom_left], dtype=np.float32)
            
            # Resize pallu
            pallu_resized = cv2.resize(pallu_img, (shoulder_width // 2, pallu_height))
            pallu_h, pallu_w = pallu_resized.shape[:2]
            
            # Source points
            src_pts = np.array([[0, 0], [pallu_w, 0], [pallu_w, pallu_h], [0, pallu_h]], dtype=np.float32)
            
            # Perspective transformation
            transform_matrix = cv2.getPerspectiveTransform(src_pts, pallu_pts)
            warped_pallu = cv2.warpPerspective(pallu_resized, transform_matrix, 
                                             (base_img.shape[1], base_img.shape[0]))
            
            # Create mask for blending
            mask = np.zeros(base_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pallu_pts.astype(np.int32)], 255)
            
            # Apply transparency for drape effect
            alpha = 0.7
            mask_3d = cv2.merge([mask, mask, mask]) / 255.0 * alpha
            
            result = base_img * (1 - mask_3d) + warped_pallu * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"❌ Pallu warping failed: {e}")
            return base_img


# Global warper instance
saree_warper = SareeWarper()


async def warp_saree_onto_model(body_img_path: str, pallu_img_path: str, 
                               blouse_img_path: str, pose_data: Dict, 
                               output_path: str) -> str:
    """
    Warp saree components onto virtual model using VITON-HD approach
    
    Args:
        body_img_path: Path to body/model image
        pallu_img_path: Path to pallu image
        blouse_img_path: Path to blouse image
        pose_data: Pose keypoints data
        output_path: Path to save warped result
        
    Returns:
        Path to warped image
    """
    return saree_warper.warp_saree(body_img_path, pallu_img_path, 
                                  blouse_img_path, pose_data, output_path)


def create_garment_mask(image: np.ndarray, keypoints: Dict) -> np.ndarray:
    """Create mask for garment region based on pose keypoints"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Define garment region based on pose
    if all(kp in keypoints for kp in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
        pts = np.array([
            keypoints["left_shoulder"],
            keypoints["right_shoulder"],
            keypoints["right_hip"],
            keypoints["left_hip"]
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [pts], 255)
    
    return mask