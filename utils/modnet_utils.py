"""
MODNet Utility Functions
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_image_for_modnet(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Preprocess image for MODNet inference
    
    Args:
        image: Input image in BGR format
        target_size: Target size for model input
        
    Returns:
        Preprocessed image tensor
    """
    # Resize image
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    # Add batch dimension and transpose to NCHW
    batch = np.expand_dims(normalized, axis=0)
    tensor = np.transpose(batch, (0, 3, 1, 2))
    
    return tensor


def postprocess_modnet_matte(matte: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    """
    Postprocess MODNet output matte
    
    Args:
        matte: Model output matte
        original_size: Original image size (width, height)
        
    Returns:
        Processed alpha matte
    """
    # Remove batch dimension if present
    if len(matte.shape) == 4:
        matte = np.squeeze(matte, axis=0)
    if len(matte.shape) == 3:
        matte = np.squeeze(matte, axis=0)
    
    # Resize to original size
    resized_matte = cv2.resize(matte, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to 0-255 range
    alpha_matte = (resized_matte * 255).astype(np.uint8)
    
    return alpha_matte


def apply_trimap_refinement(matte: np.ndarray, trimap: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply trimap-based refinement to improve matte quality
    
    Args:
        matte: Alpha matte from MODNet
        trimap: Optional trimap for refinement
        
    Returns:
        Refined alpha matte
    """
    if trimap is None:
        # Generate automatic trimap
        trimap = generate_trimap_from_matte(matte)
    
    # Apply morphological operations for refinement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Smooth uncertain regions
    uncertain_mask = (trimap == 128)
    if np.any(uncertain_mask):
        matte_smooth = cv2.morphologyEx(matte, cv2.MORPH_CLOSE, kernel)
        matte = np.where(uncertain_mask, matte_smooth, matte)
    
    return matte


def generate_trimap_from_matte(matte: np.ndarray, fg_threshold: float = 0.8, 
                              bg_threshold: float = 0.2) -> np.ndarray:
    """
    Generate trimap from alpha matte
    
    Args:
        matte: Alpha matte (0-255)
        fg_threshold: Threshold for foreground (0-1)
        bg_threshold: Threshold for background (0-1)
        
    Returns:
        Trimap (0=background, 128=uncertain, 255=foreground)
    """
    normalized_matte = matte.astype(np.float32) / 255.0
    
    trimap = np.zeros_like(matte, dtype=np.uint8)
    trimap[normalized_matte >= fg_threshold] = 255  # Foreground
    trimap[normalized_matte <= bg_threshold] = 0    # Background
    trimap[(normalized_matte > bg_threshold) & (normalized_matte < fg_threshold)] = 128  # Uncertain
    
    return trimap


def blend_with_background(foreground: np.ndarray, background: np.ndarray, 
                         alpha_matte: np.ndarray) -> np.ndarray:
    """
    Blend foreground with background using alpha matte
    
    Args:
        foreground: Foreground image
        background: Background image
        alpha_matte: Alpha matte (0-255)
        
    Returns:
        Blended image
    """
    # Ensure all images have the same size
    h, w = foreground.shape[:2]
    background = cv2.resize(background, (w, h))
    alpha_matte = cv2.resize(alpha_matte, (w, h))
    
    # Normalize alpha to [0, 1]
    alpha = alpha_matte.astype(np.float32) / 255.0
    
    # Expand alpha to 3 channels
    if len(alpha.shape) == 2:
        alpha = np.stack([alpha] * 3, axis=2)
    
    # Blend images
    blended = foreground.astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)
    
    return blended.astype(np.uint8)


def refine_edges(matte: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """
    Refine matte edges using guided filtering
    
    Args:
        matte: Alpha matte
        original_image: Original input image
        
    Returns:
        Edge-refined matte
    """
    try:
        # Convert to grayscale for guidance
        if len(original_image.shape) == 3:
            guide = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            guide = original_image
        
        # Apply bilateral filter for edge-aware smoothing
        refined_matte = cv2.bilateralFilter(matte, 9, 75, 75)
        
        # Use original matte for high-confidence regions
        high_conf_mask = (matte > 200) | (matte < 50)
        refined_matte = np.where(high_conf_mask, matte, refined_matte)
        
        return refined_matte
        
    except Exception:
        return matte