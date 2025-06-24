"""
Real-ESRGAN Utility Functions
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_for_esrgan(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for Real-ESRGAN inference
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Preprocessed image tensor
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized = rgb_image.astype(np.float32) / 255.0
    
    # Add batch dimension and transpose to NCHW
    batch = np.expand_dims(normalized, axis=0)
    tensor = np.transpose(batch, (0, 3, 1, 2))
    
    return tensor


def postprocess_esrgan_output(output: np.ndarray) -> np.ndarray:
    """
    Postprocess Real-ESRGAN output
    
    Args:
        output: Model output tensor
        
    Returns:
        Processed image in BGR format
    """
    # Remove batch dimension and transpose to HWC
    if len(output.shape) == 4:
        output = np.squeeze(output, axis=0)
    output = np.transpose(output, (1, 2, 0))
    
    # Denormalize from [0, 1] to [0, 255]
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    
    # Convert RGB to BGR
    bgr_output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    return bgr_output


def tile_process(image: np.ndarray, tile_size: int = 512, overlap: int = 32) -> np.ndarray:
    """
    Process large images in tiles to manage memory usage
    
    Args:
        image: Input image
        tile_size: Size of each tile
        overlap: Overlap between tiles
        
    Returns:
        Processed image
    """
    height, width = image.shape[:2]
    
    # If image is small enough, process directly
    if height <= tile_size and width <= tile_size:
        return enhance_tile(image)
    
    # Calculate number of tiles
    tiles_y = (height - overlap) // (tile_size - overlap) + 1
    tiles_x = (width - overlap) // (tile_size - overlap) + 1
    
    # Create output image (assuming 4x upscaling)
    output_height, output_width = height * 4, width * 4
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    for y in range(tiles_y):
        for x in range(tiles_x):
            # Calculate tile boundaries
            start_y = y * (tile_size - overlap)
            end_y = min(start_y + tile_size, height)
            start_x = x * (tile_size - overlap)
            end_x = min(start_x + tile_size, width)
            
            # Extract tile
            tile = image[start_y:end_y, start_x:end_x]
            
            # Process tile
            enhanced_tile = enhance_tile(tile)
            
            # Calculate output position
            out_start_y = start_y * 4
            out_end_y = end_y * 4
            out_start_x = start_x * 4
            out_end_x = end_x * 4
            
            # Place enhanced tile in output
            output_image[out_start_y:out_end_y, out_start_x:out_end_x] = enhanced_tile
    
    return output_image


def enhance_tile(tile: np.ndarray) -> np.ndarray:
    """
    Enhance a single tile (placeholder for actual ESRGAN inference)
    
    Args:
        tile: Input tile
        
    Returns:
        Enhanced tile
    """
    # Placeholder: Use bicubic upscaling
    height, width = tile.shape[:2]
    enhanced = cv2.resize(tile, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
    
    # Apply sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Blend original and sharpened
    result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_face_enhancement(image: np.ndarray, face_regions: Optional[list] = None) -> np.ndarray:
    """
    Apply specialized face enhancement
    
    Args:
        image: Input image
        face_regions: List of face bounding boxes
        
    Returns:
        Face-enhanced image
    """
    if face_regions is None:
        # Simple face detection using Haar cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_regions = faces
    
    enhanced_image = image.copy()
    
    for (x, y, w, h) in face_regions:
        # Extract face region
        face = image[y:y+h, x:x+w]
        
        # Apply face-specific enhancement
        enhanced_face = enhance_face_region(face)
        
        # Blend back into image
        enhanced_image[y:y+h, x:x+w] = enhanced_face
    
    return enhanced_image


def enhance_face_region(face: np.ndarray) -> np.ndarray:
    """
    Apply face-specific enhancement techniques
    
    Args:
        face: Face region image
        
    Returns:
        Enhanced face region
    """
    # Apply bilateral filter for skin smoothing
    smoothed = cv2.bilateralFilter(face, 9, 75, 75)
    
    # Enhance eyes and mouth (high-frequency details)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Blend smoothed face with edge details
    enhanced = cv2.addWeighted(smoothed, 0.8, edges_colored, 0.2, 0)
    
    return enhanced


def apply_clothing_enhancement(image: np.ndarray, clothing_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply clothing-specific enhancement
    
    Args:
        image: Input image
        clothing_mask: Binary mask for clothing regions
        
    Returns:
        Clothing-enhanced image
    """
    if clothing_mask is None:
        # Create a simple clothing mask (everything except skin tones)
        clothing_mask = create_clothing_mask(image)
    
    enhanced_image = image.copy()
    
    # Apply enhancement only to clothing regions
    clothing_regions = cv2.bitwise_and(image, image, mask=clothing_mask)
    
    # Enhance texture and colors
    enhanced_clothing = enhance_clothing_texture(clothing_regions)
    
    # Blend back
    mask_3d = cv2.merge([clothing_mask, clothing_mask, clothing_mask]) / 255.0
    enhanced_image = enhanced_image * (1 - mask_3d) + enhanced_clothing * mask_3d
    
    return enhanced_image.astype(np.uint8)


def create_clothing_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a simple clothing mask by excluding skin tones
    
    Args:
        image: Input image
        
    Returns:
        Binary clothing mask
    """
    # Convert to HSV for better skin detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define skin color range
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    
    # Create skin mask
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Clothing mask is inverse of skin mask
    clothing_mask = cv2.bitwise_not(skin_mask)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
    clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel)
    
    return clothing_mask


def enhance_clothing_texture(clothing_image: np.ndarray) -> np.ndarray:
    """
    Enhance clothing texture and patterns
    
    Args:
        clothing_image: Clothing regions image
        
    Returns:
        Texture-enhanced clothing image
    """
    # Apply unsharp masking for texture enhancement
    blurred = cv2.GaussianBlur(clothing_image, (0, 0), 1.0)
    sharpened = cv2.addWeighted(clothing_image, 1.5, blurred, -0.5, 0)
    
    # Enhance local contrast
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def calculate_enhancement_metrics(original: np.ndarray, enhanced: np.ndarray) -> dict:
    """
    Calculate quality metrics for enhancement
    
    Args:
        original: Original image
        enhanced: Enhanced image
        
    Returns:
        Dictionary of quality metrics
    """
    # Resize enhanced to match original for comparison
    if enhanced.shape != original.shape:
        scale_factor = enhanced.shape[0] // original.shape[0]
        enhanced_resized = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    else:
        enhanced_resized = enhanced
        scale_factor = 1
    
    # Calculate PSNR
    mse = np.mean((original.astype(np.float32) - enhanced_resized.astype(np.float32)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calculate sharpness (variance of Laplacian)
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced_resized, cv2.COLOR_BGR2GRAY)
    
    sharpness_original = cv2.Laplacian(gray_original, cv2.CV_64F).var()
    sharpness_enhanced = cv2.Laplacian(gray_enhanced, cv2.CV_64F).var()
    
    return {
        'psnr': psnr,
        'scale_factor': scale_factor,
        'sharpness_improvement': sharpness_enhanced / sharpness_original if sharpness_original > 0 else 0,
        'original_sharpness': sharpness_original,
        'enhanced_sharpness': sharpness_enhanced
    }