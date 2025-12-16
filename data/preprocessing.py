"""
Face dataset preprocessing with alignment, quality checks, and augmentation
"""
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import face_alignment
from facenet_pytorch import MTCNN
import torch
from typing import Tuple, List, Optional
import json

class FacePreprocessor:
    """Handles face detection, alignment, and preprocessing"""
    
    def __init__(self, 
                 output_size: int = 256,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            output_size: Target resolution (square)
            device: CUDA or CPU
        """
        self.output_size = output_size
        self.device = device
        
        # Initialize face detector
        self.mtcnn = MTCNN(
            image_size=output_size,
            margin=0,
            keep_all=False,
            device=device,
            post_process=False
        )
        
        # Initialize landmark detector for alignment
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, 
            device=device
        )
        
    def detect_and_align(self, image_path: str) -> Optional[np.ndarray]:
        """
        Detect face, align using landmarks, and crop
        
        Args:
            image_path: Path to input image
            
        Returns:
            Aligned face array or None if detection fails
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect face
            boxes, probs, landmarks = self.mtcnn.detect(img_rgb, landmarks=True)
            
            if boxes is None or len(boxes) == 0:
                return None
            
            # Use highest confidence detection
            idx = np.argmax(probs)
            box = boxes[idx]
            landmark = landmarks[idx]
            
            # Align using landmarks (5-point alignment)
            aligned = self._align_face(img_rgb, landmark)
            
            # Crop to square with padding
            face = self._crop_face(aligned, box)
            
            # Resize to target resolution
            face = cv2.resize(face, (self.output_size, self.output_size), 
                            interpolation=cv2.INTER_CUBIC)
            
            return face
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def _align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align face using eye landmarks
        
        Args:
            image: Input image
            landmarks: 5 facial landmarks (eyes, nose, mouth corners)
            
        Returns:
            Aligned image
        """
        # Get eye centers
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Calculate rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate center
        center = ((left_eye[0] + right_eye[0]) // 2,
                 (left_eye[1] + right_eye[1]) // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_CUBIC)
        
        return aligned
    
    def _crop_face(self, image: np.ndarray, box: np.ndarray, 
                   padding: float = 0.3) -> np.ndarray:
        """
        Crop face with padding
        
        Args:
            image: Input image
            box: Bounding box [x1, y1, x2, y2]
            padding: Padding ratio
            
        Returns:
            Cropped face
        """
        x1, y1, x2, y2 = box.astype(int)
        
        # Add padding
        w = x2 - x1
        h = y2 - y1
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(image.shape[1], x2 + pad_w)
        y2 = min(image.shape[0], y2 + pad_h)
        
        # Crop
        face = image[y1:y2, x1:x2]
        
        return face
    
    def check_quality(self, image: np.ndarray) -> dict:
        """
        Check image quality metrics
        
        Args:
            image: Input image
            
        Returns:
            Quality metrics dict
        """
        # Convert to grayscale for some metrics
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Blur detection (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Resolution
        resolution = image.shape[:2]
        
        return {
            'blur_score': blur_score,
            'brightness': brightness,
            'contrast': contrast,
            'resolution': resolution,
            'is_acceptable': blur_score > 100  # Threshold for blur
        }

def preprocess_dataset(input_dir: str,
                      output_dir: str,
                      output_size: int = 256,
                      skip_existing: bool = True):
    """
    Preprocess entire dataset
    
    Args:
        input_dir: Input directory with raw images
        output_dir: Output directory for processed images
        output_size: Target resolution
        skip_existing: Skip already processed images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = FacePreprocessor(output_size=output_size)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_path.rglob('*') 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images")
    
    # Statistics
    stats = {
        'total': len(image_files),
        'processed': 0,
        'failed': 0,
        'skipped': 0,
        'quality_rejected': 0
    }
    
    quality_log = []
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Output path
        rel_path = img_path.relative_to(input_path)
        out_path = output_path / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if exists
        if skip_existing and out_path.exists():
            stats['skipped'] += 1
            continue
        
        # Detect and align
        face = preprocessor.detect_and_align(str(img_path))
        
        if face is None:
            stats['failed'] += 1
            continue
        
        # Check quality
        quality = preprocessor.check_quality(face)
        quality['filename'] = str(rel_path)
        quality_log.append(quality)
        
        if not quality['is_acceptable']:
            stats['quality_rejected'] += 1
            continue
        
        # Save
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), face_bgr)
        stats['processed'] += 1
    
    # Save quality report
    quality_path = output_path / 'quality_report.json'
    with open(quality_path, 'w') as f:
        json.dump({
            'statistics': stats,
            'quality_metrics': quality_log
        }, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"Processed: {stats['processed']}")
    print(f"Failed detection: {stats['failed']}")
    print(f"Quality rejected: {stats['quality_rejected']}")
    print(f"Skipped (existing): {stats['skipped']}")

if __name__ == "__main__":
    # Example usage
    preprocess_dataset(
        input_dir="./data/raw/celeba",
        output_dir="./data/processed/celeba_256",
        output_size=256
    )