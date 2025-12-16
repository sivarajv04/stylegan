"""
Face-specific metrics: identity embeddings, landmarks, attributes
"""
import torch
import torch.nn as nn
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import face_alignment
from tqdm import tqdm
from typing import List, Tuple, Optional
from pathlib import Path
from PIL import Image

class FaceMetricsCalculator:
    """Calculate face-specific quality metrics"""
    
    def __init__(self, device='cuda', use_landmarks=False):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_landmarks = use_landmarks
        
        # Identity embedding model (FaceNet/ArcFace)
        print("Loading FaceNet model...")
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Face detector
        print("Loading MTCNN detector...")
        self.mtcnn = MTCNN(device=self.device, select_largest=False)
        
        # Landmark detector (optional - requires download)
        self.fa = None
        if use_landmarks:
            try:
                print("Loading landmark detector...")
                self.fa = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D,
                    device=self.device
                )
                print("✓ Landmark detector loaded")
            except Exception as e:
                print(f"⚠️  Landmark detector failed: {e}")
                print("   Continuing without landmarks...")
                self.use_landmarks = False
    
    @torch.no_grad()
    def extract_identity_embeddings(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract identity embeddings using FaceNet
        
        Args:
            images: Batch of images [-1, 1] normalized
            
        Returns:
            Embeddings array (N, 512)
        """
        # Denormalize to [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        
        embeddings = []
        
        for img in images:
            # Convert to PIL
            img_pil = Image.fromarray(
                (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            
            # Detect and crop face
            face_crop = self.mtcnn(img_pil)
            
            if face_crop is None:
                # If detection fails, use zero embedding
                embeddings.append(np.zeros(512))
                continue
            
            # Get embedding
            face_crop = face_crop.unsqueeze(0).to(self.device)
            embedding = self.facenet(face_crop).cpu().numpy()
            embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    @torch.no_grad()
    def extract_landmarks(self, images: torch.Tensor) -> List[np.ndarray]:
        """
        Extract facial landmarks
        
        Args:
            images: Batch of images [-1, 1] normalized
            
        Returns:
            List of landmark arrays (68, 2)
        """
        if not self.use_landmarks or self.fa is None:
            return [None] * len(images)
        
        # Denormalize
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        
        landmarks_list = []
        
        for img in images:
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            try:
                landmarks = self.fa.get_landmarks(img_np)
                if landmarks is not None and len(landmarks) > 0:
                    landmarks_list.append(landmarks[0])
                else:
                    landmarks_list.append(None)
            except Exception:
                landmarks_list.append(None)
        
        return landmarks_list
    
    def calculate_identity_diversity(self, embeddings: np.ndarray) -> dict:
        """
        Calculate identity diversity metrics
        
        Args:
            embeddings: Identity embeddings (N, 512)
            
        Returns:
            Dictionary of diversity metrics
        """
        # Remove zero embeddings (failed detections)
        valid_embeddings = embeddings[~np.all(embeddings == 0, axis=1)]
        
        if len(valid_embeddings) < 2:
            return {
                'mean_identity_distance': 0.0,
                'std_identity_distance': 0.0,
                'detection_rate': len(valid_embeddings) / len(embeddings)
            }
        
        # Pairwise cosine distances
        from scipy.spatial.distance import pdist, squareform
        
        # Normalize embeddings
        norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
        normalized = valid_embeddings / (norms + 1e-8)
        
        # Cosine similarity
        similarities = normalized @ normalized.T
        
        # Convert to distances (1 - similarity)
        distances = 1 - similarities
        
        # Get upper triangle (exclude diagonal)
        triu_indices = np.triu_indices_from(distances, k=1)
        pairwise_distances = distances[triu_indices]
        
        return {
            'mean_identity_distance': float(np.mean(pairwise_distances)),
            'std_identity_distance': float(np.std(pairwise_distances)),
            'min_identity_distance': float(np.min(pairwise_distances)),
            'max_identity_distance': float(np.max(pairwise_distances)),
            'detection_rate': float(len(valid_embeddings) / len(embeddings)),
            'num_valid_faces': int(len(valid_embeddings))
        }
    
    def calculate_landmark_variance(self, landmarks_list: List[np.ndarray]) -> dict:
        """
        Calculate landmark position variance
        
        Args:
            landmarks_list: List of landmark arrays
            
        Returns:
            Dictionary of landmark metrics
        """
        # Filter valid landmarks
        valid_landmarks = [lm for lm in landmarks_list if lm is not None]
        
        if len(valid_landmarks) < 2:
            return {
                'landmark_variance': 0.0,
                'detection_rate': len(valid_landmarks) / len(landmarks_list)
            }
        
        landmarks_array = np.array(valid_landmarks)  # (N, 68, 2)
        
        # Calculate variance across samples for each landmark
        variances = np.var(landmarks_array, axis=0)  # (68, 2)
        mean_variance = np.mean(variances)
        
        # Eye distance variance (measure of face size diversity)
        left_eye = landmarks_array[:, 36:42, :].mean(axis=1)  # (N, 2)
        right_eye = landmarks_array[:, 42:48, :].mean(axis=1)  # (N, 2)
        eye_distances = np.linalg.norm(right_eye - left_eye, axis=1)
        eye_distance_std = np.std(eye_distances)
        
        return {
            'landmark_variance': float(mean_variance),
            'eye_distance_std': float(eye_distance_std),
            'detection_rate': float(len(valid_landmarks) / len(landmarks_list)),
            'num_valid_landmarks': int(len(valid_landmarks))
        }
    
    def calculate_all_face_metrics(self,
                                   generator,
                                   num_samples: int = 1000,
                                   batch_size: int = 16) -> dict:
        """
        Calculate all face-specific metrics
        
        Args:
            generator: Generator model
            num_samples: Number of samples
            batch_size: Batch size for generation
            
        Returns:
            Complete face metrics dictionary
        """
        print("\nCalculating face-specific metrics...")
        generator.eval()
        
        all_embeddings = []
        all_landmarks = []
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for _ in tqdm(range(num_batches), desc="Processing faces"):
            # Generate images
            z = torch.randn(batch_size, generator.z_dim, device=self.device)
            with torch.no_grad():
                fake_imgs = generator(z, truncation_psi=0.7)
            
            # Extract embeddings
            embeddings = self.extract_identity_embeddings(fake_imgs)
            all_embeddings.append(embeddings)
            
            # Extract landmarks (if enabled)
            if self.use_landmarks:
                landmarks = self.extract_landmarks(fake_imgs)
                all_landmarks.extend(landmarks)
        
        # Combine all embeddings
        all_embeddings = np.concatenate(all_embeddings, axis=0)[:num_samples]
        all_landmarks = all_landmarks[:num_samples]
        
        # Calculate metrics
        print("\nCalculating identity diversity...")
        identity_metrics = self.calculate_identity_diversity(all_embeddings)
        
        results = {**identity_metrics}
        
        if self.use_landmarks and len(all_landmarks) > 0:
            print("Calculating landmark variance...")
            landmark_metrics = self.calculate_landmark_variance(all_landmarks)
            results.update(landmark_metrics)
        
        # Print summary
        print("\n" + "="*60)
        print("FACE-SPECIFIC METRICS")
        print("="*60)
        print(f"Face Detection Rate:      {results['detection_rate']:.2%}")
        print(f"Identity Distance (mean): {results['mean_identity_distance']:.4f}")
        print(f"Identity Distance (std):  {results['std_identity_distance']:.4f}")
        if 'landmark_variance' in results:
            print(f"Landmark Variance:        {results['landmark_variance']:.2f}")
            print(f"Eye Distance Std:         {results['eye_distance_std']:.2f}")
        print("="*60)
        
        return results

def test_face_metrics():
    """Test face metrics with sample data"""
    print("Testing face metrics calculator...")
    
    # Create dummy embeddings
    embeddings = np.random.randn(100, 512)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    calc = FaceMetricsCalculator(use_landmarks=False)
    
    print("\n1. Testing identity diversity...")
    identity_metrics = calc.calculate_identity_diversity(embeddings)
    print(f"✓ Mean distance: {identity_metrics['mean_identity_distance']:.4f}")
    print(f"✓ Detection rate: {identity_metrics['detection_rate']:.2%}")
    
    print("\n✓ Face metrics module working!")
    print("  (Landmarks disabled - enable with use_landmarks=True when online)")

if __name__ == "__main__":
    test_face_metrics()

    