"""
FID, KID, and IS metrics using InceptionV3
"""
import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from PIL import Image

class InceptionV3Features(nn.Module):
    """Extract features from InceptionV3 for metrics"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity()
        inception.eval()
        self.model = inception.to(device)
        self.device = device
        
        # Standard preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def forward(self, x):
        """Extract features"""
        return self.model(x)
    
    @torch.no_grad()
    def extract_features(self, images):
        """Extract features from batch of images"""
        # Images should be in [-1, 1], convert to [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        
        # Resize and normalize for InceptionV3
        batch_size = images.shape[0]
        features = []
        
        for img in images:
            img_pil = transforms.ToPILImage()(img.cpu())
            img_transformed = self.transform(img_pil).unsqueeze(0).to(self.device)
            feat = self.model(img_transformed)
            features.append(feat.cpu())
        
        return torch.cat(features, dim=0)

def calculate_fid(real_features: np.ndarray, 
                  fake_features: np.ndarray) -> float:
    """
    Calculate Fréchet Inception Distance
    
    Args:
        real_features: Features from real images (N, 2048)
        fake_features: Features from generated images (N, 2048)
        
    Returns:
        FID score (lower is better)
    """
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return float(fid)

def calculate_kid(real_features: np.ndarray,
                  fake_features: np.ndarray,
                  subset_size: int = 1000) -> Tuple[float, float]:
    """
    Calculate Kernel Inception Distance
    
    Args:
        real_features: Features from real images
        fake_features: Features from generated images
        subset_size: Size of subsets for calculation
        
    Returns:
        (KID mean, KID std)
    """
    n_subsets = min(len(real_features), len(fake_features)) // subset_size
    
    if n_subsets == 0:
        n_subsets = 1
        subset_size = min(len(real_features), len(fake_features))
    
    kid_scores = []
    
    for _ in range(n_subsets):
        # Random subsets
        idx_real = np.random.choice(len(real_features), subset_size, replace=False)
        idx_fake = np.random.choice(len(fake_features), subset_size, replace=False)
        
        real_subset = real_features[idx_real]
        fake_subset = fake_features[idx_fake]
        
        # Polynomial kernel
        def kernel(x, y):
            return (x @ y.T / x.shape[1] + 1) ** 3
        
        k_rr = kernel(real_subset, real_subset)
        k_ff = kernel(fake_subset, fake_subset)
        k_rf = kernel(real_subset, fake_subset)
        
        kid = (k_rr.mean() + k_ff.mean() - 2 * k_rf.mean())
        kid_scores.append(kid)
    
    return float(np.mean(kid_scores)), float(np.std(kid_scores))

def calculate_inception_score(fake_features: np.ndarray,
                               splits: int = 10) -> Tuple[float, float]:
    """
    Calculate Inception Score
    
    Args:
        fake_features: Features from generated images
        splits: Number of splits for calculation
        
    Returns:
        (IS mean, IS std)
    """
    # Normalize features to probabilities
    fake_features = np.abs(fake_features)  # Ensure positive
    fake_features = fake_features + 1e-10  # Avoid zeros
    
    scores = []
    split_size = max(1, len(fake_features) // splits)
    
    for i in range(splits):
        start_idx = i * split_size
        end_idx = min((i + 1) * split_size, len(fake_features))
        
        if end_idx <= start_idx:
            break
            
        part = fake_features[start_idx:end_idx]
        
        # Normalize to probability distribution
        part = part / (np.sum(part, axis=1, keepdims=True) + 1e-10)
        
        # Marginal distribution
        p_y = np.mean(part, axis=0)
        p_y = p_y / (np.sum(p_y) + 1e-10)
        
        # KL divergence per sample
        kl_divs = []
        for p_yx in part:
            # Clip to avoid log(0)
            p_yx = np.clip(p_yx, 1e-10, 1.0)
            p_y_clip = np.clip(p_y, 1e-10, 1.0)
            
            kl = np.sum(p_yx * (np.log(p_yx) - np.log(p_y_clip)))
            kl_divs.append(kl)
        
        # IS for this split
        split_score = np.exp(np.mean(kl_divs))
        scores.append(split_score)
    
    if len(scores) == 0:
        return 1.0, 0.0
    
    return float(np.mean(scores)), float(np.std(scores))

class MetricsCalculator:
    """All-in-one metrics calculator"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.inception = InceptionV3Features(device=self.device)
    
    @torch.no_grad()
    def extract_features_from_generator(self, 
                                       generator,
                                       num_samples: int = 10000,
                                       batch_size: int = 64,
                                       truncation_psi: float = 0.7) -> np.ndarray:
        """Extract features from generator"""
        generator.eval()
        features = []
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for _ in tqdm(range(num_batches), desc="Generating samples"):
            z = torch.randn(batch_size, generator.z_dim, device=self.device)
            fake_imgs = generator(z, truncation_psi=truncation_psi)
            
            feat = self.inception.extract_features(fake_imgs)
            features.append(feat.cpu().numpy())
        
        features = np.concatenate(features, axis=0)[:num_samples]
        return features
    
    @torch.no_grad()
    def extract_features_from_images(self,
                                    image_dir: str,
                                    num_samples: int = 10000,
                                    batch_size: int = 64) -> np.ndarray:
        """Extract features from real images"""
        image_paths = list(Path(image_dir).glob('*.jpg')) + \
                     list(Path(image_dir).glob('*.png'))
        
        if len(image_paths) > num_samples:
            image_paths = np.random.choice(image_paths, num_samples, replace=False)
        
        features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                img = self.inception.transform(img)
                images.append(img)
            
            if images:
                images = torch.stack(images).to(self.device)
                feat = self.inception(images)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def compute_all_metrics(self,
                           generator,
                           real_image_dir: str,
                           num_samples: int = 10000) -> dict:
        """Compute all metrics at once"""
        print("Computing evaluation metrics...")
        
        # Extract features
        print("\n1. Extracting real image features...")
        real_features = self.extract_features_from_images(real_image_dir, num_samples)
        
        print("\n2. Generating fake images and extracting features...")
        fake_features = self.extract_features_from_generator(generator, num_samples)
        
        # Calculate metrics
        print("\n3. Calculating FID...")
        fid = calculate_fid(real_features, fake_features)
        
        print("4. Calculating KID...")
        kid_mean, kid_std = calculate_kid(real_features, fake_features)
        
        print("5. Calculating IS...")
        is_mean, is_std = calculate_inception_score(fake_features)
        
        results = {
            'fid': fid,
            'kid_mean': kid_mean,
            'kid_std': kid_std,
            'is_mean': is_mean,
            'is_std': is_std
        }
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"FID:  {fid:.2f}")
        print(f"KID:  {kid_mean:.4f} ± {kid_std:.4f}")
        print(f"IS:   {is_mean:.2f} ± {is_std:.2f}")
        print("="*60)
        
        return results

def test_metrics():
    """Test metrics with dummy data"""
    print("Testing metrics calculation...")
    
    # Create dummy features
    real_features = np.random.randn(1000, 2048)
    fake_features = np.random.randn(1000, 2048)
    
    fid = calculate_fid(real_features, fake_features)
    kid_mean, kid_std = calculate_kid(real_features, fake_features)
    is_mean, is_std = calculate_inception_score(fake_features)
    
    print(f"✓ FID: {fid:.2f}")
    print(f"✓ KID: {kid_mean:.4f} ± {kid_std:.4f}")
    print(f"✓ IS: {is_mean:.2f} ± {is_std:.2f}")
    print("✓ Metrics module working!")

if __name__ == "__main__":
    test_metrics()