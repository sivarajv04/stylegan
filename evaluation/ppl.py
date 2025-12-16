"""
Perceptual Path Length (PPL) metric for latent space quality
"""
import torch
import lpips
import numpy as np
from tqdm import tqdm

class PPLCalculator:
    """Calculate Perceptual Path Length"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        # LPIPS perceptual loss
        self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_fn.eval()
    
    @torch.no_grad()
    def calculate_ppl(self, 
                     generator,
                     num_samples: int = 5000,
                     epsilon: float = 1e-4,
                     space: str = 'w') -> float:
        """
        Calculate PPL by measuring perceptual distance along latent paths
        
        Args:
            generator: Generator model
            num_samples: Number of samples for calculation
            epsilon: Step size in latent space
            space: 'z' or 'w' latent space
            
        Returns:
            PPL value (lower = smoother latent space)
        """
        generator.eval()
        distances = []
        
        for _ in tqdm(range(num_samples), desc="Calculating PPL"):
            # Sample two random points
            z1 = torch.randn(1, generator.z_dim, device=self.device)
            z2 = torch.randn(1, generator.z_dim, device=self.device)
            
            # Map to W space if needed
            if space == 'w':
                w1 = generator.mapping(z1)
                w2 = generator.mapping(z2)
            else:
                w1, w2 = z1, z2
            
            # Interpolate
            t = torch.rand(1, device=self.device)
            w_interp = w1 * (1 - t) + w2 * t
            
            # Small step
            w_step = w_interp + epsilon * (w2 - w1)
            
            # Generate images
            if space == 'w':
                # Temporarily replace mapping output
                img1 = self._generate_from_w(generator, w_interp)
                img2 = self._generate_from_w(generator, w_step)
            else:
                img1 = generator(w_interp)
                img2 = generator(w_step)
            
            # Normalize to [-1, 1] for LPIPS
            img1 = (img1 + 1) / 2
            img2 = (img2 + 1) / 2
            
            # Calculate perceptual distance
            dist = self.lpips_fn(img1, img2).item()
            distances.append(dist / epsilon)
        
        ppl = np.mean(distances)
        return float(ppl)
    
    def _generate_from_w(self, generator, w):
        """Generate image directly from W latent"""
        # Start from constant
        x = generator.const.repeat(w.shape[0], 1, 1, 1)
        rgb = None
        
        # Progressive synthesis with given w
        for block in generator.blocks:
            x, rgb = block(x, w, rgb)
        
        return rgb

def calculate_diversity_metrics(features: np.ndarray) -> dict:
    """
    Calculate diversity metrics from feature embeddings
    
    Args:
        features: Feature embeddings (N, D)
        
    Returns:
        Dictionary of diversity metrics
    """
    # Pairwise distances
    from scipy.spatial.distance import pdist
    
    distances = pdist(features, metric='euclidean')
    
    # Mode collapse indicators
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    min_dist = np.min(distances)
    
    # Coefficient of variation (low = mode collapse)
    cv = std_dist / (mean_dist + 1e-8)
    
    # Effective number of modes (entropy-based)
    # Approximate using distance distribution
    hist, _ = np.histogram(distances, bins=50, density=True)
    hist = hist + 1e-10
    entropy = -np.sum(hist * np.log(hist))
    
    return {
        'mean_distance': float(mean_dist),
        'std_distance': float(std_dist),
        'min_distance': float(min_dist),
        'coefficient_variation': float(cv),
        'entropy': float(entropy),
        'mode_collapse_score': float(1.0 / (cv + 1e-8))  # High = collapse
    }

def test_ppl():
    """Test PPL calculation"""
    print("Testing PPL calculator...")
    
    # Mock generator for testing
    class MockGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.z_dim = 512
            self.const = torch.nn.Parameter(torch.randn(1, 512, 4, 4))
            self.mapping = torch.nn.Linear(512, 512)
            self.blocks = torch.nn.ModuleList([
                torch.nn.Conv2d(512, 3, 1)
            ])
        
        def forward(self, z):
            return torch.randn(z.shape[0], 3, 256, 256)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = MockGenerator().to(device)
    
    calculator = PPLCalculator(device=device)
    ppl = calculator.calculate_ppl(gen, num_samples=10)
    
    print(f"✓ PPL: {ppl:.2f}")
    print("✓ PPL module working!")

if __name__ == "__main__":
    test_ppl()