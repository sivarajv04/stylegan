"""
StyleGAN2-ADA Generator and Discriminator
Simplified but production-ready implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class MappingNetwork(nn.Module):
    """Maps latent z to intermediate latent w"""
    
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(nn.Linear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.net(z)

class ModulatedConv2d(nn.Module):
    """Weight modulation + demodulation convolution"""
    
    def __init__(self, in_ch, out_ch, kernel_size=3, demodulate=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.padding = kernel_size // 2
        
        # Learnable weights
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        
        # Affine transform for style modulation
        self.affine = nn.Linear(512, in_ch)
        
    def forward(self, x, w):
        B, C, H, W = x.shape
        
        # Modulation
        style = self.affine(w).view(B, 1, C, 1, 1)
        weight = self.weight.unsqueeze(0) * (style + 1)
        
        # Demodulation
        if self.demodulate:
            sigma = torch.rsqrt((weight ** 2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * sigma
        
        # Reshape for group conv
        weight = weight.view(B * self.out_ch, C, self.kernel_size, self.kernel_size)
        x = x.view(1, B * C, H, W)
        
        # Convolution
        out = F.conv2d(x, weight, padding=self.padding, groups=B)
        out = out.view(B, self.out_ch, H, W)
        out = out + self.bias.view(1, -1, 1, 1)
        
        return out

class ToRGB(nn.Module):
    """Convert features to RGB"""
    
    def __init__(self, in_ch, w_dim=512):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, 3, kernel_size=1, demodulate=False)
        
    def forward(self, x, w):
        return self.conv(x, w)

class SynthesisBlock(nn.Module):
    """Synthesis block with modulated convolutions"""
    
    def __init__(self, in_ch, out_ch, resolution):
        super().__init__()
        self.resolution = resolution
        
        self.conv1 = ModulatedConv2d(in_ch, out_ch, kernel_size=3)
        self.conv2 = ModulatedConv2d(out_ch, out_ch, kernel_size=3)
        self.to_rgb = ToRGB(out_ch)
        
        self.noise1 = nn.Parameter(torch.zeros(1, 1, resolution, resolution))
        self.noise2 = nn.Parameter(torch.zeros(1, 1, resolution, resolution))
        
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x, w, skip_rgb=None):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Conv 1
        x = self.conv1(x, w)
        x = x + self.noise1 * torch.randn_like(x)
        x = self.act(x)
        
        # Conv 2
        x = self.conv2(x, w)
        x = x + self.noise2 * torch.randn_like(x)
        x = self.act(x)
        
        # RGB output
        rgb = self.to_rgb(x, w)
        
        if skip_rgb is not None:
            skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='bilinear', align_corners=False)
            rgb = rgb + skip_rgb
            
        return x, rgb

class Generator(nn.Module):
    """StyleGAN2 Generator"""
    
    def __init__(self, z_dim=512, w_dim=512, img_resolution=256, img_channels=3):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim)
        
        # Starting constant
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # Synthesis blocks
        self.blocks = nn.ModuleList()
        channels = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64, 512: 32}
        
        resolutions = [4]
        while resolutions[-1] < img_resolution:
            resolutions.append(resolutions[-1] * 2)
        
        for i in range(len(resolutions) - 1):
            res = resolutions[i + 1]
            in_ch = channels[resolutions[i]]
            out_ch = channels[res]
            self.blocks.append(SynthesisBlock(in_ch, out_ch, res))
        
    def forward(self, z, truncation_psi=1.0):
        # Map to W space
        w = self.mapping(z)
        
        # Truncation trick
        if truncation_psi < 1.0:
            w = w * truncation_psi
        
        # Start from constant
        x = self.const.repeat(z.shape[0], 1, 1, 1)
        rgb = None
        
        # Progressive synthesis
        for block in self.blocks:
            x, rgb = block(x, w, rgb)
        
        return rgb

class DiscriminatorBlock(nn.Module):
    """Discriminator residual block"""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        skip = self.skip(F.avg_pool2d(x, 2))
        
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        
        return (x + skip) / np.sqrt(2)

class Discriminator(nn.Module):
    """StyleGAN2 Discriminator"""
    
    def __init__(self, img_resolution=256, img_channels=3):
        super().__init__()
        
        # From RGB
        self.from_rgb = nn.Conv2d(img_channels, 64, 1)
        
        # Progressive blocks
        channels = {256: 64, 128: 128, 64: 256, 32: 512, 16: 512, 8: 512, 4: 512}
        
        self.blocks = nn.ModuleList()
        resolutions = [256, 128, 64, 32, 16, 8, 4]
        
        for i in range(len(resolutions) - 1):
            res = resolutions[i]
            in_ch = channels[res]
            out_ch = channels[resolutions[i + 1]]
            self.blocks.append(DiscriminatorBlock(in_ch, out_ch))
        
        # Final layers
        self.final_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.final_linear = nn.Linear(512 * 4 * 4, 1)
        
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.act(self.from_rgb(x))
        
        for block in self.blocks:
            x = block(x)
        
        x = self.act(self.final_conv(x))
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        
        return x

class ADAugment(nn.Module):
    """Adaptive Discriminator Augmentation"""
    
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if self.training and self.p > 0:
            # Horizontal flip
            if torch.rand(1).item() < self.p:
                x = torch.flip(x, dims=[3])
            
            # Color jitter
            if torch.rand(1).item() < self.p:
                factor = torch.randn(x.shape[0], 3, 1, 1, device=x.device) * 0.1
                x = x * (1 + factor)
            
            # Cutout
            if torch.rand(1).item() < self.p * 0.5:
                h, w = x.shape[2:]
                mask = torch.ones_like(x)
                y1, x1 = torch.randint(0, h, (1,)), torch.randint(0, w, (1,))
                mask[:, :, y1:y1+h//4, x1:x1+w//4] = 0
                x = x * mask
        
        return x

def test_models():
    """Quick model test"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing StyleGAN2-ADA models...")
    
    # Generator test
    G = Generator(z_dim=512, img_resolution=256).to(device)
    z = torch.randn(2, 512).to(device)
    fake_img = G(z)
    print(f"✓ Generator: {fake_img.shape}")
    
    # Discriminator test
    D = Discriminator(img_resolution=256).to(device)
    score = D(fake_img)
    print(f"✓ Discriminator: {score.shape}")
    
    # Count parameters
    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"✓ G params: {g_params:.2f}M, D params: {d_params:.2f}M")
    
    return True

if __name__ == "__main__":
    test_models()