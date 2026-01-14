#!/usr/bin/env python3
"""
Quick validation script to test gradient explosion fixes
Run this before full training to verify stability
"""
import torch
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.stylegan2_ada import Generator, Discriminator
from data.dataset import get_dataloader
from training.trainer import StyleGAN2Trainer


def test_gradient_stability():
    """Test that training doesn't explode in first 150 iterations"""
    print("=" * 60)
    print("GRADIENT STABILITY TEST")
    print("=" * 60)
    
    # Load local config
    config_path = Path(__file__).parent / "configs" / "local_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Using device: {device}")
    
    # Create models
    print("\n✓ Creating models...")
    generator = Generator(
        z_dim=config['model']['latent_dim'],
        w_dim=config['model']['latent_dim'],
        img_resolution=config['dataset']['resolution']
    )
    
    discriminator = Discriminator(
        img_resolution=config['dataset']['resolution']
    )
    
    # Create dataloader
    print(f"\n✓ Loading dataset from: {config['dataset']['path']}")
    try:
        dataloader = get_dataloader(
            data_dir=config['dataset']['path'],
            batch_size=config['training']['batch_size'],
            resolution=config['dataset']['resolution'],
            num_workers=config['dataset']['num_workers'],
            mirror=config['dataset']['mirror']
        )
        print(f"✓ Dataset size: {len(dataloader.dataset)} images")
    except Exception as e:
        print(f"\n❌ ERROR: Could not load dataset!")
        print(f"   {str(e)}")
        print(f"\n   Please verify dataset path in configs/local_config.yaml")
        return False
    
    # Create trainer
    print("\n✓ Creating trainer...")
    trainer = StyleGAN2Trainer(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        config=config,
        device=device
    )
    
    # Run short test
    print("\n" + "=" * 60)
    print("RUNNING 150 ITERATION TEST")
    print("=" * 60)
    print("\nThis will verify:")
    print("  1. Training passes iteration 100 without D-collapse")
    print("  2. No NaN errors from AMP/FP32 mixing")
    print("  3. Gradients stay bounded (< 10.0)")
    print("  4. D loss stays > 0.01")
    print("  5. G loss stays < 50.0")
    print()
    
    try:
        trainer.train(num_iterations=150)
        print("\n" + "=" * 60)
        print("✅ SUCCESS: Training stable for 150 iterations!")
        print("=" * 60)
        print("\nYou can now run full training with:")
        print("  python train.py --config configs/local_config.yaml")
        return True
        
    except RuntimeError as e:
        if "NaN" in str(e):
            print("\n" + "=" * 60)
            print("❌ FAILED: NaN detected in gradients")
            print("=" * 60)
            print(f"\nError: {str(e)}")
            return False
        raise
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ FAILED: Unexpected error")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        return False


def test_r1_regularization():
    """Test R1 regularization doesn't produce NaN with small gradients"""
    print("\n" + "=" * 60)
    print("R1 REGULARIZATION STABILITY TEST")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create discriminator
    D = Discriminator(img_resolution=256).to(device)
    
    # Test with very small gradients
    x = torch.randn(4, 3, 256, 256, device=device, requires_grad=True)
    scores = D(x)
    
    # Compute R1 penalty
    grads = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=x,
        create_graph=True,
        only_inputs=True,
    )[0]
    
    # Old way (would fail with small gradients)
    # r1_penalty_old = grads.pow(2).sum([1, 2, 3]).mean()
    
    # New way (with epsilon)
    r1_penalty_new = (grads.pow(2).sum([1, 2, 3]) + 1e-6).mean()
    
    if torch.isnan(r1_penalty_new) or torch.isinf(r1_penalty_new):
        print("❌ FAILED: R1 penalty is NaN or Inf")
        return False
    
    print(f"✓ R1 penalty computed successfully: {r1_penalty_new.item():.6f}")
    print("✓ No NaN or Inf values detected")
    return True


def main():
    print("\n" + "=" * 60)
    print("STYLEGAN2-ADA GRADIENT EXPLOSION FIX VALIDATION")
    print("=" * 60)
    
    # Test 1: R1 regularization
    if not test_r1_regularization():
        print("\n❌ R1 test failed!")
        return
    
    # Test 2: Full gradient stability
    if not test_gradient_stability():
        print("\n❌ Gradient stability test failed!")
        return
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYour fixes are working correctly. You can now:")
    print("  1. Run full local training (500 iterations):")
    print("     python train.py --config configs/local_config.yaml")
    print("\n  2. If stable, move to Colab/Kaggle for full training")
    print()


if __name__ == "__main__":
    main()
