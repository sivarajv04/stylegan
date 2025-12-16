#!/usr/bin/env python3
"""
Complete evaluation script for all metrics
"""
import argparse
import torch
import json
from pathlib import Path

from models.stylegan2_ada import Generator
from evaluation.fid_kid_is import MetricsCalculator

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--real_data', type=str, required=True,
                       help='Path to real images for comparison')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples for evaluation')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create generator
    generator = Generator(
        z_dim=512,
        w_dim=512,
        img_resolution=256
    ).to(device)
    
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    print("✓ Generator loaded")
    
    # Calculate metrics
    calculator = MetricsCalculator(device=device)
    
    results = calculator.compute_all_metrics(
        generator=generator,
        real_image_dir=args.real_data,
        num_samples=args.num_samples
    )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'metrics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()