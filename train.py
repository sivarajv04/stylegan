#!/usr/bin/env python3
"""
Main training script for StyleGAN2-ADA
"""
import argparse
import yaml
from pathlib import Path
import torch

from models.stylegan2_ada import Generator, Discriminator
from data.dataset import get_dataloader
from training.trainer import StyleGAN2Trainer

def main():
    parser = argparse.ArgumentParser(description='Train StyleGAN2-ADA')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--data', type=str, default=None,
                       help='Override dataset path from config')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Override checkpoint save directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory for samples/logs')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override paths if provided via command line
    if args.data:
        config['dataset']['path'] = args.data
        print(f"Dataset path overridden to: {args.data}")
    
    if args.checkpoint_dir:
        config['checkpoint']['save_path'] = args.checkpoint_dir
        print(f"Checkpoint directory overridden to: {args.checkpoint_dir}")
    
    if args.output_dir:
        # Create output directory structure
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        config['output_dir'] = args.output_dir
        print(f"Output directory set to: {args.output_dir}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create models
    print("Creating models...")
    generator = Generator(
        z_dim=config['model']['latent_dim'],
        w_dim=config['model']['latent_dim'],
        img_resolution=config['dataset']['resolution']
    )
    
    discriminator = Discriminator(
        img_resolution=config['dataset']['resolution']
    )
    
    # Create dataloader
    print(f"Loading dataset from: {config['dataset']['path']}")
    dataloader = get_dataloader(
        data_dir=config['dataset']['path'],
        batch_size=config['training']['batch_size'],
        resolution=config['dataset']['resolution'],
        num_workers=config['dataset']['num_workers'],
        mirror=config['dataset']['mirror']
    )
    
    print(f"Dataset size: {len(dataloader.dataset)} images")
    
    # Create trainer
    trainer = StyleGAN2Trainer(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    num_iterations = config['training']['iterations']
    trainer.train(num_iterations)
    
    print("Training complete!")

if __name__ == "__main__":
    main()