"""
StyleGAN2-ADA Trainer with ADA, R1 regularization, mixed precision
"""
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image, make_grid
import wandb
from typing import Optional


class StyleGAN2Trainer:
    """Complete training pipeline for StyleGAN2-ADA"""

    def __init__(
        self,
        generator,
        discriminator,
        dataloader,
        config: dict,
        device: str = "cuda",
    ):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.dataloader = dataloader
        self.config = config
        self.device = device

        # Optimizers
        g_lr = config["training"]["lr_g"]
        d_lr = config["training"]["lr_d"]
        betas = (config["training"]["beta1"], config["training"]["beta2"])

        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=g_lr, betas=betas
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=d_lr, betas=betas
        )

        # Mixed precision
        self.use_amp = config["training"]["use_amp"]
        self.scaler_G = GradScaler(enabled=self.use_amp)
        self.scaler_D = GradScaler(enabled=self.use_amp)

        # ADA augmentation
        self.ada_target = config["training"]["ada_target"]
        self.ada_interval = config["training"]["ada_interval"]
        self.aug_p = 0.0

        # Regularization
        self.r1_gamma = config["training"]["r1_gamma"]
        self.pl_weight = config["training"]["pl_weight"]

        # Tracking
        self.iteration = 0

        # Paths (SAVE DIRECTLY TO GOOGLE DRIVE)
        self.checkpoint_dir = Path("/content/drive/MyDrive/stylegan_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.sample_dir = Path("/content/drive/MyDrive/stylegan_samples")
        self.sample_dir.mkdir(parents=True, exist_ok=True)


        # W&B logging
        if config["logging"]["use_wandb"]:
            wandb.init(
                project=config["logging"]["wandb_project"],
                config=config,
            )

        # Fixed latents
        self.fixed_z = torch.randn(16, self.G.z_dim, device=device)

    # ----------------------------------------------------------
    # TRAIN LOOP
    # ----------------------------------------------------------

    def train(self, num_iterations: int):
        print(f"Starting training for {num_iterations} iterations")

        data_iter = iter(self.dataloader)
        pbar = tqdm(range(num_iterations), desc="Training")

        for iteration in pbar:
            self.iteration = iteration

            try:
                real_imgs = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                real_imgs = next(data_iter)

            real_imgs = real_imgs.to(self.device)

            d_loss, d_real, d_fake = self.train_discriminator(real_imgs)
            g_loss = self.train_generator()

            if iteration % self.ada_interval == 0:
                self.update_ada(d_real)

            if iteration % self.config["logging"]["log_interval"] == 0:
                pbar.set_postfix(
                    {
                        "G": f"{g_loss:.3f}",
                        "D": f"{d_loss:.3f}",
                        "aug_p": f"{self.aug_p:.3f}",
                    }
                )

                if self.config["logging"]["use_wandb"]:
                    wandb.log(
                        {
                            "g_loss": g_loss,
                            "d_loss": d_loss,
                            "d_real": d_real.mean().item(),
                            "d_fake": d_fake.mean().item(),
                            "aug_p": self.aug_p,
                        },
                        step=iteration,
                    )

            if iteration % self.config["logging"]["sample_interval"] == 0:
                self.generate_samples(iteration)

            if iteration % self.config["checkpoint"]["save_interval"] == 0:
                self.save_checkpoint(iteration)

        print("Training complete!")

    # ----------------------------------------------------------
    # DISCRIMINATOR (FIXED)
    # ----------------------------------------------------------

    def train_discriminator(self, real_imgs):
        self.D.train()
        self.G.eval()

        batch_size = real_imgs.shape[0]
        aug_real = self.apply_augmentation(real_imgs)

        # Main loss (AMP)
        with autocast(enabled=self.use_amp):
            real_scores = self.D(aug_real)

            z = torch.randn(batch_size, self.G.z_dim, device=self.device)
            with torch.no_grad():
                fake_imgs = self.G(z)

            aug_fake = self.apply_augmentation(fake_imgs)
            fake_scores = self.D(aug_fake)

            d_loss = (
                F.softplus(fake_scores).mean()
                + F.softplus(-real_scores).mean()
            )

        # R1 regularization (lazy, FP32)
        if self.iteration % 16 == 0:
            aug_real = aug_real.detach().requires_grad_(True)

            with autocast(enabled=False):
                real_scores_grad = self.D(aug_real)
                r1_grads = torch.autograd.grad(
                    outputs=real_scores_grad.sum(),
                    inputs=aug_real,
                    create_graph=True,
                    only_inputs=True,
                )[0]

                r1_penalty = r1_grads.pow(2).sum([1, 2, 3]).mean()

            d_loss = d_loss + r1_penalty * (self.r1_gamma / 2)

        # Backward (AMP-safe + clipping)
        self.optimizer_D.zero_grad()

        self.scaler_D.scale(d_loss).backward()
        self.scaler_D.unscale_(self.optimizer_D)
        torch.nn.utils.clip_grad_norm_(
            self.D.parameters(), max_norm=1.0
        )

        self.scaler_D.step(self.optimizer_D)
        self.scaler_D.update()

        return d_loss.item(), real_scores.detach(), fake_scores.detach()

    # ----------------------------------------------------------
    # GENERATOR (FIXED)
    # ----------------------------------------------------------

    def train_generator(self):
        self.G.train()
        self.D.eval()

        batch_size = self.config["training"]["batch_size"]

        with autocast(enabled=self.use_amp):
            z = torch.randn(batch_size, self.G.z_dim, device=self.device)
            fake_imgs = self.G(z)

            aug_fake = self.apply_augmentation(fake_imgs)
            fake_scores = self.D(aug_fake)

            g_loss = F.softplus(-fake_scores).mean()

        self.optimizer_G.zero_grad()

        self.scaler_G.scale(g_loss).backward()
        self.scaler_G.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(
            self.G.parameters(), max_norm=1.0
        )

        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()

        return g_loss.item()

    # ----------------------------------------------------------
    # ADA
    # ----------------------------------------------------------

    def apply_augmentation(self, imgs):
        if self.aug_p == 0:
            return imgs

        if torch.rand(1).item() < self.aug_p:
            imgs = torch.flip(imgs, dims=[3])

        if torch.rand(1).item() < self.aug_p * 0.5:
            factor = 1 + (torch.rand(1).item() - 0.5) * 0.2
            imgs = imgs * factor

        return imgs

    def update_ada(self, real_scores):
        signs = torch.sign(real_scores).detach()
        adjust = signs.mean().item()

        if adjust > self.ada_target:
            self.aug_p = min(self.aug_p + 0.001, 1.0)
        else:
            self.aug_p = max(self.aug_p - 0.001, 0.0)

    # ----------------------------------------------------------
    # UTILS
    # ----------------------------------------------------------

    @torch.no_grad()
    def generate_samples(self, iteration):
        self.G.eval()

        fake_imgs = self.G(self.fixed_z, truncation_psi=0.7)
        fake_imgs = (fake_imgs + 1) / 2
        fake_imgs = fake_imgs.clamp(0, 1)

        grid = make_grid(fake_imgs, nrow=4, padding=2)
        save_path = self.sample_dir / f"sample_{iteration:06d}.png"
        save_image(grid, save_path)

        if self.config["logging"]["use_wandb"]:
            wandb.log(
                {"samples": wandb.Image(str(save_path))},
                step=iteration,
            )

    def save_checkpoint(self, iteration):
        checkpoint = {
            "iteration": iteration,
            "generator": self.G.state_dict(),
            "discriminator": self.D.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "aug_p": self.aug_p,
            "config": self.config,
        }

        path = self.checkpoint_dir / f"checkpoint_{iteration:06d}.pt"
        torch.save(checkpoint, path)

        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt")
        )
        if len(checkpoints) > self.config["checkpoint"]["keep_last_n"]:
            checkpoints[0].unlink()

        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.G.load_state_dict(checkpoint["generator"])
        self.D.load_state_dict(checkpoint["discriminator"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        self.iteration = checkpoint["iteration"]
        self.aug_p = checkpoint.get("aug_p", 0.0)

        print(f"Loaded checkpoint from iteration {self.iteration}")
