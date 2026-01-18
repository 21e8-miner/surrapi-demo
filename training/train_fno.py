#!/usr/bin/env python3
"""
SurrAPI Training Pipeline
========================

Train FNO model on The Well physics simulation dataset.
Supports distributed training on multi-GPU setups.

Usage:
    python training/train_fno.py --config training/configs/fno_base.yaml
    
    # Multi-GPU
    torchrun --nproc_per_node=8 training/train_fno.py --config training/configs/fno_large.yaml
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.model import FNO2D

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "model": {
        "modes": 32,
        "width": 64,
        "in_channels": 3,
        "out_channels": 3,
        "num_layers": 4,
        "padding": 8
    },
    "training": {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_epochs": 5,
        "gradient_clip": 1.0,
        "mixed_precision": True,
        "compile_model": True
    },
    "data": {
        "dataset_name": "navier_stokes_2d",
        "resolution": 128,
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "num_workers": 8,
        "prefetch_factor": 2
    },
    "loss": {
        "type": "relative_l2",
        "divergence_weight": 0.1,
        "boundary_weight": 0.05
    },
    "checkpointing": {
        "save_every": 5,
        "keep_last": 3,
        "output_dir": "checkpoints"
    },
    "logging": {
        "log_every": 100,
        "wandb_project": "surrapi-training",
        "wandb_entity": None
    }
}


# =============================================================================
# Dataset
# =============================================================================

class TheWellDataset(Dataset):
    """
    Dataset loader for The Well physics simulations.
    
    Supports streaming from HuggingFace or local files.
    Each sample contains:
        - input: (C, H, W) boundary conditions + parameters
        - target: (3, H, W) velocity (ux, uy) + pressure (p)
        - params: dict with Reynolds, Mach, angle, etc.
    """
    
    def __init__(
        self,
        dataset_name: str = "navier_stokes_2d",
        split: str = "train",
        resolution: int = 128,
        data_dir: Optional[str] = None,
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.resolution = resolution
        self.transform = transform
        self.max_samples = max_samples
        
        # Try to load from HuggingFace
        try:
            from datasets import load_dataset
            self.use_hf = True
            self.dataset = load_dataset(
                "polymathic-ai/the_well",
                name=dataset_name,
                split=split,
                streaming=True  # Stream for large datasets
            )
            logging.info(f"Loaded {dataset_name} from HuggingFace (streaming)")
        except Exception as e:
            logging.warning(f"HuggingFace unavailable: {e}")
            self.use_hf = False
            self._load_local(data_dir)
    
    def _load_local(self, data_dir: Optional[str]):
        """Load from local HDF5/NPZ files"""
        if data_dir is None:
            data_dir = os.environ.get("THEWELL_DATA_DIR", "data/the_well")
        
        self.data_dir = Path(data_dir) / self.dataset_name
        
        # Find all simulation files
        self.files = sorted(self.data_dir.glob(f"{self.split}*.npz"))
        if not self.files:
            self.files = sorted(self.data_dir.glob(f"{self.split}*.h5"))
        
        if not self.files:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")
        
        logging.info(f"Found {len(self.files)} local data files")
        
        # Build index
        self._build_index()
    
    def _build_index(self):
        """Build sample index across files"""
        self.index = []
        for file_path in self.files:
            if file_path.suffix == '.npz':
                data = np.load(file_path, mmap_mode='r')
                n_samples = data['velocity'].shape[0]
            else:  # h5
                import h5py
                with h5py.File(file_path, 'r') as f:
                    n_samples = f['velocity'].shape[0]
            
            for i in range(n_samples):
                self.index.append((file_path, i))
                if self.max_samples and len(self.index) >= self.max_samples:
                    return
    
    def __len__(self):
        if self.use_hf:
            return self.max_samples or 100000  # Streaming doesn't know length
        return len(self.index)
    
    def __getitem__(self, idx):
        if self.use_hf:
            return self._get_hf_item(idx)
        return self._get_local_item(idx)
    
    def _get_hf_item(self, idx):
        """Get item from HuggingFace streaming dataset"""
        sample = next(iter(self.dataset.skip(idx).take(1)))
        
        # Extract fields
        velocity = torch.tensor(sample['velocity'], dtype=torch.float32)
        pressure = torch.tensor(sample['pressure'], dtype=torch.float32)
        params = sample.get('parameters', {})
        
        # Build input (boundary conditions + parameters)
        input_tensor = self._build_input(velocity, params)
        
        # Build target
        target = torch.cat([velocity, pressure.unsqueeze(0)], dim=0)
        
        if self.transform:
            input_tensor, target = self.transform(input_tensor, target)
        
        return input_tensor, target, params
    
    def _get_local_item(self, idx):
        """Get item from local file"""
        file_path, sample_idx = self.index[idx]
        
        if file_path.suffix == '.npz':
            data = np.load(file_path)
            velocity = torch.tensor(data['velocity'][sample_idx], dtype=torch.float32)
            pressure = torch.tensor(data['pressure'][sample_idx], dtype=torch.float32)
            params = {k: data[k][sample_idx] for k in ['reynolds', 'mach', 'angle'] 
                     if k in data}
        else:
            import h5py
            with h5py.File(file_path, 'r') as f:
                velocity = torch.tensor(f['velocity'][sample_idx], dtype=torch.float32)
                pressure = torch.tensor(f['pressure'][sample_idx], dtype=torch.float32)
                params = {k: f[k][sample_idx] for k in ['reynolds', 'mach', 'angle']
                         if k in f}
        
        input_tensor = self._build_input(velocity, params)
        target = torch.cat([velocity, pressure.unsqueeze(0)], dim=0)
        
        if self.transform:
            input_tensor, target = self.transform(input_tensor, target)
        
        return input_tensor, target, params
    
    def _build_input(self, velocity: torch.Tensor, params: dict) -> torch.Tensor:
        """Build input tensor from velocity and parameters"""
        H, W = velocity.shape[-2:]
        
        # Normalize parameters
        re = params.get('reynolds', 2000) / 10000.0
        mach = params.get('mach', 0.2) / 0.6
        angle = params.get('angle', 0.0) / 15.0
        
        # Build parameter fields
        input_tensor = torch.zeros(3, H, W)
        input_tensor[0, :, :] = re
        input_tensor[1, :, :] = angle
        input_tensor[2, :, :] = mach
        
        # Add inlet profile on left boundary
        y = torch.linspace(-1, 1, H)
        inlet = 1 - y**2
        input_tensor[0, :, 0] += 0.5 * inlet
        
        return input_tensor


# =============================================================================
# Loss Functions
# =============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function combining:
    - Relative L2 error
    - Divergence-free constraint
    - Boundary condition matching
    """
    
    def __init__(
        self,
        divergence_weight: float = 0.1,
        boundary_weight: float = 0.05
    ):
        super().__init__()
        self.div_weight = divergence_weight
        self.bc_weight = boundary_weight
    
    def relative_l2(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Relative L2 loss"""
        diff = pred - target
        return torch.mean(diff**2) / (torch.mean(target**2) + 1e-8)
    
    def divergence_loss(self, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
        """Divergence-free constraint: ∇·u = 0"""
        # Central differences
        dux_dx = (ux[:, :, :, 2:] - ux[:, :, :, :-2]) / 2
        duy_dy = (uy[:, :, 2:, :] - uy[:, :, :-2, :]) / 2
        
        # Trim to match
        dux_dx = dux_dx[:, :, 1:-1, :]
        duy_dy = duy_dy[:, :, :, 1:-1]
        
        divergence = dux_dx + duy_dy
        return torch.mean(divergence**2)
    
    def boundary_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """No-slip boundary conditions"""
        ux, uy = pred[:, 0:1], pred[:, 1:2]
        
        # Walls at y=0, y=H
        loss = torch.mean(ux[:, :, 0, :]**2) + torch.mean(ux[:, :, -1, :]**2)
        loss += torch.mean(uy[:, :, 0, :]**2) + torch.mean(uy[:, :, -1, :]**2)
        
        # Outlet at x=W
        loss += torch.mean(uy[:, :, :, -1]**2)
        
        return loss
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss with breakdown"""
        # Data loss
        data_loss = self.relative_l2(pred, target)
        
        # Physics losses
        div_loss = self.divergence_loss(pred[:, 0], pred[:, 1])
        bc_loss = self.boundary_loss(pred)
        
        # Total
        total = data_loss + self.div_weight * div_loss + self.bc_weight * bc_loss
        
        return {
            "total": total,
            "data": data_loss,
            "divergence": div_loss,
            "boundary": bc_loss
        }


# =============================================================================
# Trainer
# =============================================================================

class FNOTrainer:
    """Training loop with mixed precision, gradient accumulation, etc."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup training
        self.epochs = config["training"]["epochs"]
        self.lr = config["training"]["learning_rate"]
        self.use_amp = config["training"]["mixed_precision"] and device == "cuda"
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=config["training"]["weight_decay"]
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs
        )
        
        # Loss
        self.criterion = PhysicsInformedLoss(
            divergence_weight=config["loss"]["divergence_weight"],
            boundary_weight=config["loss"]["boundary_weight"]
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Compile model (PyTorch 2.0+)
        if config["training"]["compile_model"] and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        
        # Checkpointing
        self.checkpoint_dir = Path(config["checkpointing"]["output_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_every = config["logging"]["log_every"]
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup wandb and file logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s"
        )
        self.logger = logging.getLogger("trainer")
        
        # Try wandb
        try:
            import wandb
            wandb.init(
                project=self.config["logging"]["wandb_project"],
                entity=self.config["logging"]["wandb_entity"],
                config=self.config
            )
            self.use_wandb = True
        except Exception:
            self.use_wandb = False
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        loss_breakdown = {"data": 0, "divergence": 0, "boundary": 0}
        num_batches = 0
        
        for batch_idx, (inputs, targets, params) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    losses = self.criterion(outputs, targets)
                
                self.scaler.scale(losses["total"]).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["gradient_clip"]
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                losses = self.criterion(outputs, targets)
                losses["total"].backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["gradient_clip"]
                )
                
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses["total"].item()
            for k in loss_breakdown:
                loss_breakdown[k] += losses[k].item()
            num_batches += 1
            
            # Log progress
            if batch_idx % self.log_every == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {losses['total'].item():.6f}"
                )
        
        # Average losses
        avg_loss = total_loss / num_batches
        for k in loss_breakdown:
            loss_breakdown[k] /= num_batches
        
        return {"total": avg_loss, **loss_breakdown}
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation pass"""
        self.model.eval()
        
        total_loss = 0
        total_l2_error = 0
        num_batches = 0
        
        for inputs, targets, params in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)
            
            # Relative L2 error
            l2_error = torch.sqrt(
                torch.mean((outputs - targets)**2) /
                (torch.mean(targets**2) + 1e-8)
            )
            
            total_loss += losses["total"].item()
            total_l2_error += l2_error.item()
            num_batches += 1
        
        return {
            "val_loss": total_loss / num_batches,
            "val_l2_error": total_l2_error / num_batches * 100  # Percentage
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config
        }
        
        path = self.checkpoint_dir / f"fno_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
        
        # Also save as latest
        torch.save(checkpoint, self.checkpoint_dir / "fno_latest.pt")
        
        # Export weights only (for inference)
        torch.save(
            self.model.state_dict(),
            self.checkpoint_dir / "fno_128.pt"
        )
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        best_val_loss = float("inf")
        
        for epoch in range(1, self.epochs + 1):
            self.logger.info(f"=== Epoch {epoch}/{self.epochs} ===")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            self.logger.info(
                f"Train Loss: {train_metrics['total']:.6f} | "
                f"Val Loss: {val_metrics['val_loss']:.6f} | "
                f"Val L2 Error: {val_metrics['val_l2_error']:.2f}%"
            )
            
            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **val_metrics,
                    "lr": self.optimizer.param_groups[0]["lr"]
                })
            
            # Checkpoint
            if epoch % self.config["checkpointing"]["save_every"] == 0:
                self.save_checkpoint(epoch, {**train_metrics, **val_metrics})
            
            # Best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_dir / "fno_best.pt"
                )
                self.logger.info(f"New best model! Val L2: {val_metrics['val_l2_error']:.2f}%")
        
        self.logger.info("Training complete!")
        return self.checkpoint_dir / "fno_best.pt"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train FNO on The Well dataset")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="navier_stokes_2d")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            config.update(yaml.safe_load(f))
    
    # Override from CLI
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    config["checkpointing"]["output_dir"] = args.output_dir
    config["data"]["dataset_name"] = args.dataset
    
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    logging.info(f"Using device: {device}")
    
    # Model
    model = FNO2D(
        modes=config["model"]["modes"],
        width=config["model"]["width"],
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        num_layers=config["model"]["num_layers"],
        padding=config["model"]["padding"]
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {param_count:,}")
    
    # Datasets
    train_dataset = TheWellDataset(
        dataset_name=config["data"]["dataset_name"],
        split="train",
        resolution=config["data"]["resolution"]
    )
    
    val_dataset = TheWellDataset(
        dataset_name=config["data"]["dataset_name"],
        split="validation",
        resolution=config["data"]["resolution"]
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    # Trainer
    trainer = FNOTrainer(model, config, device)
    
    # Train!
    best_model_path = trainer.train(train_loader, val_loader)
    
    logging.info(f"Best model saved to: {best_model_path}")
    logging.info("Copy to app/assets/fno_128.pt for production use")


if __name__ == "__main__":
    main()
