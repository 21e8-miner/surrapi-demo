#!/usr/bin/env python3
"""
Generate demo FNO weights for testing.
Creates realistic-looking (but not trained) model weights.

Usage:
    python scripts/generate_demo_weights.py
    
This creates app/assets/fno_128.pt with random weights.
For production, replace with properly trained weights.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import FNO2D


def generate_demo_weights(output_path: str = "app/assets/fno_128.pt"):
    """Generate and save demo weights."""
    
    print("🧠 Generating demo FNO weights...")
    
    # Create model
    model = FNO2D(
        modes=32,
        width=64,
        in_channels=3,
        out_channels=3,
        num_layers=4,
        padding=8
    )
    
    # Initialize with Xavier/Glorot for better demo outputs
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size:           {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Save weights
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"✓ Saved to {output_path} ({file_size:.1f} MB)")
    
    # Test load
    print("\n🔬 Verifying weights...")
    model2 = FNO2D(modes=32, width=64, in_channels=3, out_channels=3)
    model2.load_state_dict(torch.load(output_path, weights_only=True))
    model2.eval()
    
    # Test inference
    with torch.no_grad():
        x = torch.randn(1, 3, 128, 128)
        y = model2(x)
    
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Output range: [{y.min():.3f}, {y.max():.3f}]")
    print("\n✅ Demo weights generated successfully!")
    print("\n⚠️  NOTE: These are random weights for demo purposes.")
    print("   For production, train on actual physics data.")


if __name__ == "__main__":
    generate_demo_weights()
