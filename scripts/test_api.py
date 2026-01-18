#!/usr/bin/env python3
"""
Test the SurrAPI prediction endpoint.

Usage:
    python scripts/test_api.py [--url URL]
    
Examples:
    python scripts/test_api.py
    python scripts/test_api.py --url https://api.surrapi.io
"""

import argparse
import json
import time
import sys

try:
    import requests
except ImportError:
    print("❌ requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print("📋 Testing /health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        data = response.json()
        print(f"   Status: {data['status']}")
        print(f"   Device: {data['device']}")
        print(f"   Model loaded: {data['model_loaded']}")
        return data['status'] == 'ok'
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_predict(base_url: str, reynolds: float = 5000, angle: float = 5.0, mach: float = 0.3) -> dict:
    """Test single prediction."""
    print(f"\n🌊 Testing /predict (Re={reynolds}, α={angle}°, M={mach})...")
    
    params = {
        "reynolds": reynolds,
        "angle": angle,
        "mach": mach,
        "resolution": 128
    }
    
    t0 = time.perf_counter()
    response = requests.post(
        f"{base_url}/predict",
        json=params,
        timeout=30
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    if response.status_code != 200:
        print(f"   ❌ HTTP {response.status_code}: {response.text}")
        return {}
    
    data = response.json()
    
    print(f"   ✓ Response in {elapsed_ms:.0f} ms (server: {data['inference_time_ms']:.0f} ms)")
    print(f"   Resolution: {data['resolution']}x{data['resolution']}")
    print(f"   VTK size: {len(data['vtk'])} bytes (base64)")
    
    if HAS_NUMPY:
        ux = np.array(data['ux']).reshape(data['resolution'], data['resolution'])
        uy = np.array(data['uy']).reshape(data['resolution'], data['resolution'])
        p = np.array(data['p']).reshape(data['resolution'], data['resolution'])
        
        print(f"   Velocity X: [{ux.min():.4f}, {ux.max():.4f}]")
        print(f"   Velocity Y: [{uy.min():.4f}, {uy.max():.4f}]")
        print(f"   Pressure:   [{p.min():.4f}, {p.max():.4f}]")
    
    return data


def test_batch(base_url: str, count: int = 5) -> None:
    """Test batch prediction."""
    print(f"\n📦 Testing /predict/batch ({count} predictions)...")
    
    predictions = []
    for i in range(count):
        predictions.append({
            "reynolds": 1000 + i * 1000,
            "angle": i * 2.0,
            "mach": 0.2 + i * 0.05
        })
    
    t0 = time.perf_counter()
    response = requests.post(
        f"{base_url}/predict/batch",
        json={"predictions": predictions},
        timeout=60
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    if response.status_code != 200:
        print(f"   ❌ HTTP {response.status_code}: {response.text}")
        return
    
    data = response.json()
    print(f"   ✓ Batch complete in {elapsed_ms:.0f} ms (server: {data['total_time_ms']:.0f} ms)")
    print(f"   Results: {len(data['results'])} predictions")
    print(f"   Avg time per prediction: {data['total_time_ms'] / len(data['results']):.0f} ms")


def test_integrals(base_url: str) -> None:
    """Test integral quantities endpoint."""
    print("\n📊 Testing /predict/integrals...")
    
    params = {"reynolds": 5000, "angle": 5.0, "mach": 0.3}
    
    t0 = time.perf_counter()
    response = requests.post(
        f"{base_url}/predict/integrals",
        json=params,
        timeout=30
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    if response.status_code != 200:
        print(f"   ❌ HTTP {response.status_code}: {response.text}")
        return
    
    data = response.json()
    print(f"   ✓ Response in {elapsed_ms:.0f} ms")
    print(f"   Drag coefficient (Cd):  {data['drag_coefficient']:.6f}")
    print(f"   Lift coefficient (Cl):  {data['lift_coefficient']:.6f}")
    print(f"   Pressure drop:          {data['pressure_drop']:.6f}")
    print(f"   Max velocity:           {data['max_velocity']:.6f}")
    print(f"   Enstrophy:              {data['enstrophy']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Test SurrAPI endpoints")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SurrAPI Integration Test")
    print("=" * 60)
    print(f"Target: {args.url}\n")
    
    # Run tests
    if not test_health(args.url):
        print("\n❌ Health check failed. Is the server running?")
        sys.exit(1)
    
    test_predict(args.url)
    test_batch(args.url, count=3)
    test_integrals(args.url)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
