#!/usr/bin/env python3
"""
Cloud Setup Test Script
Tests that the AWS instance is properly configured for training
without requiring the full MOSI dataset or SDK.
"""

import sys
import torch
import os
from pathlib import Path

def test_cuda():
    """Test CUDA availability and GPU detection."""
    print("=" * 60)
    print("CUDA Configuration Test")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total GPU Memory: {total_memory:.1f} GB")
        
        # Quick GPU computation test
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            print("✓ GPU Computation Test: PASSED")
        except Exception as e:
            print(f"✗ GPU Computation Test: FAILED - {e}")
    else:
        print("⚠ Running in CPU mode")
    
    return cuda_available

def test_imports():
    """Test that all required packages can be imported."""
    print("\n" + "=" * 60)
    print("Package Import Test")
    print("=" * 60)
    
    packages = [
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("torchaudio", "TorchAudio"),
        ("librosa", "Librosa"),
        ("matplotlib", "Matplotlib"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("pytest", "Pytest"),
        ("tqdm", "TQDM"),
        ("h5py", "H5PY"),
    ]
    
    failed = []
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"✓ {display_name:20} imported successfully")
        except ImportError as e:
            print(f"✗ {display_name:20} FAILED - {e}")
            failed.append(display_name)
    
    return len(failed) == 0

def test_encoders():
    """Test that encoder modules can be loaded."""
    print("\n" + "=" * 60)
    print("Encoder Module Test")
    print("=" * 60)
    
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    encoders_ok = True
    
    try:
        from Encoders import Text_to_Vec
        encoder = Text_to_Vec()
        print(f"✓ Text_to_Vec loaded successfully")
        del encoder  # Free memory
    except Exception as e:
        print(f"✗ Text_to_Vec failed: {e}")
        encoders_ok = False
    
    try:
        from Encoders import Audio_to_Vec
        encoder = Audio_to_Vec()
        print(f"✓ Audio_to_Vec loaded successfully")
        del encoder
    except Exception as e:
        print(f"✗ Audio_to_Vec failed: {e}")
        encoders_ok = False
    
    # Note: Image encoder might fail due to PyTorch version
    try:
        from Encoders import Image_to_Vec
        encoder = Image_to_Vec()
        print(f"✓ Image_to_Vec loaded successfully")
        del encoder
    except Exception as e:
        print(f"⚠ Image_to_Vec failed (may need PyTorch 2.6+): {e}")
    
    return encoders_ok

def test_data_directories():
    """Check if data directories exist and contain files."""
    print("\n" + "=" * 60)
    print("Data Directory Test")
    print("=" * 60)
    
    data_path = Path("data/cmumosi")
    
    if not data_path.exists():
        print(f"✗ Data directory not found: {data_path}")
        return False
    
    subdirs = ["audio", "frames", "mosi"]
    all_ok = True
    
    for subdir in subdirs:
        subpath = data_path / subdir
        if subpath.exists():
            file_count = len(list(subpath.iterdir()))
            print(f"✓ {subdir:10} directory: {file_count} files")
        else:
            print(f"✗ {subdir:10} directory: NOT FOUND")
            all_ok = False
    
    return all_ok

def test_memory():
    """Test system memory availability."""
    print("\n" + "=" * 60)
    print("System Memory Test")
    print("=" * 60)
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"Total RAM: {mem.total / 1e9:.1f} GB")
        print(f"Available RAM: {mem.available / 1e9:.1f} GB")
        print(f"Used RAM: {mem.percent:.1f}%")
    except ImportError:
        # Fallback if psutil not installed
        import os
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                for line in lines[:3]:
                    print(line.strip())
        except:
            print("⚠ Could not read memory info")
    
    return True

def main():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("AWS Instance Setup Verification")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Run tests
    results = {
        "CUDA": test_cuda(),
        "Imports": test_imports(),
        "Encoders": test_encoders(),
        "Data": test_data_directories(),
        "Memory": test_memory(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:15} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ Instance is ready for training!")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())