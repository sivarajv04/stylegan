"""
Quick test script to verify current implementation
Run this to check if everything works before proceeding
"""

import sys
import torch
import yaml
from pathlib import Path

def test_dependencies():
    """Test if required packages are installed"""
    print("="*60)
    print("1. TESTING DEPENDENCIES")
    print("="*60)
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'face_alignment': 'Face Alignment',
        'facenet_pytorch': 'FaceNet PyTorch'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️ Install missing: pip install {' '.join(missing)}")
        return False
    return True

def test_gpu():
    """Test GPU availability"""
    print("\n" + "="*60)
    print("2. TESTING GPU")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA Available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("✗ No GPU found - Training will be VERY slow")
        print("  Recommend: Use Google Colab or Kaggle")
        return False

def test_config():
    """Test config file loading"""
    print("\n" + "="*60)
    print("3. TESTING CONFIG")
    print("="*60)
    
    try:
        # Create dummy config if doesn't exist
        config_path = Path("configs/training_config.yaml")
        if not config_path.exists():
            print("⚠️ Config file not found - create configs/ directory")
            return False
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        print("✓ Config loaded successfully")
        print(f"  Dataset: {config['dataset']['name']}")
        print(f"  Resolution: {config['dataset']['resolution']}")
        print(f"  Batch size: {config['training']['batch_size']}")
        return True
        
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_preprocessing():
    """Test preprocessing module"""
    print("\n" + "="*60)
    print("4. TESTING PREPROCESSING")
    print("="*60)
    
    try:
        # Import preprocessing module
        sys.path.insert(0, str(Path.cwd()))
        
        # Check if file exists
        prep_file = Path("data/preprocessing.py")
        if not prep_file.exists():
            print("⚠️ Create data/ directory and add preprocessing.py")
            return False
        
        print("✓ Preprocessing module found")
        print("  Ready to process images")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_sample_image():
    """Test with a sample image (if available)"""
    print("\n" + "="*60)
    print("5. TESTING SAMPLE IMAGE (Optional)")
    print("="*60)
    
    # Check if sample images exist
    data_dirs = [
        Path("data/raw"),
        Path("data/samples"),
        Path("samples")
    ]
    
    sample_found = False
    for data_dir in data_dirs:
        if data_dir.exists():
            images = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
            if images:
                print(f"✓ Found {len(images)} sample images in {data_dir}")
                sample_found = True
                break
    
    if not sample_found:
        print("⚠️ No sample images found")
        print("  Add test images to data/samples/ to test preprocessing")
    
    return True

def main():
    """Run all tests"""
    print("\n🔍 STYLEGAN2 SETUP VERIFICATION\n")
    
    results = {
        'Dependencies': test_dependencies(),
        'GPU': test_gpu(),
        'Config': test_config(),
        'Preprocessing': test_preprocessing(),
        'Sample Data': test_sample_image()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test:20s} {status}")
    
    critical = ['Dependencies', 'GPU']
    critical_pass = all(results[k] for k in critical)
    
    print("\n" + "="*60)
    if critical_pass:
        print("✅ READY TO PROCEED")
        print("="*60)
        print("\nNext steps:")
        print("1. Create directory structure")
        print("2. Download CelebA-HQ or FFHQ dataset")
        print("3. Run preprocessing")
        print("4. Start training")
    else:
        print("⚠️ SETUP INCOMPLETE")
        print("="*60)
        print("\nFix issues above before proceeding")
    
    return critical_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)