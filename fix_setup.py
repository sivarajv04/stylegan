"""Quick setup fix script"""
import os
from pathlib import Path
import shutil

def fix_structure():
    """Create proper directory structure"""
    
    dirs = [
        'configs',
        'data/raw',
        'data/processed', 
        'data/samples',
        'models',
        'training',
        'evaluation',
        'checkpoints',
        'outputs',
        'reports'
    ]
    
    print("Creating directory structure...")
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"✓ {d}")
    
    # Move config if in wrong place
    if Path('training_config.yaml').exists():
        shutil.move('training_config.yaml', 'configs/training_config.yaml')
        print("\n✓ Moved config to configs/")
    
    print("\n✅ Structure fixed!")
    print("\nRun: python test_current_setup.py")

if __name__ == "__main__":
    fix_structure()