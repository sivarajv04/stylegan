"""
Dataset quality analysis and reporting
"""
import os
import cv2
import numpy as np
import imagehash
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Optional, List
import json

class DatasetQualityAnalyzer:
    """Comprehensive dataset quality analysis"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.report = {
            'total_images': 0,
            'duplicates': [],
            'blur_distribution': [],
            'brightness_distribution': [],
            'resolution_distribution': {},
            'aspect_ratios': [],
            'file_sizes': []
        }
    
    def analyze(self, save_path: Optional[str] = None):
        """Run complete quality analysis"""
        print("Starting dataset quality analysis...")
        
        # Get all images
        image_files = list(self.dataset_path.rglob('*.jpg')) + \
                     list(self.dataset_path.rglob('*.png'))
        
        self.report['total_images'] = len(image_files)
        print(f"Found {len(image_files)} images")
        
        # 1. Detect duplicates
        print("\n1. Detecting duplicates...")
        self._detect_duplicates(image_files)
        
        # 2. Analyze image quality
        print("\n2. Analyzing image quality...")
        self._analyze_quality(image_files)
        
        # 3. Check resolutions
        print("\n3. Checking resolutions...")
        self._analyze_resolutions(image_files)
        
        # 4. Generate visualizations
        print("\n4. Generating visualizations...")
        self._generate_visualizations(save_path)
        
        # 5. Save report
        if save_path:
            report_path = Path(save_path) / 'quality_report.json'
            with open(report_path, 'w') as f:
                json.dump(self.report, f, indent=2)
            print(f"\nReport saved to {report_path}")
        
        return self.report
    
    def _detect_duplicates(self, image_files: List[Path]):
        """Detect duplicate images using perceptual hashing"""
        hashes = {}
        duplicates = []
        
        for img_path in tqdm(image_files, desc="Hashing images"):
            try:
                img = Image.open(img_path)
                img_hash = str(imagehash.phash(img))
                
                if img_hash in hashes:
                    duplicates.append({
                        'original': str(hashes[img_hash]),
                        'duplicate': str(img_path)
                    })
                else:
                    hashes[img_hash] = img_path
                    
            except Exception:
                continue
        
        self.report['duplicates'] = duplicates
        self.report['num_duplicates'] = len(duplicates)
        print(f"Found {len(duplicates)} duplicate images")
    
    def _analyze_quality(self, image_files: List[Path]):
        """Analyze blur and brightness"""
        blur_scores = []
        brightness_values = []
        
        for img_path in tqdm(image_files[:1000], desc="Quality check"):  # Sample
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Blur detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_scores.append(blur)
                
                # Brightness
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
            except Exception:
                continue
        
        self.report['blur_distribution'] = {
            'mean': float(np.mean(blur_scores)),
            'std': float(np.std(blur_scores)),
            'min': float(np.min(blur_scores)),
            'max': float(np.max(blur_scores)),
            'low_quality_count': int(np.sum(np.array(blur_scores) < 100))
        }
        
        self.report['brightness_distribution'] = {
            'mean': float(np.mean(brightness_values)),
            'std': float(np.std(brightness_values)),
            'too_dark': int(np.sum(np.array(brightness_values) < 50)),
            'too_bright': int(np.sum(np.array(brightness_values) > 200))
        }
    
    def _analyze_resolutions(self, image_files: List[Path]):
        """Analyze image resolutions and aspect ratios"""
        resolutions = defaultdict(int)
        aspect_ratios = []
        file_sizes = []
        
        for img_path in tqdm(image_files, desc="Resolution check"):
            try:
                img = Image.open(img_path)
                w, h = img.size
                
                # Resolution
                res = f"{w}x{h}"
                resolutions[res] += 1
                
                # Aspect ratio
                aspect_ratios.append(w / h)
                
                # File size (MB)
                file_sizes.append(img_path.stat().st_size / (1024 * 1024))
                
            except Exception:
                continue
        
        self.report['resolution_distribution'] = dict(resolutions)
        self.report['aspect_ratios'] = {
            'mean': float(np.mean(aspect_ratios)),
            'std': float(np.std(aspect_ratios))
        }
        self.report['file_sizes'] = {
            'mean_mb': float(np.mean(file_sizes)),
            'total_gb': float(np.sum(file_sizes) / 1024)
        }
    
    def _generate_visualizations(self, save_path: Optional[str]):
        """Generate quality visualization plots"""
        if not save_path:
            return
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Resolution distribution
        if self.report['resolution_distribution']:
            _, ax = plt.subplots(figsize=(12, 6))
            resolutions = list(self.report['resolution_distribution'].keys())
            counts = list(self.report['resolution_distribution'].values())
            
            ax.bar(range(len(resolutions)), counts)
            ax.set_xticks(range(len(resolutions)))
            ax.set_xticklabels(resolutions, rotation=45, ha='right')
            ax.set_xlabel('Resolution')
            ax.set_ylabel('Count')
            ax.set_title('Image Resolution Distribution')
            plt.tight_layout()
            plt.savefig(save_path / 'resolution_distribution.png', dpi=150)
            plt.close()
        
        # 2. Quality summary
        _, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Blur quality
        blur_stats = self.report['blur_distribution']
        axes[0].bar(['Low Quality', 'Good Quality'], 
                   [blur_stats['low_quality_count'], 
                    self.report['total_images'] - blur_stats['low_quality_count']])
        axes[0].set_title('Blur Quality Distribution')
        axes[0].set_ylabel('Number of Images')
        
        # Brightness issues
        bright_stats = self.report['brightness_distribution']
        axes[1].bar(['Too Dark', 'Good', 'Too Bright'],
                   [bright_stats['too_dark'],
                    self.report['total_images'] - bright_stats['too_dark'] - bright_stats['too_bright'],
                    bright_stats['too_bright']])
        axes[1].set_title('Brightness Distribution')
        axes[1].set_ylabel('Number of Images')
        
        plt.tight_layout()
        plt.savefig(save_path / 'quality_summary.png', dpi=150)
        plt.close()
        
        print(f"Visualizations saved to {save_path}")

def generate_quality_report(dataset_path: str, output_path: str):
    """Generate comprehensive quality report"""
    analyzer = DatasetQualityAnalyzer(dataset_path)
    report = analyzer.analyze(save_path=output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET QUALITY REPORT SUMMARY")
    print("="*60)
    print("\nTotal Images: {report['total_images']}")
    print("Duplicates Found: {report['num_duplicates']}")
    print("\nBlur Statistics:")
    print("  Mean: {report['blur_distribution']['mean']:.2f}")
    print("  Low Quality: {report['blur_distribution']['low_quality_count']}")
    print("\nBrightness Issues:")
    print("  Too Dark: {report['brightness_distribution']['too_dark']}")
    print("  Too Bright: {report['brightness_distribution']['too_bright']}")
    print("\nStorage:")
    print("  Total Size: {report['file_sizes']['total_gb']:.2f} GB")
    print("="*60)

if __name__ == "__main__":
    generate_quality_report(
        dataset_path="./data/processed/celeba_256",
        output_path="./reports/data_quality"
    )