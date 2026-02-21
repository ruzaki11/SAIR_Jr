#!/usr/bin/env python3
"""
📁 DEMO 6: Batch Image Processing
===================================
Process entire folders of images with YOLO detection.

What you'll learn:
- Batch processing
- File handling
- Progress tracking
- Result export

Usage:
    python demo_06_batch_processing.py /path/to/images

Features:
- Process folders of images
- Export annotated images
- Generate CSV report
- Filter by confidence
- Class-specific detection

Author: SAIR Community 🇸🇩
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
import csv
from datetime import datetime
from tqdm import tqdm
import json

class BatchProcessor:
    def __init__(self, model_name='yolov8n', confidence=0.25):
        self.model = YOLO(f'{model_name}.pt')
        self.confidence = confidence
        self.results_data = []
        
    def process_image(self, image_path):
        """Process single image and return results"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Run detection
        results = self.model(img, conf=self.confidence, verbose=False)[0]
        
        # Extract data
        detections = []
        for box in results.boxes:
            detection = {
                'class': results.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist()
            }
            detections.append(detection)
        
        result_data = {
            'image': str(image_path),
            'num_detections': len(detections),
            'detections': detections
        }
        
        return results, result_data
    
    def process_folder(self, input_folder, output_folder=None, 
                      save_images=True, save_csv=True, save_json=True):
        """Process entire folder of images"""
        input_path = Path(input_folder)
        
        if not input_path.exists():
            print(f"❌ Error: {input_folder} does not exist")
            return
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f'*{ext}')))
            image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
        
        if not image_files:
            print(f"❌ No images found in {input_folder}")
            return
        
        print(f"📁 Found {len(image_files)} images")
        
        # Create output folder
        if output_folder is None:
            output_folder = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"📂 Output folder: {output_folder}\n")
        
        # Process images
        self.results_data = []
        
        print("🚀 Processing images...")
        for img_path in tqdm(image_files, desc="Processing"):
            results, result_data = self.process_image(img_path)
            
            if results is None:
                continue
            
            self.results_data.append(result_data)
            
            # Save annotated image
            if save_images:
                annotated = results.plot()
                output_img_path = output_path / img_path.name
                cv2.imwrite(str(output_img_path), annotated)
        
        print(f"\n✅ Processed {len(self.results_data)} images")
        
        # Export results
        if save_csv:
            self.export_csv(output_path / 'results.csv')
        
        if save_json:
            self.export_json(output_path / 'results.json')
        
        # Print summary
        self.print_summary()
        
        return self.results_data
    
    def export_csv(self, output_path):
        """Export results to CSV"""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Num Detections', 'Classes', 'Avg Confidence'])
            
            for result in self.results_data:
                classes = [d['class'] for d in result['detections']]
                avg_conf = np.mean([d['confidence'] for d in result['detections']]) if result['detections'] else 0
                
                writer.writerow([
                    Path(result['image']).name,
                    result['num_detections'],
                    ', '.join(set(classes)),
                    f"{avg_conf:.3f}"
                ])
        
        print(f"💾 CSV saved: {output_path}")
    
    def export_json(self, output_path):
        """Export results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.results_data, f, indent=2)
        
        print(f"💾 JSON saved: {output_path}")
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*60)
        print("📊 PROCESSING SUMMARY")
        print("="*60)
        
        total_detections = sum(r['num_detections'] for r in self.results_data)
        avg_detections = total_detections / len(self.results_data) if self.results_data else 0
        
        print(f"Total images: {len(self.results_data)}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {avg_detections:.1f}")
        
        # Class distribution
        all_classes = []
        for result in self.results_data:
            all_classes.extend([d['class'] for d in result['detections']])
        
        if all_classes:
            from collections import Counter
            class_counts = Counter(all_classes)
            
            print("\nClass Distribution:")
            for cls, count in class_counts.most_common(10):
                print(f"  {cls}: {count}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Batch process images with YOLO')
    parser.add_argument('input_folder', nargs='?', default='.',
                       help='Input folder containing images')
    parser.add_argument('-o', '--output', default=None,
                       help='Output folder for results')
    parser.add_argument('-m', '--model', default='yolov8n',
                       help='Model to use (yolov8n, yolov8s, yolov8m)')
    parser.add_argument('-c', '--confidence', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--no-images', action='store_true',
                       help='Do not save annotated images')
    parser.add_argument('--no-csv', action='store_true',
                       help='Do not save CSV report')
    parser.add_argument('--no-json', action='store_true',
                       help='Do not save JSON report')
    
    args = parser.parse_args()
    
    print("="*60)
    print("📁 BATCH IMAGE PROCESSING")
    print("="*60)
    print(f"\nInput folder: {args.input_folder}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    print()
    
    # Initialize processor
    processor = BatchProcessor(
        model_name=args.model,
        confidence=args.confidence
    )
    
    # Process folder
    processor.process_folder(
        input_folder=args.input_folder,
        output_folder=args.output,
        save_images=not args.no_images,
        save_csv=not args.no_csv,
        save_json=not args.no_json
    )
    
    print("\n✅ Batch processing complete! 🇸🇩\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
