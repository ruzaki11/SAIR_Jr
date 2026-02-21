#!/usr/bin/env python3
"""
⚖️ DEMO 5: Model Comparison Tool
==================================
Compare different YOLO models side-by-side in real-time.

What you'll learn:
- Model performance differences
- Speed vs accuracy tradeoffs
- Resource usage
- Model selection

Models compared:
    - YOLOv8n (Nano) - Fastest
    - YOLOv8s (Small) - Balanced
    - YOLOv8m (Medium) - Accurate

Usage:
    python demo_05_model_comparison.py

Controls:
    'q' - Quit
    '1' - Show only Nano
    '2' - Show only Small  
    '3' - Show only Medium
    'a' - Show all models
    's' - Save comparison

Author: SAIR Community 🇸🇩
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

class ModelComparator:
    def __init__(self):
        self.models = {}
        self.stats = {}
        self.active_models = []
        
    def load_models(self, model_names):
        """Load multiple YOLO models"""
        print("Loading models...\n")
        for name in model_names:
            print(f"  Loading {name}...")
            try:
                self.models[name] = YOLO(f'{name}.pt')
                self.stats[name] = {
                    'inference_times': [],
                    'detections': [],
                    'fps': []
                }
                self.active_models.append(name)
                print(f"  ✅ {name} loaded")
            except Exception as e:
                print(f"  ❌ Failed to load {name}: {e}")
        
        print(f"\n✅ Loaded {len(self.models)} models\n")
    
    def run_inference(self, frame, model_name):
        """Run inference and collect stats"""
        start_time = time.time()
        
        results = self.models[model_name](frame, verbose=False)[0]
        
        inference_time = (time.time() - start_time) * 1000  # ms
        num_detections = len(results.boxes)
        fps = 1000 / inference_time if inference_time > 0 else 0
        
        # Update stats
        self.stats[model_name]['inference_times'].append(inference_time)
        self.stats[model_name]['detections'].append(num_detections)
        self.stats[model_name]['fps'].append(fps)
        
        # Keep last 30 frames
        for key in ['inference_times', 'detections', 'fps']:
            if len(self.stats[model_name][key]) > 30:
                self.stats[model_name][key].pop(0)
        
        return results, inference_time, fps
    
    def get_avg_stats(self, model_name):
        """Get average statistics"""
        stats = self.stats[model_name]
        if not stats['inference_times']:
            return None
        
        return {
            'avg_time': np.mean(stats['inference_times']),
            'avg_fps': np.mean(stats['fps']),
            'avg_detections': np.mean(stats['detections'])
        }

def create_comparison_view(frames, model_names, stats_list):
    """Create side-by-side comparison view"""
    n_models = len(frames)
    
    if n_models == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Resize frames
    h, w = 240, 320  # Half size for comparison
    resized_frames = [cv2.resize(f, (w, h)) for f in frames]
    
    # Create grid layout
    if n_models == 1:
        # Single model - full size
        return cv2.resize(frames[0], (640, 480))
    elif n_models == 2:
        # Side by side
        return np.hstack(resized_frames)
    else:
        # Grid layout
        row1 = np.hstack(resized_frames[:2])
        row2 = np.hstack([resized_frames[2], np.zeros((h, w, 3), dtype=np.uint8)])
        return np.vstack([row1, row2])

def draw_stats_overlay(frame, model_name, stats, y_offset=0):
    """Draw statistics overlay on frame"""
    if stats is None:
        return frame
    
    # Model name
    cv2.putText(frame, model_name.upper(), (10, y_offset + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Stats
    cv2.putText(frame, f"FPS: {stats['avg_fps']:.1f}", (10, y_offset + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Time: {stats['avg_time']:.1f}ms", (10, y_offset + 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.putText(frame, f"Objects: {stats['avg_detections']:.1f}", (10, y_offset + 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    return frame

def main():
    print("="*60)
    print("⚖️ MODEL COMPARISON TOOL")
    print("="*60)
    print("\nThis tool compares different YOLO models side-by-side")
    print("\nControls:")
    print("  'q' - Quit")
    print("  '1' - Show only Nano")
    print("  '2' - Show only Small")
    print("  '3' - Show only Medium")
    print("  'a' - Show all models")
    print("  's' - Save comparison screenshot")
    print("  'r' - Show performance report")
    print()
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load models
    model_names = ['yolov8n', 'yolov8s', 'yolov8m']
    comparator.load_models(model_names)
    
    if not comparator.models:
        print("❌ No models loaded. Exiting.")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Create large window
    window_name = 'Model Comparison - SAIR 🇸🇩'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print("🚀 Comparison started!\n")
    
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference on all active models
        annotated_frames = []
        
        for model_name in comparator.active_models:
            results, inf_time, fps = comparator.run_inference(frame, model_name)
            
            # Annotate
            annotated = results.plot()
            
            # Add stats overlay
            stats = comparator.get_avg_stats(model_name)
            annotated = draw_stats_overlay(annotated, model_name, stats)
            
            annotated_frames.append(annotated)
        
        # Create comparison view
        comparison = create_comparison_view(
            annotated_frames, 
            comparator.active_models,
            [comparator.get_avg_stats(m) for m in comparator.active_models]
        )
        
        # Display
        cv2.imshow(window_name, comparison)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        elif key == ord('1'):
            comparator.active_models = ['yolov8n']
            print("Showing: Nano only")
        elif key == ord('2'):
            comparator.active_models = ['yolov8s']
            print("Showing: Small only")
        elif key == ord('3'):
            comparator.active_models = ['yolov8m']
            print("Showing: Medium only")
        elif key == ord('a'):
            comparator.active_models = list(comparator.models.keys())
            print("Showing: All models")
        elif key == ord('s'):
            screenshot_count += 1
            filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, comparison)
            print(f"💾 Saved: {filename}")
        elif key == ord('r'):
            # Print performance report
            print("\n" + "="*60)
            print("📊 PERFORMANCE REPORT")
            print("="*60)
            for model_name in comparator.models.keys():
                stats = comparator.get_avg_stats(model_name)
                if stats:
                    print(f"\n{model_name.upper()}:")
                    print(f"  Average FPS: {stats['avg_fps']:.1f}")
                    print(f"  Average Inference Time: {stats['avg_time']:.1f}ms")
                    print(f"  Average Detections: {stats['avg_detections']:.1f}")
            print("="*60 + "\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final report
    print("\n" + "="*60)
    print("📊 FINAL COMPARISON REPORT")
    print("="*60)
    
    for model_name in comparator.models.keys():
        stats = comparator.get_avg_stats(model_name)
        if stats:
            print(f"\n{model_name.upper()}:")
            print(f"  Average FPS: {stats['avg_fps']:.1f}")
            print(f"  Average Time: {stats['avg_time']:.1f}ms")
            print(f"  Detections: {stats['avg_detections']:.1f}")
    
    # Recommendations
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    # Find fastest model
    fastest = max(comparator.models.keys(), 
                 key=lambda m: comparator.get_avg_stats(m)['avg_fps'] 
                 if comparator.get_avg_stats(m) else 0)
    print(f"⚡ Fastest: {fastest.upper()}")
    
    # Find most accurate (most detections on average)
    most_accurate = max(comparator.models.keys(),
                       key=lambda m: comparator.get_avg_stats(m)['avg_detections']
                       if comparator.get_avg_stats(m) else 0)
    print(f"🎯 Most Detections: {most_accurate.upper()}")
    
    print("\n✅ Demo complete! 🇸🇩\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()