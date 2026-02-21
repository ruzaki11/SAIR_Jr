#!/usr/bin/env python3
"""
🎬 DEMO 7: Video Processing
============================
Process videos with object detection and tracking.

What you'll learn:
- Video I/O with OpenCV
- Frame-by-frame processing
- Video encoding
- Progress tracking

Usage:
    python demo_07_video_processing.py input_video.mp4

Features:
- Process any video file
- Add object detection
- Export processed video
- Show processing progress
- Configurable output quality

Author: SAIR Community 🇸🇩
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, model_name='yolov8n', confidence=0.25):
        self.model = YOLO(f'{model_name}.pt')
        self.confidence = confidence
        
    def process_video(self, input_path, output_path=None, 
                     show_preview=False, skip_frames=1):
        """Process video file with object detection"""
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {input_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f}s")
        
        # Create output path
        if output_path is None:
            input_file = Path(input_path)
            output_path = f"{input_file.stem}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ Error: Could not create output video")
            cap.release()
            return
        
        print(f"💾 Output: {output_path}\n")
        
        # Processing stats
        detection_counts = []
        processing_times = []
        
        frame_count = 0
        processed_count = 0
        
        # Process frames
        print("🚀 Processing video...")
        
        pbar = tqdm(total=total_frames, desc="Processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if specified
            if frame_count % skip_frames != 0:
                out.write(frame)
                pbar.update(1)
                continue
            
            # Run detection
            import time
            start_time = time.time()
            
            results = self.model(frame, conf=self.confidence, verbose=False)[0]
            
            proc_time = (time.time() - start_time) * 1000
            processing_times.append(proc_time)
            
            # Annotate frame
            annotated = results.plot()
            
            # Add frame info
            cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Objects: {len(results.boxes)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            detection_counts.append(len(results.boxes))
            
            # Write frame
            out.write(annotated)
            processed_count += 1
            
            # Show preview
            if show_preview:
                cv2.imshow('Processing Preview', cv2.resize(annotated, (640, 480)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️ Processing interrupted by user")
                    break
            
            pbar.update(1)
        
        pbar.close()
        
        # Cleanup
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Print stats
        print("\n" + "="*60)
        print("📊 PROCESSING STATS")
        print("="*60)
        print(f"Frames processed: {processed_count}/{total_frames}")
        print(f"Average processing time: {np.mean(processing_times):.1f}ms/frame")
        print(f"Average FPS: {1000/np.mean(processing_times):.1f}")
        print(f"Average detections: {np.mean(detection_counts):.1f}")
        print(f"Total detections: {sum(detection_counts)}")
        print(f"\n✅ Output saved: {output_path}")
        print("="*60)
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Process video with YOLO detection')
    parser.add_argument('input_video', help='Input video file')
    parser.add_argument('-o', '--output', default=None,
                       help='Output video file')
    parser.add_argument('-m', '--model', default='yolov8n',
                       help='Model to use (yolov8n, yolov8s, yolov8m)')
    parser.add_argument('-c', '--confidence', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--preview', action='store_true',
                       help='Show processing preview')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth frame (1=all frames)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎬 VIDEO PROCESSING")
    print("="*60)
    print(f"\nInput: {args.input_video}")
    print(f"Model: {args.model}")
    print(f"Confidence: {args.confidence}")
    print(f"Skip frames: {args.skip}")
    
    # Check if input exists
    if not Path(args.input_video).exists():
        print(f"\n❌ Error: {args.input_video} not found")
        return
    
    # Initialize processor
    processor = VideoProcessor(
        model_name=args.model,
        confidence=args.confidence
    )
    
    # Process video
    output_path = processor.process_video(
        input_path=args.input_video,
        output_path=args.output,
        show_preview=args.preview,
        skip_frames=args.skip
    )
    
    print("\n✅ Video processing complete! 🇸🇩\n")
    
    if output_path:
        print(f"📹 Play with: vlc {output_path}")
        print(f"or: ffplay {output_path}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
