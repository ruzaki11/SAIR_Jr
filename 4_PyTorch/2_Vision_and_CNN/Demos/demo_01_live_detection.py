#!/usr/bin/env python3
"""
🎥 DEMO 1: Live YOLO Detection
===============================
Real-time object detection using your webcam.

What you'll learn:
- How to use YOLO for real-time detection
- Frame-by-frame processing
- FPS calculation
- Visual annotation

Usage:
    python demo_01_live_detection.py

Controls:
    'q' - Quit
    's' - Take screenshot
    'c' - Toggle confidence display

Author: SAIR Community 🇸🇩
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime

def main():
    print("="*60)
    print("🎥 LIVE YOLO DETECTION DEMO")
    print("="*60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Screenshot")
    print("  'c' - Toggle confidence")
    print("  'f' - Toggle fullscreen")
    print("  '+' - Increase confidence threshold")
    print("  '-' - Decrease confidence threshold")
    print("\nStarting webcam...\n")
    
    # Load YOLO model
    print("Loading YOLOv8n...")
    model = YOLO('yolov8n.pt')
    print("✅ Model loaded!\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Set resolution (higher for better visibility)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Configuration
    conf_threshold = 0.25
    show_confidence = True
    fps_values = []
    prev_time = time.time()
    screenshot_count = 0
    fullscreen = False
    
    print("🚀 Detection started! Wave at the camera!")
    print("="*60 + "\n")
    
    # Create named window with ability to resize
    window_name = 'YOLO Live Detection - SAIR 🇸🇩'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  # Start with large window
    
    print("💡 TIP: Press 'f' to toggle fullscreen")
    print("💡 TIP: Drag window edges to resize\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading frame")
            break
        
        # Run YOLO detection
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        fps_values.append(fps)
        if len(fps_values) > 30:
            fps_values.pop(0)
        avg_fps = np.mean(fps_values)
        prev_time = current_time
        
        # Annotate frame
        annotated = results.plot()
        
        # Add info overlay
        overlay_y = 30
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, overlay_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        overlay_y += 30
        cv2.putText(annotated, f"Objects: {len(results.boxes)}", (10, overlay_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        overlay_y += 30
        cv2.putText(annotated, f"Confidence: {conf_threshold:.2f}", (10, overlay_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Show detected classes
        if len(results.boxes) > 0 and show_confidence:
            overlay_y += 30
            classes = [results.names[int(c)] for c in results.boxes.cls]
            class_counts = {}
            for cls in classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            for cls, count in class_counts.items():
                cv2.putText(annotated, f"{cls}: {count}", (10, overlay_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                overlay_y += 25
        
        # Display
        cv2.imshow(window_name, annotated)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        elif key == ord('s'):
            # Save screenshot
            screenshot_count += 1
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"📸 Screenshot saved: {filename}")
        elif key == ord('c'):
            # Toggle confidence display
            show_confidence = not show_confidence
            print(f"Confidence display: {'ON' if show_confidence else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            # Increase confidence
            conf_threshold = min(0.9, conf_threshold + 0.05)
            print(f"Confidence threshold: {conf_threshold:.2f}")
        elif key == ord('-'):
            # Decrease confidence
            conf_threshold = max(0.1, conf_threshold - 0.05)
            print(f"Confidence threshold: {conf_threshold:.2f}")
        elif key == ord('f'):
            # Toggle fullscreen
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Fullscreen: ON")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Fullscreen: OFF")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Stats
    print("\n" + "="*60)
    print("📊 SESSION STATS")
    print("="*60)
    print(f"Average FPS: {np.mean(fps_values):.1f}")
    print(f"Screenshots taken: {screenshot_count}")
    print(f"Final confidence threshold: {conf_threshold:.2f}")
    print("\n✅ Demo complete! Thanks for using SAIR demos! 🇸🇩\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()