#!/usr/bin/env python3
"""
🎨 DEMO 2: Real-time Background Removal
=========================================
Remove backgrounds in real-time using YOLO segmentation.
Like Zoom/Teams virtual backgrounds!

What you'll learn:
- Instance segmentation
- Mask processing
- Background replacement
- Real-time image compositing

Usage:
    python demo_02_background_removal.py

Controls:
    'q' - Quit
    'b' - Cycle background colors
    's' - Save result
    'f' - Toggle blur effect

Author: SAIR Community 🇸🇩
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

class BackgroundRemover:
    def __init__(self):
        self.backgrounds = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 100, 255),
            'purple': (128, 0, 128),
            'blur': None  # Special case
        }
        self.bg_names = list(self.backgrounds.keys())
        self.current_bg_idx = 0
        self.use_blur = False
        
    def get_current_background(self):
        return self.bg_names[self.current_bg_idx]
    
    def next_background(self):
        self.current_bg_idx = (self.current_bg_idx + 1) % len(self.bg_names)
        bg_name = self.get_current_background()
        print(f"🎨 Background: {bg_name.upper()}")
        return bg_name
    
    def remove_background(self, frame, masks, use_blur=False):
        """Remove background and replace with solid color or blur"""
        h, w = frame.shape[:2]
        
        # Create combined mask
        combined_mask = np.zeros((h, w), dtype=bool)
        
        if masks is not None and len(masks) > 0:
            for mask in masks.data:
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (w, h))
                combined_mask = np.logical_or(combined_mask, mask_resized > 0.5)
        
        # Create result
        result = frame.copy()
        
        bg_name = self.get_current_background()
        
        if bg_name == 'blur' or use_blur:
            # Blur background
            blurred = cv2.GaussianBlur(frame, (51, 51), 0)
            result[~combined_mask] = blurred[~combined_mask]
        else:
            # Solid color background
            bg_color = self.backgrounds[bg_name]
            result[~combined_mask] = bg_color
        
        return result, combined_mask

def main():
    print("="*60)
    print("🎨 REAL-TIME BACKGROUND REMOVAL")
    print("="*60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  'b' - Change background color")
    print("  's' - Save result")
    print("  'f' - Toggle blur background")
    print("\nLoading model...\n")
    
    # Load segmentation model
    model = YOLO('yolov8n-seg.pt')
    print("✅ Segmentation model loaded!\n")
    
    # Initialize background remover
    bg_remover = BackgroundRemover()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Create large window
    window_name = 'Background Removal - SAIR 🇸🇩'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    save_count = 0
    use_blur = False
    
    print("🚀 Background removal started!")
    print(f"🎨 Current background: {bg_remover.get_current_background().upper()}\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run segmentation
        results = model(frame, classes=[0], verbose=False)[0]  # class 0 = person
        
        # Remove background
        result, mask = bg_remover.remove_background(frame, results.masks, use_blur)
        
        # Add info
        bg_name = bg_remover.get_current_background()
        status_text = f"Background: {bg_name.upper()}"
        if use_blur:
            status_text += " + BLUR"
        
        cv2.putText(result, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show mask coverage
        mask_percent = (mask.sum() / mask.size) * 100
        cv2.putText(result, f"Person: {mask_percent:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display
        cv2.imshow(window_name, result)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        elif key == ord('b'):
            bg_remover.next_background()
        elif key == ord('s'):
            save_count += 1
            filename = f"bg_removed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, result)
            print(f"💾 Saved: {filename}")
        elif key == ord('f'):
            use_blur = not use_blur
            print(f"Blur effect: {'ON' if use_blur else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("📊 SESSION STATS")
    print("="*60)
    print(f"Images saved: {save_count}")
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