#!/usr/bin/env python3
"""
🦾 DEMO 3: Real-time Pose Estimation
======================================
Track 17 body keypoints in real-time.

What you'll learn:
- Human pose estimation
- Keypoint detection
- Skeleton visualization
- Movement tracking

Usage:
    python demo_03_pose_estimation.py

Controls:
    'q' - Quit
    'k' - Toggle keypoint numbers
    's' - Toggle skeleton
    'r' - Record pose sequence

Author: SAIR Community 🇸🇩
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import json

# COCO keypoint names
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

class PoseTracker:
    def __init__(self):
        self.show_keypoint_numbers = False
        self.show_skeleton = True
        self.recording = False
        self.pose_sequence = []
        
    def draw_pose(self, frame, keypoints, confidence_threshold=0.5):
        """Draw keypoints and skeleton on frame"""
        h, w = frame.shape[:2]
        
        if keypoints is None or len(keypoints) == 0:
            return frame
        
        result = frame.copy()
        
        # Get first person's keypoints
        kps = keypoints[0].cpu().numpy() if len(keypoints) > 0 else None
        
        if kps is None:
            return result
        
        # Draw skeleton
        if self.show_skeleton:
            for start_idx, end_idx in SKELETON:
                if (kps[start_idx][2] > confidence_threshold and 
                    kps[end_idx][2] > confidence_threshold):
                    start_point = (int(kps[start_idx][0]), int(kps[start_idx][1]))
                    end_point = (int(kps[end_idx][0]), int(kps[end_idx][1]))
                    cv2.line(result, start_point, end_point, (0, 255, 0), 2)
        
        # Draw keypoints
        for idx, (x, y, conf) in enumerate(kps):
            if conf > confidence_threshold:
                # Draw point
                cv2.circle(result, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.circle(result, (int(x), int(y)), 6, (255, 255, 255), 1)
                
                # Draw keypoint number
                if self.show_keypoint_numbers:
                    cv2.putText(result, str(idx), (int(x)+8, int(y)-8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return result
    
    def calculate_pose_stats(self, keypoints, confidence_threshold=0.5):
        """Calculate pose statistics"""
        if keypoints is None or len(keypoints) == 0:
            return {}
        
        kps = keypoints[0].cpu().numpy()
        
        visible_keypoints = sum(1 for kp in kps if kp[2] > confidence_threshold)
        
        # Calculate body metrics
        stats = {
            'visible_keypoints': visible_keypoints,
            'total_keypoints': len(kps),
            'visibility_percent': (visible_keypoints / len(kps)) * 100
        }
        
        # Calculate arm angles if visible
        left_shoulder = kps[5]
        left_elbow = kps[7]
        left_wrist = kps[9]
        
        if all(kp[2] > confidence_threshold for kp in [left_shoulder, left_elbow, left_wrist]):
            # Simple angle calculation
            v1 = left_shoulder[:2] - left_elbow[:2]
            v2 = left_wrist[:2] - left_elbow[:2]
            
            angle = np.degrees(np.arccos(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            ))
            stats['left_elbow_angle'] = angle
        
        return stats

def main():
    print("="*60)
    print("🦾 REAL-TIME POSE ESTIMATION")
    print("="*60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  'k' - Toggle keypoint numbers")
    print("  's' - Toggle skeleton")
    print("  'r' - Start/stop recording")
    print("  'i' - Show keypoint info")
    print("\nLoading model...\n")
    
    # Load pose model
    model = YOLO('yolov8n-pose.pt')
    print("✅ Pose model loaded!\n")
    
    # Initialize tracker
    tracker = PoseTracker()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Create large window
    window_name = 'Pose Estimation - SAIR 🇸🇩'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print("🚀 Pose tracking started!")
    print("Stand in front of camera and move around!\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run pose estimation
        results = model(frame, verbose=False)[0]
        
        # Draw pose
        result = tracker.draw_pose(frame, results.keypoints.data)
        
        # Calculate stats
        stats = tracker.calculate_pose_stats(results.keypoints.data)
        
        # Add info overlay
        y_offset = 30
        cv2.putText(result, f"People: {len(results.keypoints)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if stats:
            y_offset += 30
            cv2.putText(result, f"Keypoints: {stats['visible_keypoints']}/17", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if 'left_elbow_angle' in stats:
                y_offset += 30
                cv2.putText(result, f"L. Elbow: {stats['left_elbow_angle']:.0f}°", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Recording indicator
        if tracker.recording:
            cv2.circle(result, (result.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(result, "REC", (result.shape[1] - 70, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Record keypoints
            if len(results.keypoints) > 0:
                tracker.pose_sequence.append({
                    'frame': frame_count,
                    'keypoints': results.keypoints[0].cpu().numpy().tolist()
                })
        
        # Display
        cv2.imshow(window_name, result)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        elif key == ord('k'):
            tracker.show_keypoint_numbers = not tracker.show_keypoint_numbers
            print(f"Keypoint numbers: {'ON' if tracker.show_keypoint_numbers else 'OFF'}")
        elif key == ord('s'):
            tracker.show_skeleton = not tracker.show_skeleton
            print(f"Skeleton: {'ON' if tracker.show_skeleton else 'OFF'}")
        elif key == ord('r'):
            tracker.recording = not tracker.recording
            if tracker.recording:
                tracker.pose_sequence = []
                print("🔴 Recording started")
            else:
                # Save recording
                filename = f"pose_sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(tracker.pose_sequence, f)
                print(f"💾 Saved {len(tracker.pose_sequence)} frames to {filename}")
        elif key == ord('i'):
            print("\n" + "="*60)
            print("KEYPOINT INFORMATION")
            print("="*60)
            for idx, name in enumerate(KEYPOINT_NAMES):
                print(f"  {idx:2d}: {name}")
            print("="*60 + "\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
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