#!/usr/bin/env python3
"""
✨ DEMO 4: Gesture Control System
==================================
Control your computer with hand gestures!

What you'll learn:
- Gesture recognition from pose
- State machines
- Action triggering
- Real-time interaction

Supported Gestures:
    ✋ OPEN HAND - Stop/Pause
    👆 INDEX UP - Volume Up
    ✌️ PEACE SIGN - Volume Down
    👊 FIST - Select/Click
    👍 THUMBS UP - Like/Approve
    🙏 PRAYER - Center/Reset

Usage:
    python demo_04_gesture_control.py

Controls:
    'q' - Quit
    'd' - Toggle debug mode
    'h' - Show help

Author: SAIR Community 🇸🇩
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

class GestureRecognizer:
    def __init__(self):
        self.gesture_history = deque(maxlen=5)
        self.last_action_time = 0
        self.action_cooldown = 1.0  # seconds
        self.debug_mode = False
        
        # Gesture action counters
        self.gesture_counts = {
            'fist': 0,
            'open_hand': 0,
            'index_up': 0,
            'peace': 0,
            'thumbs_up': 0,
            'prayer': 0
        }
        
    def detect_gesture(self, keypoints, confidence_threshold=0.5):
        """Detect gesture from pose keypoints"""
        if keypoints is None or len(keypoints) == 0:
            return "no_person", None
        
        kps = keypoints[0].cpu().numpy()
        
        # Extract relevant keypoints
        left_wrist = kps[9]
        right_wrist = kps[10]
        left_elbow = kps[7]
        right_elbow = kps[8]
        left_shoulder = kps[5]
        right_shoulder = kps[6]
        nose = kps[0]
        
        # Check if key points are visible
        required_points = [left_wrist, right_wrist, left_shoulder, right_shoulder]
        if not all(pt[2] > confidence_threshold for pt in required_points):
            return "low_confidence", None
        
        # Calculate hand positions relative to body
        body_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        hands_distance = np.linalg.norm(left_wrist[:2] - right_wrist[:2])
        
        # Calculate hand heights
        left_hand_height = left_shoulder[1] - left_wrist[1]
        right_hand_height = right_shoulder[1] - right_wrist[1]
        
        # Gesture detection logic
        gesture_data = {
            'hands_distance': hands_distance,
            'body_width': body_width,
            'left_height': left_hand_height,
            'right_height': right_hand_height
        }
        
        # FIST - Hands close together
        if hands_distance < body_width * 0.3:
            return "fist", gesture_data
        
        # OPEN HAND - Hands far apart
        elif hands_distance > body_width * 1.5:
            return "open_hand", gesture_data
        
        # PRAYER - Hands together at center, high up
        elif (hands_distance < body_width * 0.5 and 
              left_hand_height > 50 and right_hand_height > 50):
            # Check if hands are centered
            center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            hands_center_x = (left_wrist[0] + right_wrist[0]) / 2
            if abs(center_x - hands_center_x) < body_width * 0.3:
                return "prayer", gesture_data
        
        # INDEX UP - One hand significantly higher
        elif left_hand_height > right_hand_height + 80:
            return "index_up", gesture_data
        
        # PEACE SIGN - Right hand higher (simplified)
        elif right_hand_height > left_hand_height + 80:
            return "peace", gesture_data
        
        # THUMBS UP - One hand at shoulder level
        elif (abs(left_hand_height) < 30 or abs(right_hand_height) < 30):
            return "thumbs_up", gesture_data
        
        return "neutral", gesture_data
    
    def smooth_gesture(self, gesture):
        """Smooth gesture detection using history"""
        self.gesture_history.append(gesture)
        
        # Return most common gesture in history
        if len(self.gesture_history) >= 3:
            from collections import Counter
            counts = Counter(self.gesture_history)
            most_common = counts.most_common(1)[0][0]
            return most_common
        
        return gesture
    
    def trigger_action(self, gesture):
        """Trigger action based on gesture"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        
        # Perform action
        action = None
        
        if gesture == "fist":
            action = "🖱️ CLICK/SELECT"
            self.gesture_counts['fist'] += 1
            
        elif gesture == "open_hand":
            action = "⏸️ STOP/PAUSE"
            self.gesture_counts['open_hand'] += 1
            
        elif gesture == "index_up":
            action = "🔊 VOLUME UP"
            self.gesture_counts['index_up'] += 1
            
        elif gesture == "peace":
            action = "🔉 VOLUME DOWN"
            self.gesture_counts['peace'] += 1
            
        elif gesture == "thumbs_up":
            action = "👍 LIKE/APPROVE"
            self.gesture_counts['thumbs_up'] += 1
            
        elif gesture == "prayer":
            action = "🎯 CENTER/RESET"
            self.gesture_counts['prayer'] += 1
        
        if action:
            self.last_action_time = current_time
            print(f"\n⚡ ACTION: {action}")
        
        return action

def draw_gesture_ui(frame, gesture, action, recognizer):
    """Draw gesture UI overlay"""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay for gesture display
    overlay = frame.copy()
    cv2.rectangle(overlay, (w-300, 0), (w, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Gesture mapping
    gesture_display = {
        'fist': '👊 FIST',
        'open_hand': '✋ OPEN HAND',
        'index_up': '👆 INDEX UP',
        'peace': '✌️ PEACE',
        'thumbs_up': '👍 THUMBS UP',
        'prayer': '🙏 PRAYER',
        'neutral': '👋 NEUTRAL',
        'no_person': '❌ NO PERSON',
        'low_confidence': '⚠️ LOW CONF'
    }
    
    # Current gesture
    gesture_text = gesture_display.get(gesture, gesture)
    cv2.putText(frame, "GESTURE:", (w-290, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, gesture_text, (w-290, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Last action
    if action:
        cv2.putText(frame, "ACTION:", (w-290, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, action[:15], (w-290, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Gesture counts (small)
    y = h - 150
    cv2.putText(frame, "STATS:", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for gesture_name, count in recognizer.gesture_counts.items():
        if count > 0:
            y += 20
            cv2.putText(frame, f"{gesture_name}: {count}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame

def main():
    print("="*60)
    print("✨ GESTURE CONTROL SYSTEM")
    print("="*60)
    print("\n🎮 Supported Gestures:")
    print("  ✋ OPEN HAND    → Stop/Pause")
    print("  👆 INDEX UP    → Volume Up")
    print("  ✌️ PEACE       → Volume Down")
    print("  👊 FIST        → Select/Click")
    print("  👍 THUMBS UP   → Like/Approve")
    print("  🙏 PRAYER      → Center/Reset")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'd' - Toggle debug mode")
    print("  'h' - Show help")
    print("\nLoading model...\n")
    
    # Load pose model
    model = YOLO('yolov8n-pose.pt')
    print("✅ Pose model loaded!\n")
    
    # Initialize recognizer
    recognizer = GestureRecognizer()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Create large window
    window_name = 'Gesture Control - SAIR 🇸🇩'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print("🚀 Gesture control started!")
    print("Make gestures in front of the camera!\n")
    
    last_action = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run pose estimation
        results = model(frame, verbose=False)[0]
        
        # Detect gesture
        gesture, gesture_data = recognizer.detect_gesture(results.keypoints.data)
        
        # Smooth gesture
        smooth_gesture = recognizer.smooth_gesture(gesture)
        
        # Trigger action
        action = recognizer.trigger_action(smooth_gesture)
        if action:
            last_action = action
        
        # Draw pose
        annotated = results.plot()
        
        # Draw gesture UI
        annotated = draw_gesture_ui(annotated, smooth_gesture, last_action, recognizer)
        
        # Debug info
        if recognizer.debug_mode and gesture_data:
            y = 30
            for key, value in gesture_data.items():
                cv2.putText(annotated, f"{key}: {value:.1f}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                y += 20
        
        # Display
        cv2.imshow(window_name, annotated)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        elif key == ord('d'):
            recognizer.debug_mode = not recognizer.debug_mode
            print(f"Debug mode: {'ON' if recognizer.debug_mode else 'OFF'}")
        elif key == ord('h'):
            print("\n" + "="*60)
            print("GESTURE HELP")
            print("="*60)
            print("✋ OPEN HAND    - Spread arms wide")
            print("👆 INDEX UP    - Raise left hand high")
            print("✌️ PEACE       - Raise right hand high")
            print("👊 FIST        - Bring hands together")
            print("👍 THUMBS UP   - Hand at shoulder level")
            print("🙏 PRAYER      - Hands together, centered, raised")
            print("="*60 + "\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final stats
    print("\n" + "="*60)
    print("📊 SESSION STATS")
    print("="*60)
    total = sum(recognizer.gesture_counts.values())
    print(f"Total gestures recognized: {total}")
    for gesture, count in recognizer.gesture_counts.items():
        if count > 0:
            print(f"  {gesture}: {count}")
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