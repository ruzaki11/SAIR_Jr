#!/usr/bin/env python3
"""
🎓 SAIR Computer Vision Demos - Master Launcher
================================================
Interactive launcher for all CV demos.

Usage:
    python run_demos.py

Author: SAIR Community 🇸🇩
"""

import subprocess
import sys
from pathlib import Path

DEMOS = {
    '1': {
        'name': 'Live Detection',
        'file': 'demo_01_live_detection.py',
        'description': '🎥 Real-time object detection with webcam',
        'requirements': 'Webcam'
    },
    '2': {
        'name': 'Background Removal',
        'file': 'demo_02_background_removal.py',
        'description': '🎨 Remove backgrounds like Zoom/Teams',
        'requirements': 'Webcam'
    },
    '3': {
        'name': 'Pose Estimation',
        'file': 'demo_03_pose_estimation.py',
        'description': '🦾 Track 17 body keypoints',
        'requirements': 'Webcam'
    },
    '4': {
        'name': 'Gesture Control',
        'file': 'demo_04_gesture_control.py',
        'description': '✨ Control with hand gestures',
        'requirements': 'Webcam'
    },
    '5': {
        'name': 'Model Comparison',
        'file': 'demo_05_model_comparison.py',
        'description': '⚖️ Compare YOLO models side-by-side',
        'requirements': 'Webcam'
    },
    '6': {
        'name': 'Batch Processing',
        'file': 'demo_06_batch_processing.py',
        'description': '📁 Process folders of images',
        'requirements': 'Image folder'
    },
    '7': {
        'name': 'Video Processing',
        'file': 'demo_07_video_processing.py',
        'description': '🎬 Process videos with detection',
        'requirements': 'Video file'
    }
}

def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🎓 SAIR COMPUTER VISION DEMOS                              ║
║   Standalone Python Demos for Hands-On Learning 🇸🇩          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print demo selection menu"""
    print("\n📚 Available Demos:")
    print("─" * 60)
    
    for key, demo in DEMOS.items():
        print(f"  [{key}] {demo['name']}")
        print(f"      {demo['description']}")
        print(f"      Requirements: {demo['requirements']}")
        print()
    
    print("  [i] Install dependencies")
    print("  [h] Help & Documentation")
    print("  [q] Quit")
    print("─" * 60)

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import cv2
        import numpy
        from ultralytics import YOLO
        return True
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("\n💡 Install with: pip install -r requirements.txt")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✅ Dependencies installed successfully!")
        print("You can now run the demos.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        print("Please install manually: pip install -r requirements.txt\n")

def show_help():
    """Show help and documentation"""
    help_text = """
╔══════════════════════════════════════════════════════════════╗
║                         HELP & INFO                          ║
╚══════════════════════════════════════════════════════════════╝

🎯 LEARNING PATH:

Beginner (Start here!):
  1. Demo 1 - Live Detection
  2. Demo 6 - Batch Processing
  3. Demo 7 - Video Processing

Intermediate:
  4. Demo 2 - Background Removal
  5. Demo 5 - Model Comparison

Advanced:
  6. Demo 3 - Pose Estimation
  7. Demo 4 - Gesture Control

🛠️ REQUIREMENTS:

- Python 3.8+
- Webcam (for demos 1-5)
- Images/Videos (for demos 6-7)

📦 INSTALLATION:

From this launcher:
  Select [i] to install dependencies

Manual:
  pip install -r requirements.txt

🚀 RUNNING DEMOS:

From this launcher:
  Select demo number (1-7)

Direct:
  python demo_01_live_detection.py
  python demo_06_batch_processing.py ./images
  python demo_07_video_processing.py video.mp4

📚 DOCUMENTATION:

Full documentation: README_DEMOS.md
YOLO docs: https://docs.ultralytics.com

💬 SUPPORT:

- GitHub issues
- SAIR Community channels
- Contact instructor

🇸🇩 SAIR - Building Sudan's AI Future 🇸🇩
    """
    print(help_text)
    input("\nPress Enter to continue...")

def run_demo(demo_key):
    """Run selected demo"""
    if demo_key not in DEMOS:
        print("❌ Invalid demo selection")
        return
    
    demo = DEMOS[demo_key]
    demo_file = Path(demo['file'])
    
    if not demo_file.exists():
        print(f"❌ Demo file not found: {demo['file']}")
        print("Make sure all demo files are in the same directory.")
        return
    
    print(f"\n🚀 Launching: {demo['name']}")
    print(f"📝 {demo['description']}")
    print(f"📋 Requirements: {demo['requirements']}\n")
    
    # Special handling for demos requiring arguments
    if demo_key == '6':
        # Batch processing
        folder = input("Enter image folder path (or . for current): ").strip()
        if not folder:
            folder = '.'
        args = [folder]
    elif demo_key == '7':
        # Video processing
        video = input("Enter video file path: ").strip()
        if not video:
            print("❌ Video path required")
            return
        args = [video]
    else:
        args = []
    
    try:
        subprocess.run([sys.executable, str(demo_file)] + args)
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")

def main():
    """Main launcher loop"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️ Some dependencies are missing.")
        choice = input("Install now? (y/n): ").lower()
        if choice == 'y':
            install_dependencies()
        else:
            print("Please install dependencies before running demos.")
            return
    
    while True:
        print_menu()
        choice = input("\n🎯 Select demo (1-7, i, h, q): ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Thanks for using SAIR demos! 🇸🇩\n")
            break
        elif choice == 'i':
            install_dependencies()
        elif choice == 'h':
            show_help()
        elif choice in DEMOS:
            run_demo(choice)
        else:
            print("❌ Invalid selection. Please choose 1-7, i, h, or q")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! 🇸🇩\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
