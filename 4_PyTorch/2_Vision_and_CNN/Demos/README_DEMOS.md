# 🎓 SAIR Computer Vision Demos
## Standalone Python Demos for Hands-On Learning 🇸🇩

Welcome to the SAIR CV Demo Collection! These are self-contained Python scripts that demonstrate different aspects of computer vision with YOLO.

---

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download YOLO Models (Automatic)
The scripts will automatically download models on first run.

---

## 🎯 Demos Overview

### **Demo 1: Live Detection** 🎥
**File:** `demo_01_live_detection.py`  
**What it does:** Real-time object detection using your webcam  
**Learn:** YOLO inference, FPS calculation, real-time processing

```bash
python demo_01_live_detection.py
```

**Controls:**
- `q` - Quit
- `s` - Screenshot
- `c` - Toggle confidence display
- `+/-` - Adjust confidence threshold

---

### **Demo 2: Background Removal** 🎨
**File:** `demo_02_background_removal.py`  
**What it does:** Remove backgrounds in real-time (like Zoom!)  
**Learn:** Segmentation, mask processing, image compositing

```bash
python demo_02_background_removal.py
```

**Controls:**
- `q` - Quit
- `b` - Cycle background colors
- `s` - Save result
- `f` - Toggle blur effect

---

### **Demo 3: Pose Estimation** 🦾
**File:** `demo_03_pose_estimation.py`  
**What it does:** Track 17 body keypoints in real-time  
**Learn:** Pose estimation, skeleton visualization, movement tracking

```bash
python demo_03_pose_estimation.py
```

**Controls:**
- `q` - Quit
- `k` - Toggle keypoint numbers
- `s` - Toggle skeleton
- `r` - Record pose sequence
- `i` - Show keypoint info

---

### **Demo 4: Gesture Control** ✨
**File:** `demo_04_gesture_control.py`  
**What it does:** Control your computer with hand gestures!  
**Learn:** Gesture recognition, state machines, interaction

```bash
python demo_04_gesture_control.py
```

**Gestures:**
- ✋ Open Hand - Stop/Pause
- 👆 Index Up - Volume Up
- ✌️ Peace - Volume Down
- 👊 Fist - Select/Click
- 👍 Thumbs Up - Like
- 🙏 Prayer - Reset

---

### **Demo 5: Model Comparison** ⚖️
**File:** `demo_05_model_comparison.py`  
**What it does:** Compare different YOLO models side-by-side  
**Learn:** Performance tradeoffs, model selection

```bash
python demo_05_model_comparison.py
```

**Controls:**
- `q` - Quit
- `1/2/3` - Show individual models
- `a` - Show all models
- `s` - Save comparison
- `r` - Performance report

---

### **Demo 6: Batch Processing** 📁
**File:** `demo_06_batch_processing.py`  
**What it does:** Process entire folders of images  
**Learn:** Batch processing, file handling, report generation

```bash
python demo_06_batch_processing.py /path/to/images
```

**Options:**
```bash
-o OUTPUT_FOLDER  # Specify output folder
-m MODEL          # Model to use (yolov8n/s/m)
-c CONFIDENCE     # Confidence threshold
--no-images       # Skip saving annotated images
--no-csv          # Skip CSV report
--no-json         # Skip JSON report
```

**Example:**
```bash
python demo_06_batch_processing.py ./my_images -o ./results -m yolov8s -c 0.5
```

---

### **Demo 7: Video Processing** 🎬
**File:** `demo_07_video_processing.py`  
**What it does:** Process videos with object detection  
**Learn:** Video I/O, frame processing, encoding

```bash
python demo_07_video_processing.py input_video.mp4
```

**Options:**
```bash
-o OUTPUT_FILE    # Output video path
-m MODEL          # Model to use
-c CONFIDENCE     # Confidence threshold
--preview         # Show processing preview
--skip N          # Process every Nth frame
```

**Example:**
```bash
python demo_07_video_processing.py video.mp4 -m yolov8n --preview --skip 2
```

---

## 🎓 Learning Path

### **Beginner** (Start here!)
1. ✅ Demo 1 - Live Detection
2. ✅ Demo 6 - Batch Processing  
3. ✅ Demo 7 - Video Processing

### **Intermediate**
4. ✅ Demo 2 - Background Removal
5. ✅ Demo 5 - Model Comparison

### **Advanced**
6. ✅ Demo 3 - Pose Estimation
7. ✅ Demo 4 - Gesture Control

---

## 🛠️ Troubleshooting

### **Webcam not found**
```python
# Check available cameras
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()
```

### **Model download fails**
```bash
# Manually download models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
```

### **Slow performance**
- Use YOLOv8n (nano) instead of larger models
- Reduce image resolution
- Skip frames in video processing
- Use `--skip 2` or higher

### **CUDA/GPU issues**
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode if needed
export CUDA_VISIBLE_DEVICES=""
```

---

## 📊 Performance Tips

### **For Real-time Applications (Demos 1-4):**
- Use YOLOv8n (nano model)
- Set confidence to 0.5+ to reduce false positives
- Lower camera resolution to 640x480
- Close other applications

### **For Batch Processing (Demo 6):**
- Use larger models (YOLOv8m/l) for better accuracy
- Process in batches to optimize GPU usage
- Use multithreading for large datasets

### **For Video Processing (Demo 7):**
- Skip frames (`--skip 2`) for faster processing
- Use preview mode to check settings first
- Process shorter clips to test settings

---

## 🎨 Customization Ideas

### **Modify Demo 1 (Live Detection)**
- Add custom alert sounds when objects detected
- Count specific objects (e.g., people counter)
- Track objects across frames

### **Modify Demo 2 (Background Removal)**
- Add custom background images
- Create virtual backgrounds
- Green screen effects

### **Modify Demo 4 (Gesture Control)**
- Add new gesture types
- Control media players
- Create presentation controller
- Control smart home devices

### **Modify Demo 6 (Batch Processing)**
- Filter by specific classes
- Generate statistics graphs
- Create detection heatmaps

---

## 📚 Additional Resources

### **YOLO Documentation:**
- https://docs.ultralytics.com

### **SAIR Community:**
- Join our Discord/Telegram (link from instructor)
- Share your projects with #SAIR
- Help fellow learners

### **Next Steps:**
1. Complete all 7 demos
2. Modify demos for your use case
3. Build a custom project
4. Train YOLO on your own data (see Lecture 5B)
5. Deploy to production (see Lecture 5C)

---

## 🏆 Challenge Projects

Use these demos as building blocks:

1. **Security Camera System**
   - Use Demo 1 + alert system
   - Send notifications when person detected
   
2. **Virtual Meeting Assistant**
   - Use Demo 2 for background removal
   - Add gesture controls (Demo 4)

3. **Sports Analytics**
   - Use Demo 7 on game footage
   - Track player movements (Demo 3)

4. **Traffic Monitor**
   - Use Demo 6 on traffic camera images
   - Count vehicles by type

5. **Gesture-Controlled Game**
   - Use Demo 4 as base
   - Create interactive experience

---

## 🐛 Known Issues

1. **Windows:** Some demos may require admin rights for webcam
2. **macOS:** May need to grant camera permissions in System Preferences
3. **Linux:** May need to add user to `video` group

---

## 📝 License

MIT License - Free for educational and commercial use

---

## 🙏 Credits

**Created by:** SAIR Community 🇸🇩  
**Instructor:** Mohammed Awad Ahmed (Silva)  
**Course:** Ultimate Applied Deep Learning with PyTorch

---

## 💬 Support

**Issues?** Open an issue on GitHub or contact instructor  
**Questions?** Join SAIR community channels  
**Improvements?** Pull requests welcome!

---

**🇸🇩 Building Sudan's AI Future, One Model at a Time 🇸🇩**

---

## Quick Start (TL;DR)

```bash
# Install
pip install -r requirements.txt

# Run demos
python demo_01_live_detection.py          # Webcam detection
python demo_02_background_removal.py      # Background removal
python demo_03_pose_estimation.py         # Pose tracking
python demo_04_gesture_control.py         # Gesture control
python demo_05_model_comparison.py        # Model comparison
python demo_06_batch_processing.py .      # Batch processing
python demo_07_video_processing.py video.mp4  # Video processing
```

**Enjoy learning! 🚀**
