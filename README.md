# 🚁 MuJoCo Drone Simulation with Camera Recording

A complete interactive drone simulation environment built with MuJoCo, featuring controllable quadrotor with multiple camera views, recording capabilities, and real-time controls.

## ✨ Features

- **🎮 Interactive 3D Controls** - Real-time drone control with WASD keys
- **📹 Multiple Camera Views** - External, drone-mounted, FPV, and top-down perspectives
- **🎥 Video Recording** - Record flights from all camera angles simultaneously
- **🖥️ WSL Compatible** - Works in Windows Subsystem for Linux
- **⚡ Real-time Physics** - Realistic quadrotor dynamics and motor mixing
- **🎯 Easy Controls** - Intuitive keyboard controls for all flight maneuvers

## 🚀 Quick Start

### **WORKING DRONE SIMULATOR**
```bash
python3 drone_working.py
```
**This is the working version with properly functioning controls!**

## 🎮 Complete Controls Reference

### **Flight Controls (WASD Style)**
| Key | Action | Description |
|-----|--------|-------------|
| **W** | Forward | Pitch forward |
| **S** | Backward | Pitch backward |
| **A** | Left | Roll left |
| **D** | Right | Roll right |
| **Q** | Yaw Left | Rotate counterclockwise |
| **E** | Yaw Right | Rotate clockwise |
| **Z** | Up | Increase altitude |
| **X** | Down | Decrease altitude |

### **System Controls**
| Key | Action | Description |
|-----|--------|-------------|
| **Space** | Hover | Stop all movement (emergency stop) |
| **R** | Reset | Return drone to starting position |
| **T** | Toggle | Switch between active/hover mode |
| **ESC** | Exit | Close simulation |

### **Camera Controls**
| Key | Action | Views Available |
|-----|--------|-----------------|
| **[** | Previous Camera | Cycles backward through camera views |
| **]** | Next Camera | Cycles forward through camera views |

**Camera Views:**
1. **External View** - Third-person perspective (default)
2. **Drone Camera** - Camera attached to drone body
3. **FPV Camera** - First-person view (pilot perspective)
4. **Top-Down View** - Bird's eye view for navigation

### **Camera Usage**
- Use **[** and **]** keys to switch between camera views
- **Mouse drag** to rotate camera view
- **Right-click + drag** to pan
- **Scroll wheel** to zoom in/out

## 🛩️ How to Fly

### **Basic Flying**
1. **Start simulation:** `python3 drone_working.py`
2. **Take off:** Hold **Z** to lift off
3. **Fly around:** Use **WASD** for movement
4. **Land safely:** Use **X** to descend gently

### **First-Person Flying (FPV)**
1. **Take off** in external view first
2. **Press ]** to cycle to "FPV Camera"
3. **Fly from drone's perspective** using WASD
4. **Switch back** with **[** or **]** if needed

### **Advanced Maneuvers**
- **Hover in place:** Press **Space** for instant stop
- **Precise control:** Quick taps for small movements, hold for continuous
- **Emergency reset:** Press **R** if you crash or get lost

## 📁 Available Simulations

| File | Description | Purpose |
|------|-------------|---------|
| `drone_working.py` | **🌟 MAIN SIMULATOR** - Working controls and camera switching | Interactive 3D drone flying |
| `drone_text_ui.py` | Web-based telemetry interface | Remote monitoring via browser |
| `record_flight.py` | Pre-programmed flight recording | Creating demo videos |
| `test_simulation.py` | Basic functionality test | Troubleshooting setup |

## 🎥 Recording and Screenshots

### **Automatic Recording**
- Videos are automatically saved during simulation
- Multiple camera angles recorded simultaneously
- Files saved in `recordings/` directory

### **Manual Screenshots**
```bash
python3 record_flight.py  # Creates demo flight videos
```

### **Video Output**
- **Format:** MP4 (H.264)
- **Resolution:** 640x480
- **Frame Rate:** 30 FPS
- **Cameras:** All 4 perspectives included

## 🖥️ WSL Setup (Windows Users)

### **If You Get Display Errors:**
1. **Install X11 server:** VcXsrv, Xming, or X410
2. **Install graphics libraries:**
   ```bash
   sudo apt update
   sudo apt install libgl1-mesa-glx libglu1-mesa libxrandr2 libxss1
   sudo apt install libgtk2.0-dev pkg-config
   ```
3. **Set display:** `export DISPLAY=:0`
4. **Enable access control** in X11 server settings

### **Alternative: Use Web Interface**
If 3D view doesn't work:
```bash
python3 drone_text_ui.py
# Then open http://localhost:8080 in browser
```

## 🔧 Requirements

- **Python 3.10+**
- **MuJoCo 3.x** (already installed)
- **OpenCV** (`pip install opencv-python`)
- **NumPy** (already installed)

## 🎯 Pro Tips

### **Flying Tips**
- **Start gentle:** Use quick taps before holding keys
- **Use hover:** Press Space frequently to stabilize
- **FPV flying:** Start in external view, then switch to FPV
- **Emergency reset:** Press R if you lose control

### **Camera Tips**
- **Watch the circles:** Bottom indicators show active camera
- **Try all views:** Each camera offers different perspectives
- **FPV for immersion:** Best experience for first-person flying
- **External for learning:** Easiest to see drone orientation

### **Performance Tips**
- **Close other programs** for best performance
- **Use fullscreen** for immersive experience
- **Try different cameras** if one view is slow

## 🚁 Ready to Fly!

**Start with this command:**
```bash
python3 drone_working.py
```

**Then:**
1. Hold **Z** to take off
2. Use **WASD** to fly around
3. Press **[** or **]** to switch camera views
4. Press **Space** to hover
5. Have fun! 🎉

Enjoy your drone simulation! 🚁✨