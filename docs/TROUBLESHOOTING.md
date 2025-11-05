# Troubleshooting Guide

Solutions to common problems and issues with the Advanced Lane Departure Warning System.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Camera Problems](#camera-problems)
- [Performance Issues](#performance-issues)
- [Detection Problems](#detection-problems)
- [Configuration Issues](#configuration-issues)
- [Audio Problems](#audio-problems)
- [Platform-Specific Issues](#platform-specific-issues)
- [Error Messages](#error-messages)

## Installation Issues

### Problem: "No module named 'cv2'"

**Cause**: OpenCV not installed or incorrectly installed

**Solutions**:

```bash
# Solution 1: Reinstall OpenCV
pip uninstall opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-python

# Solution 2: Check installation
python -c "import cv2; print(cv2.__version__)"

# Solution 3: Install in correct environment
# Make sure virtual environment is activated
pip list | grep opencv
```

### Problem: "DLL load failed" (Windows)

**Cause**: Missing Visual C++ Redistributable

**Solution**:

1. Download [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Install and restart computer
3. Reinstall OpenCV:
```bash
pip uninstall opencv-python
pip install opencv-python
```

### Problem: Pygame mixer errors

**Cause**: Missing audio libraries

**Solutions**:

```bash
# Linux
sudo apt-get install portaudio19-dev python3-dev
pip install --upgrade pygame

# macOS
brew install portaudio sdl2 sdl2_mixer
pip install --upgrade pygame

# Windows
pip uninstall pygame
pip install pygame

# Disable audio as workaround
python advanced_ldws.py --no-audio
```

### Problem: NumPy/SciPy compilation errors

**Cause**: Missing build tools

**Solutions**:

```bash
# Windows: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Linux
sudo apt-get install python3-dev build-essential

# macOS
xcode-select --install

# Alternative: Use pre-built wheels
pip install --only-binary :all: numpy scipy
```

## Camera Problems

### Problem: "Camera not detected" or "Failed to open video source"

**Diagnostic**:

```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Check camera permissions
# Windows: Settings → Privacy → Camera
# Linux: ls -l /dev/video*
# macOS: System Preferences → Security & Privacy → Camera
```

**Solutions**:

```bash
# Solution 1: Try different camera index
python advanced_ldws.py --input 1
python advanced_ldws.py --input 2

# Solution 2: Check if camera is used by another application
# Close other apps using camera (Zoom, Skype, etc.)

# Solution 3: Linux - Add user to video group
sudo usermod -a -G video $USER
# Logout and login again

# Solution 4: Test with simple script
python -c "
import cv2
cap = cv2.VideoCapture(0)
print('Camera opened:', cap.isOpened())
ret, frame = cap.read()
print('Frame captured:', ret)
cap.release()
"
```

### Problem: Black screen or no video

**Causes & Solutions**:

1. **Camera covered**: Check physical camera
2. **Wrong driver**: Use different backend
```bash
# Windows - try DirectShow
# Already implemented in code

# Linux - try different backend
python -c "import cv2; print(dir(cv2))" | grep CAP_

# Test with v4l2
sudo apt-get install v4l-utils
v4l2-ctl --list-devices
```

3. **Permission denied**: Fix permissions
```bash
# Linux
sudo chmod 666 /dev/video0
```

### Problem: Low frame rate or choppy video

**Solutions**:

```bash
# Solution 1: Reduce resolution
python advanced_ldws.py --resolution 640x480

# Solution 2: Check config
{
    "capture": {
        "resolution": {"width": 640, "height": 480},
        "fps": 30
    },
    "performance": {
        "buffer_size": 1
    }
}

# Solution 3: Close other applications

# Solution 4: Check USB connection
# Use USB 3.0 port for high-resolution cameras
```

## Performance Issues

### Problem: Low FPS (< 15 FPS)

**Diagnostic**:

```python
# Check processing time
python advanced_ldws.py --debug
# Watch debug panel for frame processing time
```

**Solutions**:

```json
// 1. Reduce resolution
{
    "capture": {
        "resolution": {"width": 640, "height": 480}
    }
}

// 2. Optimize processing
{
    "processing": {
        "frame_history": 8,
        "hough_threshold": 40,
        "kernel_size": 3
    }
}

// 3. Disable expensive operations
{
    "output": {
        "save_warnings": false,
        "visualization": {
            "show_roi": false
        }
    }
}

// 4. Disable GPU if slower
{
    "performance": {
        "enable_gpu": false,
        "buffer_size": 1
    }
}
```

### Problem: High CPU usage

**Solutions**:

1. **Reduce frame rate**:
```json
{
    "capture": {"fps": 20}
}
```

2. **Use hardware acceleration**:
```json
{
    "performance": {
        "enable_gpu": true,
        "use_threading": true
    }
}
```

3. **Monitor system**:
```bash
# Linux
top -p $(pgrep -f advanced_ldws.py)

# Windows
# Task Manager → Performance

# macOS
Activity Monitor
```

### Problem: High memory usage

**Solutions**:

```json
{
    "processing": {
        "frame_history": 10  // Reduce from 15
    },
    "performance": {
        "buffer_size": 2  // Reduce from 5
    },
    "output": {
        "save_warnings": false  // Disable to save RAM
    }
}
```

## Detection Problems

### Problem: No lanes detected

**Diagnostic Steps**:

1. **Enable debug mode**:
```bash
python advanced_ldws.py --debug
```

2. **Check edge detection**:
   - Look at debug panel
   - Verify edge pixels count
   - Check if lines are detected

**Solutions**:

```json
// 1. Adjust Canny thresholds (lower for more edges)
{
    "processing": {
        "canny_low": 25,
        "canny_high": 90
    }
}

// 2. Adjust Hough parameters (lower for more lines)
{
    "processing": {
        "hough_threshold": 20,
        "hough_min_line_length": 30
    }
}

// 3. Widen ROI
{
    "processing": {
        "roi_vertices": {
            "top_left": [0.3, 0.65],
            "top_right": [0.7, 0.65],
            "bottom_right": [0.95, 1.0],
            "bottom_left": [0.05, 1.0]
        }
    }
}

// 4. Enable adaptive mode
{
    "processing": {
        "adaptive_canny": true
    }
}
```

### Problem: False lane detections

**Symptoms**: Detecting shadows, road markings, or other objects as lanes

**Solutions**:

```json
// 1. Stricter detection
{
    "processing": {
        "hough_threshold": 40,
        "slope_threshold": 0.5,
        "canny_high": 140
    }
}

// 2. Tighter ROI
{
    "processing": {
        "roi_vertices": {
            "top_left": [0.42, 0.62],
            "top_right": [0.58, 0.62],
            "bottom_right": [0.88, 1.0],
            "bottom_left": [0.12, 1.0]
        }
    }
}

// 3. Increase smoothing
{
    "processing": {
        "frame_history": 20,
        "smoothing_window": 7
    }
}
```

### Problem: Jittery or unstable lane lines

**Solutions**:

```json
// Increase temporal smoothing
{
    "processing": {
        "frame_history": 25,
        "smoothing_window": 10
    }
}
```

### Problem: Lanes not detected on curves

**Solutions**:

```json
// 1. Adjust slope threshold
{
    "processing": {
        "slope_threshold": 0.3  // Lower for sharper curves
    }
}

// 2. Increase line gap tolerance
{
    "processing": {
        "hough_max_line_gap": 150
    }
}

// 3. Reduce min line length
{
    "processing": {
        "hough_min_line_length": 30
    }
}
```

## Configuration Issues

### Problem: Config file not loading

**Solutions**:

```bash
# Check file exists
ls -l ldws_config.json

# Validate JSON syntax
python -m json.tool ldws_config.json

# Check for UTF-8 encoding issues
file ldws_config.json

# Use absolute path
python advanced_ldws.py --config /full/path/to/config.json

# Check permissions
chmod 644 ldws_config.json
```

### Problem: Changes not taking effect

**Solutions**:

1. **Verify config is loaded**:
```bash
# Check logs
tail -f ldws.log
# Should see: "Configuration loaded from..."
```

2. **Restart application**

3. **Clear cache** (if any):
```bash
rm -rf __pycache__
```

## Audio Problems

### Problem: No audio alerts

**Diagnostic**:

```python
# Test audio system
python -c "
import pygame
pygame.mixer.init()
print('Mixer initialized:', pygame.mixer.get_init())
"
```

**Solutions**:

```bash
# Solution 1: Check configuration
{
    "warning_system": {
        "enable_audio": true,
        "audio_volume": 0.8
    }
}

# Solution 2: Test system audio
# Play a test sound outside the application

# Solution 3: Check audio dependencies
pip install --upgrade pygame

# Solution 4: Linux - check PulseAudio
pulseaudio --check
pulseaudio --start

# Solution 5: Workaround - disable audio
python advanced_ldws.py --no-audio
```

### Problem: Audio too loud/quiet

**Solution**:

```json
{
    "warning_system": {
        "audio_volume": 0.5  // Range: 0.0 to 1.0
    }
}
```

### Problem: Audio lag or delay

**Solutions**:

```json
// Reduce audio processing overhead
{
    "warning_system": {
        "warning_persistence": 2.0  // Increase cooldown
    }
}
```

## Platform-Specific Issues

### Windows

#### Problem: "Python not recognized"

**Solution**:
1. Add Python to PATH:
   - System Properties → Environment Variables
   - Add `C:\Python313\` to PATH
   - Restart terminal

#### Problem: Permission errors

**Solution**:
```bash
# Run as administrator (last resort)
# Or check antivirus isn't blocking
```

### Linux

#### Problem: "Permission denied" on /dev/video*

**Solution**:
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Or change permissions (temporary)
sudo chmod 666 /dev/video0

# Permanent solution - create udev rule
sudo nano /etc/udev/rules.d/99-camera.rules
# Add: KERNEL=="video[0-9]*", GROUP="video", MODE="0666"
sudo udevadm control --reload-rules
```

#### Problem: "No module named '_tkinter'"

**Solution**:
```bash
sudo apt-get install python3-tk
```

### macOS

#### Problem: Camera permission denied

**Solution**:
1. System Preferences → Security & Privacy → Camera
2. Enable for Terminal/Python
3. Restart terminal

#### Problem: "Operation not permitted"

**Solution**:
```bash
# Grant Full Disk Access to Terminal
# System Preferences → Security & Privacy → Full Disk Access
```

## Error Messages

### "ValueError: ROI vertices out of bounds"

**Solution**:
```json
// Ensure all vertices are between 0.0 and 1.0
{
    "processing": {
        "roi_vertices": {
            "top_left": [0.45, 0.6],   // Must be < 1.0
            "top_right": [0.55, 0.6],
            "bottom_right": [0.9, 1.0],
            "bottom_left": [0.1, 1.0]
        }
    }
}
```

### "cv2.error: (-215:Assertion failed)"

**Common causes & solutions**:

```python
# 1. Empty frame
# Check camera connection

# 2. Invalid kernel size
# Must be odd number: 3, 5, 7, 9...
{
    "processing": {"kernel_size": 5}
}

# 3. Invalid image dimensions
# Check resolution settings
```

### "FileNotFoundError: [Errno 2] No such file or directory"

**Solution**:
```bash
# Create required directories
mkdir -p output/lane_warnings
mkdir -p output/reports
mkdir -p output/snapshots

# Or let the script create them (already implemented)
```

### "json.decoder.JSONDecodeError"

**Solution**:
```bash
# Validate JSON syntax
python -m json.tool ldws_config.json

# Common issues:
# - Trailing commas
# - Missing quotes
# - Unescaped characters
```

## Getting Additional Help

If your problem isn't covered here:

### 1. Check Logs

```bash
# View detailed logs
cat ldws.log

# Follow logs in real-time
tail -f ldws.log
```

### 2. Run Diagnostics

```bash
# System information
python -c "
import sys, cv2, numpy, pygame
print('Python:', sys.version)
print('OpenCV:', cv2.__version__)
print('NumPy:', numpy.__version__)
print('Pygame:', pygame.version.ver)
"

# Test camera
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    print(f'Camera {i}: {cap.isOpened()}')
    cap.release()
"
```

### 3. Enable Verbose Logging

```python
# Edit advanced_ldws.py temporarily
logging.basicConfig(level=logging.DEBUG)
```

### 4. Create GitHub Issue

Include:
- Python version: `python --version`
- OS and version
- Error message and full traceback
- Configuration file (if custom)
- Steps to reproduce
- Output of diagnostic commands

### 5. Community Support

- GitHub Discussions
- Stack Overflow (tag: `lane-detection` `opencv` `python`)
- OpenCV Forum

## Preventive Measures

### Best Practices

1. **Regular Updates**:
```bash
git pull origin main
pip install --upgrade -r requirements.txt
```

2. **Test After Changes**:
```bash
python -m pytest tests/
python advanced_ldws.py --duration 30
```

3. **Backup Configurations**:
```bash
cp ldws_config.json ldws_config.backup.json
```

4. **Monitor Performance**:
```bash
# Check system resources during operation
# Adjust settings if needed
```

5. **Use Version Control**:
```bash
# Track configuration changes
git add ldws_config.json
git commit -m "Updated config for night driving"
```

---

**Still having issues?** Open an issue on GitHub with detailed information, and we'll help you resolve it!