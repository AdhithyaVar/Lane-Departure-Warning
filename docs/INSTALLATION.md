# Installation Guide

Complete installation instructions for the Advanced Lane Departure Warning System.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10+, Ubuntu 20.04+, macOS 11+ |
| **Python** | 3.13 or higher |
| **RAM** | 4GB |
| **Storage** | 500MB free space |
| **Camera** | Any USB webcam or integrated camera |
| **CPU** | Intel Core i5 / AMD Ryzen 5 or equivalent |

### Recommended Requirements

| Component | Requirement |
|-----------|-------------|
| **RAM** | 8GB or more |
| **Storage** | 2GB free space |
| **GPU** | NVIDIA GTX 1050+ with CUDA support |
| **Camera** | 720p or 1080p webcam |
| **CPU** | Intel Core i7 / AMD Ryzen 7 or better |

## Installation Steps

### 1. Install Python 3.13

#### Windows

Download Python 3.13 from [python.org](https://www.python.org/downloads/):

```bash
# Verify installation
python --version
# Should output: Python 3.13.x
```

**Important**: Check "Add Python to PATH" during installation.

#### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python 3.13
sudo apt install python3.13 python3.13-venv python3.13-dev

# Verify installation
python3.13 --version
```

#### macOS

```bash
# Using Homebrew
brew install python@3.13

# Verify installation
python3.13 --version
```

### 2. Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/yourusername/Advanced-LDWS.git
cd Advanced-LDWS

# Or using SSH
git clone git@github.com:yourusername/Advanced-LDWS.git
cd Advanced-LDWS
```

### 3. Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3.13 -m venv venv
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal.

### 4. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
# Run quick test
python -c "import cv2; import numpy; import pygame; print('✓ All imports successful')"

# Check OpenCV version
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# List available cameras
python -c "import cv2; print('Available cameras:', [i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

## Platform-Specific Instructions

### Windows

#### Install Visual C++ Redistributable

Some dependencies require Visual C++ runtime:

1. Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Run the installer
3. Restart your computer

#### Camera Permissions

1. Go to **Settings → Privacy → Camera**
2. Enable "Allow desktop apps to access your camera"

#### DirectShow Support

For better camera performance, ensure DirectShow is available (included in Windows by default).

### Linux

#### Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3.13-dev \
    python3-pip \
    libopencv-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    portaudio19-dev

# Fedora/RHEL
sudo dnf install -y \
    python3.13-devel \
    opencv-devel \
    gtk3-devel \
    boost-devel \
    portaudio-devel
```

#### Camera Permissions

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Apply changes (logout and login, or run)
newgrp video

# Check camera devices
ls -l /dev/video*
```

#### SDL2 for Pygame (if needed)

```bash
sudo apt-get install -y libsdl2-dev libsdl2-mixer-dev
```

### macOS

#### Install Xcode Command Line Tools

```bash
xcode-select --install
```

#### Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Install System Dependencies

```bash
brew install python@3.13
brew install opencv
brew install portaudio
brew install sdl2 sdl2_mixer
```

#### Camera Permissions

1. Go to **System Preferences → Security & Privacy → Privacy → Camera**
2. Grant access to Terminal or your Python IDE

## GPU Acceleration (Optional)

### NVIDIA CUDA Setup

For GPU-accelerated processing with NVIDIA graphics cards:

#### Windows & Linux

1. **Install NVIDIA Drivers**
   - Download from [NVIDIA](https://www.nvidia.com/Download/index.aspx)

2. **Install CUDA Toolkit 11.8+**
   ```bash
   # Verify CUDA installation
   nvcc --version
   ```

3. **Install cuDNN**
   - Download from [NVIDIA Developer](https://developer.nvidia.com/cudnn)

4. **Install TensorFlow with GPU support**
   ```bash
   pip install tensorflow[and-cuda]
   ```

5. **Verify GPU Detection**
   ```bash
   python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
   ```

## Verification

### Quick Test

```bash
# Run system diagnostics
python -c "
import cv2
import numpy as np
import pygame
import sys

print('='*50)
print('System Diagnostics')
print('='*50)
print(f'Python version: {sys.version}')
print(f'OpenCV version: {cv2.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Pygame version: {pygame.version.ver}')
print('='*50)
print('✓ All core libraries installed successfully')
"
```

### Full Test

```bash
# Run unit tests
python -m pytest tests/ -v

# Run main application (30 seconds)
python advanced_ldws.py --duration 30
```

## Troubleshooting

### Common Issues

#### Issue: "No module named 'cv2'"

**Solution**:
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

#### Issue: "Camera not detected"

**Solution**:
```bash
# List all cameras
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"

# Try different camera index
python advanced_ldws.py --input 1
```

#### Issue: "pygame.mixer not available"

**Solution**:
```bash
# Linux
sudo apt-get install portaudio19-dev
pip install --upgrade pygame

# macOS
brew install portaudio
pip install --upgrade pygame

# Windows
pip uninstall pygame
pip install pygame
```

#### Issue: "Import Error: DLL load failed"

**Solution (Windows)**:
1. Install Visual C++ Redistributable
2. Restart computer
3. Reinstall OpenCV: `pip install --force-reinstall opencv-python`

#### Issue: Low FPS / Performance

**Solutions**:
1. Reduce resolution in config: `"width": 640, "height": 480`
2. Disable GPU if causing issues: `"enable_gpu": false`
3. Close other applications
4. Use hardware acceleration if available

### Getting Help

If you encounter issues not covered here:

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Search existing [GitHub Issues](https://github.com/yourusername/Advanced-LDWS/issues)
3. Create a new issue with:
   - Python version
   - Operating system
   - Error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. Read [CONFIGURATION.md](CONFIGURATION.md) to customize settings
2. Review [API_REFERENCE.md](API_REFERENCE.md) for development
3. Run the application: `python advanced_ldws.py`

## Updating

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Run tests
python -m pytest tests/
```

---

**Installation Complete!** 🎉

Now you're ready to use the Advanced LDWS. Run `python advanced_ldws.py` to start.