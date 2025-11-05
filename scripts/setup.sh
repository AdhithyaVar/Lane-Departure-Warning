#!/bin/bash
# Advanced LDWS Setup Script for Linux/macOS
# Automates installation and environment setup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)

print_header "Advanced LDWS Setup Script"
echo "Operating System: $OS"
echo "Python Version Required: 3.13+"
echo ""

# Check Python installation
print_info "Checking Python installation..."
if command_exists python3.13; then
    PYTHON_CMD="python3.13"
    print_success "Python 3.13 found"
elif command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if (( $(echo "$PYTHON_VERSION >= 3.13" | bc -l) )); then
        PYTHON_CMD="python3"
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.13+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3.13+ not found"
    print_info "Please install Python 3.13 first"
    exit 1
fi

# Install system dependencies
print_header "Installing System Dependencies"

if [ "$OS" = "linux" ]; then
    print_info "Detected Linux, checking package manager..."
    
    if command_exists apt-get; then
        print_info "Using apt-get..."
        sudo apt-get update
        sudo apt-get install -y \
            python3-dev \
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
            libtbb2 \
            libtbb-dev \
            libdc1394-22-dev \
            portaudio19-dev \
            libsdl2-dev \
            libsdl2-mixer-dev
        print_success "System dependencies installed"
    elif command_exists dnf; then
        print_info "Using dnf..."
        sudo dnf install -y \
            python3-devel \
            opencv-devel \
            gtk3-devel \
            boost-devel \
            portaudio-devel \
            SDL2-devel
        print_success "System dependencies installed"
    else
        print_error "Unknown package manager"
        print_info "Please install dependencies manually"
    fi
    
    # Add user to video group for camera access
    print_info "Adding user to video group..."
    sudo usermod -a -G video $USER
    print_success "User added to video group (logout and login to apply)"
    
elif [ "$OS" = "macos" ]; then
    print_info "Detected macOS..."
    
    if ! command_exists brew; then
        print_error "Homebrew not found"
        print_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    print_info "Installing dependencies via Homebrew..."
    brew install python@3.13 opencv portaudio sdl2 sdl2_mixer
    print_success "System dependencies installed"
fi

# Create virtual environment
print_header "Setting Up Virtual Environment"

if [ -d "venv" ]; then
    print_info "Virtual environment already exists"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        print_info "Removed existing virtual environment"
    else
        print_info "Using existing virtual environment"
    fi
fi

if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_success "Pip upgraded"

# Install Python dependencies
print_header "Installing Python Dependencies"

if [ -f "requirements.txt" ]; then
    print_info "Installing from requirements.txt..."
    pip install -r requirements.txt
    print_success "Python dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Create necessary directories
print_header "Creating Directory Structure"

directories=(
    "output/lane_warnings"
    "output/reports"
    "output/snapshots"
    "output/processed"
    "data/videos"
    "data/calibration"
    "logs"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_success "Created $dir"
    fi
done

# Create .gitkeep files
touch output/lane_warnings/.gitkeep
touch output/reports/.gitkeep
touch output/snapshots/.gitkeep
touch data/videos/.gitkeep
touch data/calibration/.gitkeep

# Test installation
print_header "Testing Installation"

print_info "Testing Python imports..."
$PYTHON_CMD -c "
import sys
import cv2
import numpy as np
import pygame

print(f'Python: {sys.version}')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Pygame: {pygame.version.ver}')
" && print_success "All imports successful" || print_error "Import test failed"

# Check for cameras
print_info "Checking for cameras..."
$PYTHON_CMD -c "
import cv2
cameras = [i for i in range(5) if cv2.VideoCapture(i).isOpened()]
if cameras:
    print(f'Found cameras: {cameras}')
else:
    print('No cameras detected')
"

# Run tests
if [ -d "tests" ] && [ -n "$(ls -A tests/*.py 2>/dev/null)" ]; then
    print_info "Running unit tests..."
    if pytest tests/ -v; then
        print_success "All tests passed"
    else
        print_error "Some tests failed"
    fi
else
    print_info "No tests found, skipping..."
fi

# Final summary
print_header "Setup Complete!"

echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the application:"
echo "   python advanced_ldws.py"
echo ""
echo "3. For custom configuration:"
echo "   python advanced_ldws.py --config configs/config_city.json"
echo ""
echo "4. For help:"
echo "   python advanced_ldws.py --help"
echo ""

print_success "Setup completed successfully!"

# Note for Linux users
if [ "$OS" = "linux" ]; then
    echo ""
    print_info "NOTE: You may need to logout and login for camera permissions to take effect"
fi

echo ""
echo "For documentation, see docs/ directory"
echo "For issues, visit: https://github.com/yourusername/Advanced-LDWS/issues"
echo ""