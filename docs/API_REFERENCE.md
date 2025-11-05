# API Reference

Complete API documentation for the Advanced Lane Departure Warning System.

## Table of Contents

- [Core Classes](#core-classes)
- [Configuration](#configuration)
- [Lane Detection](#lane-detection)
- [Audio System](#audio-system)
- [Metrics](#metrics)
- [Utilities](#utilities)
- [Usage Examples](#usage-examples)

## Core Classes

### LaneDetectionConfig

Configuration manager for lane detection parameters.

```python
class LaneDetectionConfig:
    def __init__(self, config_file: str = "ldws_config.json")
    def load_config(self) -> None
    def save_config(self) -> None
    def get(self, key: str, default: Any = None) -> Any
```

#### Methods

**`__init__(config_file: str)`**

Initialize configuration from JSON file.

**Parameters:**
- `config_file` (str): Path to configuration file

**Example:**
```python
config = LaneDetectionConfig("configs/config_night.json")
```

---

**`load_config() -> None`**

Load configuration from file with fallback to defaults.

**Raises:**
- `FileNotFoundError`: If config file doesn't exist (creates default)
- `JSONDecodeError`: If config file is invalid JSON

---

**`save_config() -> None`**

Save current configuration to file.

---

**`get(key: str, default: Any = None) -> Any`**

Get configuration value with default fallback.

**Parameters:**
- `key` (str): Configuration key (e.g., 'processing_canny_low')
- `default` (Any): Default value if key not found

**Returns:**
- Configuration value or default

**Example:**
```python
threshold = config.get('processing_canny_low', 40)
```

#### Attributes

After loading, all config values are accessible as flat attributes:

```python
config.processing_canny_low        # int
config.processing_canny_high       # int
config.warning_system_enable_audio # bool
config.capture_resolution_width    # int
config.roi_vertices                # dict
```

---

### AdvancedLaneDetector

Main lane detection and tracking system.

```python
class AdvancedLaneDetector:
    def __init__(self, config: LaneDetectionConfig)
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, Optional[float]]
    def region_of_interest(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    def adaptive_canny(self, img: np.ndarray) -> np.ndarray
    def fit_lane_line(self, lines: List[np.ndarray]) -> Optional[np.ndarray]
    def smooth_lane_lines(self, left_fit, right_fit) -> Tuple
    def calculate_lateral_offset(self, left_fit, right_fit, width, height) -> Tuple[Optional[float], float]
    def draw_advanced_lanes(self, img, left_fit, right_fit, status, offset, confidence) -> np.ndarray
    def save_final_report(self) -> str
    def cleanup(self) -> None
```

#### Constructor

**`__init__(config: LaneDetectionConfig)`**

Initialize lane detector with configuration.

**Parameters:**
- `config` (LaneDetectionConfig): Configuration object

**Example:**
```python
config = LaneDetectionConfig()
detector = AdvancedLaneDetector(config)
```

#### Methods

**`process_frame(frame: np.ndarray) -> Tuple[np.ndarray, str, Optional[float]]`**

Process a single frame and detect lanes.

**Parameters:**
- `frame` (np.ndarray): Input frame (BGR format)

**Returns:**
- `tuple`:
  - `output` (np.ndarray): Annotated frame with visualizations
  - `status` (str): Lane keeping status ("CENTER", "WARNING", "CRITICAL", "NO_LANES", "UNCERTAIN")
  - `offset` (Optional[float]): Lateral offset from lane center (-1.0 to 1.0, None if no detection)

**Example:**
```python
output, status, offset = detector.process_frame(frame)

if status == "CRITICAL":
    print(f"Lane departure detected! Offset: {offset:.2f}")
```

---

**`region_of_interest(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`**

Apply region of interest mask to image.

**Parameters:**
- `img` (np.ndarray): Grayscale image

**Returns:**
- `tuple`:
  - `masked_image` (np.ndarray): Image with ROI applied
  - `polygon` (np.ndarray): ROI polygon vertices

**Example:**
```python
edges = cv2.Canny(gray, 50, 150)
masked_edges, roi_poly = detector.region_of_interest(edges)
```

---

**`adaptive_canny(img: np.ndarray) -> np.ndarray`**

Apply adaptive Canny edge detection.

**Parameters:**
- `img` (np.ndarray): Grayscale image

**Returns:**
- `np.ndarray`: Edge-detected image

**Example:**
```python
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = detector.adaptive_canny(blur)
```

---

**`fit_lane_line(lines: List[np.ndarray]) -> Optional[np.ndarray]`**

Fit polynomial through detected line segments.

**Parameters:**
- `lines` (List[np.ndarray]): List of line segments from Hough transform

**Returns:**
- `Optional[np.ndarray]`: Polynomial coefficients [slope, intercept] or None

**Example:**
```python
left_fit = detector.fit_lane_line(left_lines)
if left_fit is not None:
    # Use polynomial: x = left_fit[0] * y + left_fit[1]
    pass
```

---

**`smooth_lane_lines(left_fit, right_fit) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]`**

Apply temporal smoothing to reduce jitter.

**Parameters:**
- `left_fit` (Optional[np.ndarray]): Left lane polynomial
- `right_fit` (Optional[np.ndarray]): Right lane polynomial

**Returns:**
- `tuple`: (smoothed_left_fit, smoothed_right_fit)

---

**`calculate_lateral_offset(...) -> Tuple[Optional[float], float]`**

Calculate vehicle's lateral offset from lane center.

**Parameters:**
- `left_fit` (Optional[np.ndarray]): Left lane polynomial
- `right_fit` (Optional[np.ndarray]): Right lane polynomial
- `img_width` (int): Image width
- `img_height` (int): Image height

**Returns:**
- `tuple`:
  - `offset` (Optional[float]): Normalized lateral offset
  - `confidence` (float): Detection confidence (0.0 to 1.0)

**Offset Values:**
- Negative: Vehicle is left of center
- Zero: Perfect center
- Positive: Vehicle is right of center
- Range: -1.0 (far left) to +1.0 (far right)

---

**`draw_advanced_lanes(...) -> np.ndarray`**

Draw lane visualizations on frame.

**Parameters:**
- `img` (np.ndarray): Input frame
- `left_fit`, `right_fit`: Lane polynomials
- `status` (str): Lane status
- `offset` (Optional[float]): Lateral offset
- `confidence` (float): Detection confidence

**Returns:**
- `np.ndarray`: Annotated frame

---

**`save_final_report() -> str`**

Generate and save final performance report.

**Returns:**
- `str`: Path to saved report file

---

**`cleanup() -> None`**

Release resources and cleanup.

#### Attributes

```python
detector.config           # LaneDetectionConfig
detector.audio           # AudioAlert
detector.metrics         # LaneMetrics
detector.paused          # bool
detector.debug_mode      # bool
detector.last_status     # str
```

---

### AudioAlert

Audio alert system for warnings.

```python
class AudioAlert:
    def __init__(self, enabled: bool = True, volume: float = 0.8)
    def warning_alert(self) -> None
    def critical_alert(self) -> None
    def cleanup(self) -> None
```

#### Constructor

**`__init__(enabled: bool = True, volume: float = 0.8)`**

Initialize audio system.

**Parameters:**
- `enabled` (bool): Enable audio alerts
- `volume` (float): Volume level (0.0 to 1.0)

#### Methods

**`warning_alert() -> None`**

Play warning level alert (single beep, 800Hz).

---

**`critical_alert() -> None`**

Play critical level alert (double beep, 1000Hz + 1200Hz).

---

**`cleanup() -> None`**

Cleanup audio resources.

#### Attributes

```python
audio.enabled           # bool
audio.volume           # float
audio.warning_count    # int
audio.critical_count   # int
```

---

### LaneMetrics

Performance metrics tracking and analysis.

```python
class LaneMetrics:
    def __init__(self)
    def update(self, status: str, lateral_offset: float = 0, confidence: float = 1.0) -> None
    def update_frame_time(self, frame_time: float) -> None
    def get_fps(self) -> float
    def get_statistics(self) -> Dict[str, float]
    def calculate_score(self) -> float
    def get_report(self) -> str
```

#### Constructor

**`__init__()`**

Initialize metrics tracker.

#### Methods

**`update(status: str, lateral_offset: float = 0, confidence: float = 1.0) -> None`**

Update metrics with current frame data.

**Parameters:**
- `status` (str): Lane status
- `lateral_offset` (float): Current offset
- `confidence` (float): Detection confidence

---

**`update_frame_time(frame_time: float) -> None`**

Track frame processing time for FPS calculation.

**Parameters:**
- `frame_time` (float): Frame processing time in seconds

---

**`get_fps() -> float`**

Calculate current frames per second.

**Returns:**
- `float`: Current FPS

---

**`get_statistics() -> Dict[str, float]`**

Get comprehensive statistics.

**Returns:**
- `dict`: Statistics including:
  - `center_percentage`: % of frames in center
  - `warning_percentage`: % of warning frames
  - `critical_percentage`: % of critical frames
  - `detection_rate`: % of successful detections
  - `avg_offset`: Average lateral offset
  - `max_offset`: Maximum offset
  - `std_offset`: Offset standard deviation
  - `avg_confidence`: Average detection confidence
  - `avg_fps`: Average FPS

**Example:**
```python
stats = metrics.get_statistics()
print(f"Detection rate: {stats['detection_rate']:.1f}%")
print(f"Average FPS: {stats['avg_fps']:.1f}")
```

---

**`calculate_score() -> float`**

Calculate overall performance score.

**Returns:**
- `float`: Score from 0 to 100

**Scoring:**
- 95-100: Exceptional
- 85-94: Excellent
- 75-84: Good
- 60-74: Acceptable
- 40-59: Needs Improvement
- 0-39: Poor

---

**`get_report() -> str`**

Generate comprehensive text report.

**Returns:**
- `str`: Formatted report text

#### Attributes

```python
metrics.total_frames          # int
metrics.center_frames         # int
metrics.warning_frames        # int
metrics.critical_frames       # int
metrics.no_detection_frames   # int
metrics.lane_positions        # deque
metrics.start_time           # float
```

---

## Configuration

### Configuration File Format

```json
{
    "processing": {
        "canny_low": 40,
        "canny_high": 120,
        "hough_threshold": 30,
        "roi_vertices": {
            "top_left": [0.45, 0.6],
            "top_right": [0.55, 0.6],
            "bottom_right": [0.9, 1.0],
            "bottom_left": [0.1, 1.0]
        }
    },
    "warning_system": {
        "warning_threshold": 0.15,
        "critical_threshold": 0.25,
        "enable_audio": true
    },
    "capture": {
        "resolution": {"width": 1280, "height": 720},
        "fps": 30
    }
}
```

### Accessing Configuration

```python
# Load configuration
config = LaneDetectionConfig("my_config.json")

# Access nested values (flattened)
canny_low = config.processing_canny_low
audio_enabled = config.warning_system_enable_audio

# Or use get method
threshold = config.get('processing_hough_threshold', 30)

# Modify and save
config.config_data['processing']['canny_low'] = 35
config.save_config()
```

---

## Usage Examples

### Basic Usage

```python
import cv2
from advanced_ldws import LaneDetectionConfig, AdvancedLaneDetector

# Initialize
config = LaneDetectionConfig()
detector = AdvancedLaneDetector(config)

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    output, status, offset = detector.process_frame(frame)
    
    # Display
    cv2.imshow('LDWS', output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
detector.cleanup()

# Print report
print(detector.metrics.get_report())
```

### Custom Configuration

```python
# Load custom config
config = LaneDetectionConfig("configs/config_night.json")

# Modify for specific needs
config.config_data['warning_system']['warning_threshold'] = 0.12
config._flatten_config()

# Initialize detector
detector = AdvancedLaneDetector(config)
```

### Video File Processing

```python
# Process video file
cap = cv2.VideoCapture("test_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    output, status, offset = detector.process_frame(frame)
    
    # Save if warning/critical
    if status in ["WARNING", "CRITICAL"]:
        cv2.imwrite(f"frame_{status}.jpg", output)
```

### Batch Processing

```python
import os
from pathlib import Path

video_dir = Path("data/videos")
output_dir = Path("output/processed")

for video_file in video_dir.glob("*.mp4"):
    print(f"Processing {video_file.name}...")
    
    cap = cv2.VideoCapture(str(video_file))
    detector = AdvancedLaneDetector(config)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        output, status, offset = detector.process_frame(frame)
    
    # Save report
    report_file = output_dir / f"{video_file.stem}_report.txt"
    with open(report_file, 'w') as f:
        f.write(detector.metrics.get_report())
    
    cap.release()
    detector.cleanup()
```

### Real-time Metrics

```python
import time

last_print = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    output, status, offset = detector.process_frame(frame)
    
    # Print metrics every 5 seconds
    if time.time() - last_print > 5:
        stats = detector.metrics.get_statistics()
        print(f"FPS: {stats['avg_fps']:.1f}, "
              f"Center: {stats['center_percentage']:.1f}%, "
              f"Score: {detector.metrics.calculate_score():.1f}")
        last_print = time.time()
    
    cv2.imshow('LDWS', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Custom Alert Handling

```python
def custom_alert_handler(status, offset):
    """Custom handler for lane departure alerts"""
    if status == "WARNING":
        print(f"⚠ WARNING: Drifting {offset:.2f}")
        # Send to logging system
        # Trigger vehicle system
    elif status == "CRITICAL":
        print(f"✗ CRITICAL: Lane departure {offset:.2f}!")
        # Emergency alert
        # Log incident

# Use in processing loop
output, status, offset = detector.process_frame(frame)
if status in ["WARNING", "CRITICAL"]:
    custom_alert_handler(status, offset)
```

### Integration with Other Systems

```python
class VehicleSystem:
    def __init__(self):
        self.config = LaneDetectionConfig()
        self.detector = AdvancedLaneDetector(self.config)
        
    def process_camera_feed(self, frame):
        """Process camera frame and return control signals"""
        output, status, offset = self.detector.process_frame(frame)
        
        # Generate control signals
        control = {
            'steering_correction': self.calculate_steering(offset),
            'warning_level': status,
            'confidence': self.get_confidence()
        }
        
        return output, control
    
    def calculate_steering(self, offset):
        """Calculate steering correction based on offset"""
        if offset is None:
            return 0.0
        # Simple proportional control
        return -offset * 0.5  # Negative because offset is opposite to correction
    
    def get_confidence(self):
        """Get current detection confidence"""
        stats = self.detector.metrics.get_statistics()
        return stats.get('avg_confidence', 0.0)
```

---

## Type Hints

Full type annotations are provided for better IDE support:

```python
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, Optional[float]]:
    ...

def get_statistics(self) -> Dict[str, float]:
    ...
```

---

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Configuration guide
- [INSTALLATION.md](INSTALLATION.md) - Installation instructions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving

---

For questions about the API, please open an issue on GitHub.