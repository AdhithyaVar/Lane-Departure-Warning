# Configuration Guide

Comprehensive guide to configuring the Advanced Lane Departure Warning System for optimal performance.

## Table of Contents

- [Configuration File Structure](#configuration-file-structure)
- [Processing Parameters](#processing-parameters)
- [Warning System](#warning-system)
- [Capture Settings](#capture-settings)
- [Output Options](#output-options)
- [Performance Tuning](#performance-tuning)
- [Environment-Specific Configs](#environment-specific-configs)
- [Best Practices](#best-practices)

## Configuration File Structure

Configuration files use JSON format with nested sections:

```json
{
    "processing": { ... },
    "warning_system": { ... },
    "capture": { ... },
    "output": { ... },
    "performance": { ... },
    "environment": { ... }
}
```

### Loading Custom Configurations

```bash
# Use default configuration
python advanced_ldws.py

# Use custom configuration
python advanced_ldws.py --config path/to/custom_config.json

# Use environment-specific config
python advanced_ldws.py --config configs/config_city.json
python advanced_ldws.py --config configs/config_highway.json
python advanced_ldws.py --config configs/config_night.json
```

## Processing Parameters

Controls lane detection algorithms and image processing.

### Edge Detection

```json
"processing": {
    "canny_low": 40,
    "canny_high": 120,
    "adaptive_canny": false
}
```

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `canny_low` | int | 40 | 10-100 | Lower threshold for Canny edge detection |
| `canny_high` | int | 120 | 50-250 | Upper threshold for Canny edge detection |
| `adaptive_canny` | bool | false | - | Use automatic threshold calculation |

**Tuning Guide:**
- **Low light conditions**: Decrease both thresholds (30, 90)
- **Bright conditions**: Increase both thresholds (50, 150)
- **Noisy images**: Increase lower threshold
- **Weak edges**: Decrease both thresholds

### Hough Transform

```json
"processing": {
    "hough_threshold": 30,
    "hough_min_line_length": 40,
    "hough_max_line_gap": 100
}
```

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `hough_threshold` | int | 30 | 10-100 | Minimum votes for line detection |
| `hough_min_line_length` | int | 40 | 20-100 | Minimum line segment length (pixels) |
| `hough_max_line_gap` | int | 100 | 50-200 | Maximum gap between line segments |

**Tuning Guide:**
- **More lines detected**: Decrease `hough_threshold`
- **Fewer false positives**: Increase `hough_threshold`
- **Dashed lanes**: Increase `hough_max_line_gap`
- **Short line segments**: Decrease `hough_min_line_length`

### Region of Interest (ROI)

```json
"processing": {
    "roi_vertices": {
        "top_left": [0.45, 0.6],
        "top_right": [0.55, 0.6],
        "bottom_right": [0.9, 1.0],
        "bottom_left": [0.1, 1.0]
    }
}
```

**Parameters:**

Coordinates are normalized (0.0 to 1.0) relative to frame dimensions.

**Tuning Guide:**
- **Narrow roads**: Decrease horizontal spread
- **Wide highways**: Increase horizontal spread
- **Curved roads**: Adjust top vertices closer
- **Straight roads**: Move top vertices farther

**Example Presets:**

```json
// City (tight roads)
"roi_vertices": {
    "top_left": [0.4, 0.65],
    "top_right": [0.6, 0.65],
    "bottom_right": [0.95, 1.0],
    "bottom_left": [0.05, 1.0]
}

// Highway (wide view)
"roi_vertices": {
    "top_left": [0.35, 0.6],
    "top_right": [0.65, 0.6],
    "bottom_right": [0.9, 1.0],
    "bottom_left": [0.1, 1.0]
}
```

### Lane Filtering

```json
"processing": {
    "slope_threshold": 0.4,
    "kernel_size": 5,
    "frame_history": 15,
    "smoothing_window": 5
}
```

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `slope_threshold` | float | 0.4 | 0.2-0.8 | Minimum absolute slope for valid lane |
| `kernel_size` | int | 5 | 3-9 | Gaussian blur kernel size (must be odd) |
| `frame_history` | int | 15 | 5-30 | Number of frames for temporal smoothing |
| `smoothing_window` | int | 5 | 3-10 | Moving average window size |

**Tuning Guide:**
- **Sharp turns**: Decrease `slope_threshold` (0.3)
- **Straight roads**: Increase `slope_threshold` (0.5)
- **Jittery detection**: Increase `frame_history`
- **Faster response**: Decrease `frame_history`

## Warning System

Controls alert thresholds and audio feedback.

```json
"warning_system": {
    "warning_threshold": 0.15,
    "critical_threshold": 0.25,
    "time_to_warning": 0.5,
    "warning_persistence": 1.0,
    "enable_audio": true,
    "audio_volume": 0.8
}
```

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `warning_threshold` | float | 0.15 | 0.05-0.30 | Lateral offset for warning (normalized) |
| `critical_threshold` | float | 0.25 | 0.10-0.40 | Lateral offset for critical alert |
| `time_to_warning` | float | 0.5 | 0.1-2.0 | Time before first warning (seconds) |
| `warning_persistence` | float | 1.0 | 0.5-3.0 | Time between repeated warnings |
| `enable_audio` | bool | true | - | Enable audio alerts |
| `audio_volume` | float | 0.8 | 0.0-1.0 | Audio alert volume |

**Sensitivity Presets:**

```json
// Conservative (less sensitive)
{
    "warning_threshold": 0.20,
    "critical_threshold": 0.30
}

// Balanced (default)
{
    "warning_threshold": 0.15,
    "critical_threshold": 0.25
}

// Aggressive (very sensitive)
{
    "warning_threshold": 0.10,
    "critical_threshold": 0.18
}
```

## Capture Settings

Camera and video input configuration.

```json
"capture": {
    "resolution": {
        "width": 1280,
        "height": 720
    },
    "fps": 30,
    "video_duration": 30,
    "auto_exposure": true,
    "exposure_time": 0.008,
    "gain": 1.0
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution.width` | int | 1280 | Frame width in pixels |
| `resolution.height` | int | 720 | Frame height in pixels |
| `fps` | int | 30 | Target frames per second |
| `video_duration` | int | 30 | Recording duration (seconds, 0 = unlimited) |
| `auto_exposure` | bool | true | Enable automatic exposure |
| `exposure_time` | float | 0.008 | Manual exposure time (seconds) |
| `gain` | float | 1.0 | Camera gain multiplier |

**Resolution Presets:**

| Preset | Width | Height | Use Case |
|--------|-------|--------|----------|
| Low | 640 | 480 | Low-end hardware, testing |
| Standard | 1280 | 720 | Default, balanced |
| High | 1920 | 1080 | High-quality, powerful hardware |

**FPS Guidelines:**
- **30 FPS**: Standard, good balance
- **60 FPS**: Highway driving, fast response needed
- **15-20 FPS**: Low-end hardware, reduce CPU load

## Output Options

Control saved outputs and visualizations.

```json
"output": {
    "save_warnings": true,
    "save_metrics": true,
    "generate_reports": true,
    "snapshot_quality": 95,
    "report_format": "utf-8",
    "visualization": {
        "show_roi": true,
        "show_lines": true,
        "show_metrics": true,
        "line_thickness": 2,
        "text_scale": 0.6
    }
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_warnings` | bool | true | Save frames when warnings occur |
| `save_metrics` | bool | true | Log performance metrics |
| `generate_reports` | bool | true | Create final report |
| `snapshot_quality` | int | 95 | JPEG quality (0-100) |
| `visualization.show_roi` | bool | true | Display ROI polygon |
| `visualization.show_lines` | bool | true | Display detected lanes |
| `visualization.show_metrics` | bool | true | Display metrics overlay |
| `visualization.line_thickness` | int | 2 | Lane line thickness |
| `visualization.text_scale` | float | 0.6 | Text size multiplier |

## Performance Tuning

Optimize system performance for your hardware.

```json
"performance": {
    "enable_gpu": true,
    "use_threading": true,
    "max_threads": 4,
    "buffer_size": 3,
    "batch_processing": false,
    "batch_size": 2
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_gpu` | bool | true | Use GPU acceleration if available |
| `use_threading` | bool | true | Enable multi-threaded processing |
| `max_threads` | int | 4 | Maximum worker threads |
| `buffer_size` | int | 3 | Frame buffer size |
| `batch_processing` | bool | false | Process frames in batches |
| `batch_size` | int | 2 | Frames per batch |

**Performance Profiles:**

```json
// Low-End Hardware
{
    "enable_gpu": false,
    "use_threading": false,
    "max_threads": 2,
    "buffer_size": 1
}

// Mid-Range Hardware (Default)
{
    "enable_gpu": true,
    "use_threading": true,
    "max_threads": 4,
    "buffer_size": 3
}

// High-End Hardware
{
    "enable_gpu": true,
    "use_threading": true,
    "max_threads": 8,
    "buffer_size": 5,
    "batch_processing": true
}
```

## Environment-Specific Configs

Pre-configured settings for different driving conditions.

### City Driving (`config_city.json`)

**Characteristics:**
- Frequent turns and intersections
- Lower speeds
- More visual noise
- Tighter roads

**Key Settings:**
- Lower thresholds for faster response
- Tighter ROI
- Shorter frame history
- More sensitive warnings

### Highway Driving (`config_highway.json`)

**Characteristics:**
- High speeds
- Long straight sections
- Wide lanes
- Distant detection needed

**Key Settings:**
- Higher resolution (1920x1080)
- Higher FPS (60)
- Wider ROI
- Longer frame history
- Less sensitive warnings

### Night Driving (`config_night.json`)

**Characteristics:**
- Low light conditions
- Headlight glare
- Reduced visibility
- Higher image noise

**Key Settings:**
- Lower Canny thresholds
- Adaptive edge detection
- Larger kernel size
- Enhanced smoothing
- Adjusted gain

## Best Practices

### 1. Camera Positioning

- **Height**: Mount 1-1.5m above ground
- **Angle**: Slightly tilted down (5-10 degrees)
- **Position**: Centered on windshield
- **View**: Clear view of road ahead (10-30m)

### 2. Calibration Process

```bash
# Run calibration tool
python utils/calibration_tool.py

# Test configuration
python advanced_ldws.py --config custom_config.json --duration 60

# Iterate and refine
```

### 3. Environment Selection

| Condition | Recommended Config |
|-----------|-------------------|
| City streets | `config_city.json` |
| Highways/freeways | `config_highway.json` |
| Night/dark | `config_night.json` |
| Mixed/unknown | `config_default.json` |

### 4. Performance Optimization

1. Start with default settings
2. Test and measure FPS
3. Adjust resolution if needed
4. Tune processing parameters
5. Enable GPU if available
6. Monitor CPU/memory usage

### 5. Testing and Validation

```bash
# Test with different configs
python advanced_ldws.py --config configs/config_city.json --duration 30
python advanced_ldws.py --config configs/config_highway.json --duration 30

# Compare results
cat output/reports/*.txt
```

## Configuration Examples

### Example 1: Aggressive Lane Keeping

```json
{
    "processing": {
        "canny_low": 35,
        "canny_high": 110,
        "frame_history": 20
    },
    "warning_system": {
        "warning_threshold": 0.10,
        "critical_threshold": 0.18,
        "warning_persistence": 0.8
    }
}
```

### Example 2: Low-Light Optimized

```json
{
    "processing": {
        "canny_low": 25,
        "canny_high": 85,
        "adaptive_canny": true,
        "kernel_size": 7
    },
    "capture": {
        "auto_exposure": false,
        "exposure_time": 0.020,
        "gain": 2.5
    }
}
```

### Example 3: Performance Mode

```json
{
    "capture": {
        "resolution": {
            "width": 640,
            "height": 480
        },
        "fps": 30
    },
    "processing": {
        "frame_history": 8,
        "hough_threshold": 40
    },
    "performance": {
        "enable_gpu": false,
        "use_threading": true,
        "max_threads": 2,
        "buffer_size": 1
    }
}
```

## Troubleshooting Configuration Issues

### Problem: Poor lane detection

**Solutions:**
1. Adjust ROI vertices to match your camera view
2. Tune Canny thresholds for lighting conditions
3. Increase `frame_history` for stability
4. Check camera positioning and focus

### Problem: Too many false positives

**Solutions:**
1. Increase `hough_threshold`
2. Increase `slope_threshold`
3. Reduce `warning_threshold`
4. Tighten ROI boundaries

### Problem: Slow performance

**Solutions:**
1. Reduce resolution
2. Decrease `frame_history`
3. Disable `save_warnings`
4. Enable GPU acceleration
5. Reduce buffer size

### Problem: Missed lane departures

**Solutions:**
1. Decrease `warning_threshold`
2. Increase `frame_history` (up to a point)
3. Widen ROI
4. Lower Canny thresholds

## Advanced Configuration

For advanced users who want to extend the system:

### Custom Configuration Loader

```python
from advanced_ldws import LaneDetectionConfig

# Load and modify config
config = LaneDetectionConfig('my_config.json')
config.config_data['processing']['canny_low'] = 35
config._flatten_config()
config.save_config()
```

### Runtime Configuration Changes

```python
# Modify during runtime (experimental)
detector.config.warning_threshold = 0.12
```

## See Also

- [INSTALLATION.md](INSTALLATION.md) - Installation instructions
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving guide

---

For questions or issues with configuration, please open an issue on GitHub.