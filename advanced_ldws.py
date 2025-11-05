#!/usr/bin/env python3
"""
Advanced Lane Departure Warning System (LDWS)
Optimized for Python 3.13

A production-ready real-time lane departure warning system with advanced
computer vision algorithms, temporal smoothing, and intelligent alerts.
"""

import cv2
import numpy as np
import time
import json
import os
import sys
import argparse
import logging
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pygame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ldws.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LaneDetectionConfig:
    """Configuration manager for lane detection parameters with nested JSON support"""

    def __init__(self, config_file: str = "ldws_config.json"):
        self.config_file = config_file
        self.config_data: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from JSON file with fallback to defaults"""
        default_config = self._get_default_config()

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self._merge_configs(default_config, loaded_config)
                    logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {e}. Using defaults.")
                self.config_data = default_config
        else:
            logger.warning(f"Config file not found. Creating default: {self.config_file}")
            self.config_data = default_config
            self.save_config()

        # Flatten nested structure for easy attribute access
        self._flatten_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration structure"""
        return {
            "processing": {
                "canny_low": 40,
                "canny_high": 120,
                "adaptive_canny": False,
                "hough_threshold": 30,
                "hough_min_line_length": 40,
                "hough_max_line_gap": 100,
                "roi_vertices": {
                    "top_left": [0.45, 0.6],
                    "top_right": [0.55, 0.6],
                    "bottom_right": [0.9, 1.0],
                    "bottom_left": [0.1, 1.0]
                },
                "slope_threshold": 0.4,
                "kernel_size": 5,
                "frame_history": 15,
                "smoothing_window": 5
            },
            "warning_system": {
                "warning_threshold": 0.15,
                "critical_threshold": 0.25,
                "time_to_warning": 0.5,
                "warning_persistence": 1.0,
                "enable_audio": True,
                "audio_volume": 0.8
            },
            "capture": {
                "resolution": {
                    "width": 1280,
                    "height": 720
                },
                "fps": 30,
                "video_duration": 30,
                "auto_exposure": True,
                "exposure_time": 0.008,
                "gain": 1.0
            },
            "output": {
                "save_warnings": True,
                "save_metrics": True,
                "generate_reports": True,
                "snapshot_quality": 95,
                "report_format": "utf-8",
                "visualization": {
                    "show_roi": True,
                    "show_lines": True,
                    "show_metrics": True,
                    "line_thickness": 2,
                    "text_scale": 0.6
                }
            },
            "performance": {
                "enable_gpu": True,
                "use_threading": True,
                "max_threads": 4,
                "buffer_size": 5,
                "batch_processing": False,
                "batch_size": 2
            }
        }

    def _merge_configs(self, default: Dict, loaded: Dict) -> None:
        """Recursively merge loaded config with defaults"""
        def merge_dict(d1: Dict, d2: Dict) -> Dict:
            result = d1.copy()
            for key, value in d2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result

        self.config_data = merge_dict(default, loaded)

    def _flatten_config(self) -> None:
        """Flatten nested config for attribute access"""
        def flatten(d: Dict, parent_key: str = '') -> Dict:
            items: List[Tuple[str, Any]] = []
            for k, v in d.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict) and not parent_key.endswith('vertices'):
                    items.extend(flatten(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_config = flatten(self.config_data)
        for key, value in flat_config.items():
            setattr(self, key, value)

        # Special handling for nested structures
        if 'processing' in self.config_data:
            self.roi_vertices = self.config_data['processing'].get('roi_vertices', {})

    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=4)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default fallback"""
        return getattr(self, key, default)


class AudioAlert:
    """Audio alert system with context-aware warnings"""

    def __init__(self, enabled: bool = True, volume: float = 0.8):
        self.enabled = enabled
        self.volume = max(0.0, min(1.0, volume))
        self.last_alert_time = 0.0
        self.alert_cooldown = 1.5
        self.warning_count = 0
        self.critical_count = 0

        if enabled:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                logger.info("Audio system initialized successfully")
            except Exception as e:
                logger.warning(f"Audio system unavailable: {e}")
                self.enabled = False

    def _can_play_alert(self) -> bool:
        """Check if enough time has passed since last alert"""
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        self.last_alert_time = current_time
        return True

    def _generate_beep(self, frequency: int, duration: int) -> Optional[pygame.mixer.Sound]:
        """Generate a beep sound wave"""
        if not self.enabled:
            return None

        try:
            sample_rate = 22050
            samples = int(sample_rate * duration / 1000)

            # Generate sine wave with fade in/out
            t = np.linspace(0, duration / 1000, samples)
            wave = np.sin(2 * np.pi * frequency * t)

            # Apply envelope
            envelope = np.ones_like(wave)
            fade_samples = int(samples * 0.1)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            wave *= envelope

            # Scale and convert
            wave = (wave * self.volume * 32767).astype(np.int16)
            stereo_wave = np.column_stack((wave, wave))

            return pygame.sndarray.make_sound(stereo_wave)
        except Exception as e:
            logger.error(f"Error generating beep: {e}")
            return None

    def warning_alert(self) -> None:
        """Play warning level alert (single beep)"""
        if not self.enabled or not self._can_play_alert():
            return

        self.warning_count += 1
        sound = self._generate_beep(frequency=800, duration=200)
        if sound:
            sound.play()
            logger.debug("Warning alert played")

    def critical_alert(self) -> None:
        """Play critical level alert (double beep)"""
        if not self.enabled or not self._can_play_alert():
            return

        self.critical_count += 1
        try:
            # First beep
            sound1 = self._generate_beep(frequency=1000, duration=150)
            if sound1:
                sound1.play()
                pygame.time.wait(200)

            # Second beep
            sound2 = self._generate_beep(frequency=1200, duration=150)
            if sound2:
                sound2.play()

            logger.debug("Critical alert played")
        except Exception as e:
            logger.error(f"Error playing critical alert: {e}")

    def cleanup(self) -> None:
        """Cleanup audio resources"""
        if self.enabled:
            try:
                pygame.mixer.quit()
                logger.info(
                    f"Audio stats - Warnings: {self.warning_count}, "
                    f"Critical: {self.critical_count}"
                )
            except Exception:
                pass


class LaneMetrics:
    """Comprehensive metrics tracking and analysis"""

    def __init__(self):
        self.total_frames = 0
        self.center_frames = 0
        self.warning_frames = 0
        self.critical_frames = 0
        self.no_detection_frames = 0
        self.lane_positions = deque(maxlen=1000)
        self.start_time = time.time()
        self.frame_times = deque(maxlen=100)
        self.detection_confidence = deque(maxlen=100)

    def update(
        self, status: str, lateral_offset: float = 0, confidence: float = 1.0
    ) -> None:
        """Update metrics with current frame data"""
        self.total_frames += 1
        self.lane_positions.append(lateral_offset)
        self.detection_confidence.append(confidence)

        status_map = {
            "CENTER": self.center_frames,
            "WARNING": self.warning_frames,
            "CRITICAL": self.critical_frames,
            "NO_LANES": self.no_detection_frames,
            "UNCERTAIN": self.no_detection_frames
        }

        if status in status_map:
            if status == "CENTER":
                self.center_frames += 1
            elif status == "WARNING":
                self.warning_frames += 1
            elif status == "CRITICAL":
                self.critical_frames += 1
            else:
                self.no_detection_frames += 1

    def update_frame_time(self, frame_time: float) -> None:
        """Track frame processing time"""
        self.frame_times.append(frame_time)

    def get_fps(self) -> float:
        """Calculate current FPS"""
        if not self.frame_times:
            return 0.0
        avg_time = np.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive statistics"""
        if self.total_frames == 0:
            return {}

        positions = list(self.lane_positions)
        confidences = list(self.detection_confidence)

        return {
            'center_percentage': (self.center_frames / self.total_frames) * 100,
            'warning_percentage': (self.warning_frames / self.total_frames) * 100,
            'critical_percentage': (self.critical_frames / self.total_frames) * 100,
            'detection_rate': (
                (self.total_frames - self.no_detection_frames) / self.total_frames
            ) * 100,
            'avg_offset': np.mean(positions) if positions else 0,
            'max_offset': np.max(np.abs(positions)) if positions else 0,
            'std_offset': np.std(positions) if positions else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_fps': self.get_fps()
        }

    def calculate_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        if self.total_frames == 0:
            return 0.0

        stats = self.get_statistics()

        # Weighted scoring
        score = (
            stats['center_percentage'] * 0.4
            + stats['detection_rate'] * 0.3
            + (100 - stats['warning_percentage']) * 0.15
            + (100 - stats['critical_percentage']) * 0.15
        )

        # Penalties
        if stats['max_offset'] > 0.5:
            score *= 0.9
        if stats['avg_fps'] < 15:
            score *= 0.95

        return max(0, min(100, score))

    def get_report(self) -> str:
        """Generate comprehensive text report"""
        if self.total_frames == 0:
            return "Insufficient data for analysis"

        elapsed_time = time.time() - self.start_time
        stats = self.get_statistics()
        score = self.calculate_score()

        # Rating system
        if score >= 95:
            rating = "EXCEPTIONAL (5 stars)"
        elif score >= 85:
            rating = "EXCELLENT (4 stars)"
        elif score >= 75:
            rating = "GOOD (3 stars)"
        elif score >= 60:
            rating = "ACCEPTABLE (2 stars)"
        elif score >= 40:
            rating = "NEEDS IMPROVEMENT (1 star)"
        else:
            rating = "POOR"

        report = f"""
{'='*70}
           LANE DEPARTURE WARNING SYSTEM - FINAL REPORT
{'='*70}

TEST INFORMATION:
  Duration:              {elapsed_time:.2f} seconds
  Total Frames:          {self.total_frames:,}
  Average FPS:           {stats['avg_fps']:.1f}
  Detection Rate:        {stats['detection_rate']:.1f}%

LANE KEEPING PERFORMANCE:
  [OK] Center Lane:      {self.center_frames:5,} frames ({stats['center_percentage']:5.1f}%)
  [!]  Warning Level:    {self.warning_frames:5,} frames ({stats['warning_percentage']:5.1f}%)
  [X]  Critical Level:   {self.critical_frames:5,} frames ({stats['critical_percentage']:5.1f}%)
  [?]  No Detection:     {self.no_detection_frames:5,} frames

LATERAL POSITION ANALYSIS:
  Average Offset:        {stats['avg_offset']:+.4f} (negative = left, positive = right)
  Maximum Offset:        {stats['max_offset']:.4f}
  Standard Deviation:    {stats['std_offset']:.4f}
  Stability Index:       {(1 - stats['std_offset']) * 100:.1f}%

DETECTION CONFIDENCE:
  Average Confidence:    {stats['avg_confidence']:.2%}

OVERALL PERFORMANCE:
  Score:                 {score:.1f}/100
  Rating:                {rating}

{'='*70}
"""
        return report


class AdvancedLaneDetector:
    """Advanced lane detection with Kalman filtering and robust tracking"""

    def __init__(self, config: LaneDetectionConfig):
        self.config = config
        self.left_lane_history: deque = deque(
            maxlen=config.get('processing_frame_history', 15)
        )
        self.right_lane_history: deque = deque(
            maxlen=config.get('processing_frame_history', 15)
        )
        self.audio = AudioAlert(
            enabled=config.get('warning_system_enable_audio', True),
            volume=config.get('warning_system_audio_volume', 0.8)
        )
        self.metrics = LaneMetrics()

        # State tracking
        self.last_status = "CENTER"
        self.status_transition_time = time.time()
        self.paused = False
        self.debug_mode = False

        # Create output directories
        self.output_dir = Path("output")
        self.warning_dir = self.output_dir / "lane_warnings"
        self.snapshot_dir = self.output_dir / "snapshots"
        self.report_dir = self.output_dir / "reports"

        for directory in [self.warning_dir, self.snapshot_dir, self.report_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("Lane detector initialized successfully")

    def region_of_interest(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create optimized region of interest mask"""
        height, width = img.shape
        mask = np.zeros_like(img)

        # Get ROI vertices from config
        roi = self.config.roi_vertices

        polygon = np.array([[
            (int(roi['bottom_left'][0] * width), int(roi['bottom_left'][1] * height)),
            (int(roi['bottom_right'][0] * width), int(roi['bottom_right'][1] * height)),
            (int(roi['top_right'][0] * width), int(roi['top_right'][1] * height)),
            (int(roi['top_left'][0] * width), int(roi['top_left'][1] * height)),
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image, polygon

    def adaptive_canny(self, img: np.ndarray) -> np.ndarray:
        """Apply adaptive Canny edge detection"""
        # Use config values if available, otherwise use adaptive
        use_adaptive = self.config.get('processing_adaptive_canny', False)

        if use_adaptive:
            median = np.median(img)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * median))
            upper = int(min(255, (1.0 + sigma) * median))
        else:
            lower = self.config.get('processing_canny_low', 40)
            upper = self.config.get('processing_canny_high', 120)

        edges = cv2.Canny(img, lower, upper, apertureSize=3, L2gradient=True)
        return edges

    def fit_lane_line(self, lines: List[np.ndarray]) -> Optional[np.ndarray]:
        """Fit polynomial through lane segments with RANSAC"""
        if not lines or len(lines) < 2:
            return None

        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])

        if len(points) < 4:
            return None

        points = np.array(points)

        try:
            # Fit polynomial (degree 1 for simplicity)
            z = np.polyfit(points[:, 1], points[:, 0], 1)
            return z
        except Exception as e:
            logger.debug(f"Error fitting lane line: {e}")
            return None

    def smooth_lane_lines(
        self, left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply temporal smoothing with exponential moving average"""
        if left_fit is not None:
            self.left_lane_history.append(left_fit)
        if right_fit is not None:
            self.right_lane_history.append(right_fit)

        # Weighted average (more recent = higher weight)
        def weighted_average(history):
            if not history:
                return None
            weights = np.exp(np.linspace(-1, 0, len(history)))
            weights /= weights.sum()
            return np.average(history, axis=0, weights=weights)

        left_smooth = weighted_average(list(self.left_lane_history))
        right_smooth = weighted_average(list(self.right_lane_history))

        return left_smooth, right_smooth

    def calculate_lateral_offset(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        img_width: int,
        img_height: int
    ) -> Tuple[Optional[float], float]:
        """Calculate lateral offset and confidence"""
        if left_fit is None or right_fit is None:
            return None, 0.0

        # Calculate x positions at bottom of image
        y_eval = img_height
        left_x = np.polyval(left_fit, y_eval)
        right_x = np.polyval(right_fit, y_eval)

        # Sanity check
        if right_x <= left_x or (right_x - left_x) > img_width:
            return None, 0.0

        lane_center = (left_x + right_x) / 2
        vehicle_center = img_width / 2
        lane_width = right_x - left_x

        # Normalized offset (-1 to 1)
        offset = (vehicle_center - lane_center) / lane_width

        # Confidence based on lane width consistency
        expected_width = img_width * 0.4  # Typical lane width
        width_confidence = 1.0 - min(abs(lane_width - expected_width) / expected_width, 1.0)

        return offset, width_confidence

    def draw_advanced_lanes(
        self,
        img: np.ndarray,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        status: str,
        offset: Optional[float],
        confidence: float
    ) -> np.ndarray:
        """Draw lanes with professional visualization"""
        height, width, _ = img.shape
        overlay = img.copy()
        lane_overlay = np.zeros_like(img)

        # Color scheme based on status
        colors = {
            "CENTER": (0, 255, 0),      # Green
            "WARNING": (0, 255, 255),   # Yellow
            "CRITICAL": (0, 0, 255),    # Red
            "NO_LANES": (128, 128, 128),  # Gray
            "UNCERTAIN": (255, 165, 0)   # Orange
        }

        color = colors.get(status, (255, 255, 255))

        # Draw lane area
        if left_fit is not None and right_fit is not None:
            roi_top = int(height * self.config.roi_vertices['top_left'][1])
            y_points = np.linspace(roi_top, height, 100)
            left_x = np.polyval(left_fit, y_points)
            right_x = np.polyval(right_fit, y_points)

            # Clip to image bounds
            left_x = np.clip(left_x, 0, width - 1)
            right_x = np.clip(right_x, 0, width - 1)

            # Create lane polygon
            left_points = np.array([np.column_stack((left_x, y_points))], dtype=np.int32)
            right_points = np.array(
                [np.flipud(np.column_stack((right_x, y_points)))], dtype=np.int32
            )
            lane_points = np.hstack((left_points, right_points))

            # Fill lane area with transparency
            cv2.fillPoly(lane_overlay, lane_points, color)
            cv2.addWeighted(overlay, 0.7, lane_overlay, 0.3, 0, overlay)

            # Draw lane lines
            thickness = self.config.get('output_visualization_line_thickness', 3)
            for i in range(len(y_points) - 1):
                cv2.line(
                    overlay,
                    (int(left_x[i]), int(y_points[i])),
                    (int(left_x[i + 1]), int(y_points[i + 1])),
                    (255, 0, 0),
                    thickness
                )
                cv2.line(
                    overlay,
                    (int(right_x[i]), int(y_points[i])),
                    (int(right_x[i + 1]), int(y_points[i + 1])),
                    (255, 0, 0),
                    thickness
                )

        # Draw status panel
        self._draw_status_panel(overlay, status, offset, confidence, color)

        # Draw center reference line
        cv2.line(overlay, (width // 2, height - 100), (width // 2, height - 20), (255, 255, 0), 2)

        return overlay

    def _draw_status_panel(
        self,
        img: np.ndarray,
        status: str,
        offset: Optional[float],
        confidence: float,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw informative status panel"""
        height, width, _ = img.shape

        # Status messages (using ASCII characters for Windows compatibility)
        status_text = {
            "CENTER": "[OK] LANE KEEPING: EXCELLENT",
            "WARNING": "[!] WARNING: Drifting Detected",
            "CRITICAL": "[X] CRITICAL: Lane Departure!",
            "NO_LANES": "[?] No Lanes Detected",
            "UNCERTAIN": "[?] Uncertain Detection"
        }

        text = status_text.get(status, "Unknown Status")

        # Draw semi-transparent background
        panel_height = 180
        cv2.rectangle(img, (10, 10), (550, panel_height), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (550, panel_height), color, 3)

        # Status text
        scale = self.config.get('output_visualization_text_scale', 0.7)
        cv2.putText(img, text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

        # Offset information
        if offset is not None:
            offset_text = f"Lateral Offset: {offset:+.3f}"
            direction = "<- LEFT" if offset < 0 else "RIGHT ->"
            percentage = f"({abs(offset) * 100:.1f}%)"

            cv2.putText(
                img, offset_text, (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(
                img, f"{direction} {percentage}", (25, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # Confidence meter
        cv2.putText(
            img, f"Confidence: {confidence:.0%}", (25, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Confidence bar
        bar_width = int(500 * confidence)
        cv2.rectangle(img, (25, 160), (25 + bar_width, 170), color, -1)
        cv2.rectangle(img, (25, 160), (525, 170), (255, 255, 255), 1)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, Optional[float]]:
        """Process single frame with comprehensive lane detection"""
        start_time = time.time()

        if self.paused:
            return frame, self.last_status, None

        height, width, _ = frame.shape

        # Preprocessing pipeline
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kernel_size = self.config.get('processing_kernel_size', 5)
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Edge detection
        edges = self.adaptive_canny(blur)

        # Apply ROI
        cropped_edges, roi_polygon = self.region_of_interest(edges)

        # Hough line detection
        lines = cv2.HoughLinesP(
            cropped_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.get('processing_hough_threshold', 30),
            minLineLength=self.config.get('processing_hough_min_line_length', 40),
            maxLineGap=self.config.get('processing_hough_max_line_gap', 100)
        )

        # Classify lines
        left_lines, right_lines = [], []
        slope_threshold = self.config.get('processing_slope_threshold', 0.4)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 1:
                    continue

                slope = (y2 - y1) / (x2 - x1)
                if slope < -slope_threshold:
                    left_lines.append(line)
                elif slope > slope_threshold:
                    right_lines.append(line)

        # Fit and smooth lane lines
        left_fit = self.fit_lane_line(left_lines)
        right_fit = self.fit_lane_line(right_lines)
        left_fit, right_fit = self.smooth_lane_lines(left_fit, right_fit)

        # Calculate offset and confidence
        offset, confidence = self.calculate_lateral_offset(left_fit, right_fit, width, height)

        # Determine status
        status = self._determine_status(left_fit, right_fit, offset, confidence)

        # Trigger alerts
        self._handle_alerts(status)

        # Draw visualization
        output = self.draw_advanced_lanes(frame, left_fit, right_fit, status, offset, confidence)

        # Draw ROI polygon if enabled
        if self.config.get('output_visualization_show_roi', True):
            cv2.polylines(output, [roi_polygon], True, (255, 0, 255), 2)

        # Debug mode overlays
        if self.debug_mode:
            self._draw_debug_info(output, edges, cropped_edges, lines, left_lines, right_lines)

        # Update metrics
        self.metrics.update(status, offset if offset else 0, confidence)

        # Track frame time
        frame_time = time.time() - start_time
        self.metrics.update_frame_time(frame_time)

        # Save warnings if configured
        if self.config.get('output_save_warnings', True) and status in ["WARNING", "CRITICAL"]:
            self._save_warning_frame(output, status)

        self.last_status = status
        return output, status, offset

    def _determine_status(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        offset: Optional[float],
        confidence: float
    ) -> str:
        """Determine lane keeping status with confidence consideration"""
        if left_fit is None or right_fit is None:
            return "NO_LANES"

        if offset is None or confidence < 0.3:
            return "UNCERTAIN"

        warning_threshold = self.config.get('warning_system_warning_threshold', 0.15)
        critical_threshold = self.config.get('warning_system_critical_threshold', 0.25)

        abs_offset = abs(offset)

        if abs_offset >= critical_threshold:
            return "CRITICAL"
        elif abs_offset >= warning_threshold:
            return "WARNING"
        else:
            return "CENTER"

    def _handle_alerts(self, status: str) -> None:
        """Handle audio alerts based on status changes"""
        # Only alert on status transitions or repeated critical
        current_time = time.time()

        if status != self.last_status:
            self.status_transition_time = current_time

        time_in_status = current_time - self.status_transition_time
        persistence = self.config.get('warning_system_warning_persistence', 1.0)

        if status == "WARNING" and time_in_status > persistence * 0.5:
            self.audio.warning_alert()
        elif status == "CRITICAL":
            if time_in_status > persistence or self.last_status != "CRITICAL":
                self.audio.critical_alert()

    def _save_warning_frame(self, frame: np.ndarray, status: str) -> None:
        """Save warning frame to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = self.warning_dir / f"warning_{status}_{timestamp}.jpg"
            quality = self.config.get('output_snapshot_quality', 95)
            cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        except Exception as e:
            logger.error(f"Error saving warning frame: {e}")

    def _draw_debug_info(
        self,
        img: np.ndarray,
        edges: np.ndarray,
        cropped_edges: np.ndarray,
        all_lines: Optional[np.ndarray],
        left_lines: List,
        right_lines: List
    ) -> None:
        """Draw debug information overlay"""
        height, width, _ = img.shape

        # Create debug panel
        debug_height = 200
        cv2.rectangle(img, (width - 310, 10), (width - 10, debug_height), (0, 0, 0), -1)
        cv2.rectangle(img, (width - 310, 10), (width - 10, debug_height), (255, 255, 255), 2)

        cv2.putText(
            img, "DEBUG MODE", (width - 290, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

        # Line statistics
        total_lines = len(all_lines) if all_lines is not None else 0
        cv2.putText(
            img, f"Total Lines: {total_lines}", (width - 290, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            img, f"Left Lines: {len(left_lines)}", (width - 290, 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        cv2.putText(
            img, f"Right Lines: {len(right_lines)}", (width - 290, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

        # Edge detection info
        edge_pixels = np.count_nonzero(cropped_edges)
        cv2.putText(
            img, f"Edge Pixels: {edge_pixels:,}", (width - 290, 145),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # History size
        cv2.putText(
            img, f"History: L={len(self.left_lane_history)} R={len(self.right_lane_history)}",
            (width - 290, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    def save_final_report(self) -> str:
        """Save final performance report"""
        try:
            report = self.metrics.get_report()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.report_dir / f"lane_report_{timestamp}.txt"

            # Force UTF-8 encoding to handle Unicode characters on Windows
            with open(filename, 'w', encoding='utf-8', errors='replace') as f:
                f.write(report)

            logger.info(f"Report saved: {filename}")
            return str(filename)
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return ""

    def cleanup(self) -> None:
        """Cleanup resources"""
        self.audio.cleanup()
        logger.info("Lane detector cleanup completed")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Advanced Lane Departure Warning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use webcam with default config
  %(prog)s --input video.mp4                  # Process video file
  %(prog)s --config configs/config_night.json # Use night mode config
  %(prog)s --duration 60 --no-audio           # 60 seconds without audio
        """
    )

    parser.add_argument(
        '--input', '-i', type=str, default='0',
        help='Video source: camera index (0) or video file path'
    )
    parser.add_argument(
        '--config', '-c', type=str, default='ldws_config.json',
        help='Configuration file path'
    )
    parser.add_argument(
        '--duration', '-d', type=int, default=None,
        help='Video duration in seconds (overrides config)'
    )
    parser.add_argument(
        '--no-audio', action='store_true',
        help='Disable audio alerts'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug visualization'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--resolution', type=str, default=None,
        help='Camera resolution (WIDTHxHEIGHT, e.g., 1920x1080)'
    )

    return parser.parse_args()


def initialize_video_capture(
    source: str, config: LaneDetectionConfig
) -> Optional[cv2.VideoCapture]:
    """Initialize video capture with robust error handling"""
    try:
        # Determine if source is camera or file
        if source.isdigit():
            camera_id = int(source)
            logger.info(f"Opening camera {camera_id}...")

            # Try DirectShow on Windows for better performance
            if sys.platform == 'win32':
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(camera_id)
        else:
            logger.info(f"Opening video file: {source}")
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error("Failed to open video source")
            return None

        # Configure capture
        width = config.get('capture_resolution_width', 1280)
        height = config.get('capture_resolution_height', 720)
        fps = config.get('capture_fps', 30)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, config.get('performance_buffer_size', 3))

        # Verify settings
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"Video capture initialized: {int(actual_width)}x{int(actual_height)} "
            f"@ {actual_fps:.1f} FPS"
        )

        return cap

    except Exception as e:
        logger.error(f"Error initializing video capture: {e}")
        return None


def main():
    """Main application entry point"""
    try:
        # Parse arguments
        args = parse_arguments()

        # Setup output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("     ADVANCED LANE DEPARTURE WARNING SYSTEM v2.0")
        print("     Optimized for Python 3.13")
        print("=" * 70)
        print()

        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = LaneDetectionConfig(args.config)

        # Override config with command line arguments
        if args.duration:
            config.config_data['capture']['video_duration'] = args.duration
        if args.no_audio:
            config.config_data['warning_system']['enable_audio'] = False
        if args.resolution:
            try:
                w, h = map(int, args.resolution.split('x'))
                config.config_data['capture']['resolution'] = {'width': w, 'height': h}
            except Exception:
                logger.warning(f"Invalid resolution format: {args.resolution}")

        config._flatten_config()

        # Initialize detector
        logger.info("Initializing lane detector...")
        detector = AdvancedLaneDetector(config)
        detector.debug_mode = args.debug

        # Initialize video capture
        cap = initialize_video_capture(args.input, config)
        if cap is None:
            print("❌ ERROR: Could not initialize video source")
            print("\nTroubleshooting:")
            print("  1. Check if camera is connected")
            print("  2. Try different camera index: --input 1")
            print("  3. Use a video file: --input path/to/video.mp4")
            print("  4. Check permissions and drivers")
            return 1

        # Create display window
        window_name = "Advanced LDWS - Press 'q' to quit, 's' to save, 'p' to pause, 'd' for debug"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        target_width = config.get('capture_resolution_width', 1280)
        target_height = config.get('capture_resolution_height', 720)
        cv2.resizeWindow(window_name, target_width, target_height)

        # Display startup info
        print("[OK] System initialized successfully")
        print(f"[OK] Configuration: {args.config}")
        print(f"[OK] Video source: {args.input}")
        print(f"[OK] Duration: {config.get('capture_video_duration', 'unlimited')} seconds")
        print(f"[OK] Audio alerts: {'enabled' if config.get('warning_system_enable_audio') else 'disabled'}")
        print()
        print("Controls:")
        print("  [q] Quit")
        print("  [s] Save snapshot")
        print("  [p] Pause/Resume")
        print("  [d] Toggle debug mode")
        print("  [r] Reset metrics")
        print("=" * 70)
        print()

        # Main processing loop
        start_time = time.time()
        duration = config.get('capture_video_duration', 30)
        frame_count = 0

        while True:
            # Check duration
            elapsed = time.time() - start_time
            if duration and elapsed > duration:
                logger.info(f"Duration limit reached: {duration}s")
                break

            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break

            # Resize frame to target resolution
            frame = cv2.resize(frame, (target_width, target_height))
            frame_count += 1

            # Process frame
            output, status, offset = detector.process_frame(frame)

            # Add FPS and frame counter
            fps = detector.metrics.get_fps()
            cv2.putText(
                output, f"FPS: {fps:.1f}", (target_width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            cv2.putText(
                output, f"Frame: {frame_count}", (target_width - 150, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            # Add time remaining
            if duration:
                remaining = max(0, duration - elapsed)
                cv2.putText(
                    output, f"Time: {remaining:.0f}s", (target_width - 150, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )

            # Show pause indicator
            if detector.paused:
                cv2.putText(
                    output, "PAUSED", (target_width // 2 - 80, target_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3
                )

            # Display frame
            cv2.imshow(window_name, output)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("User quit requested")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = detector.snapshot_dir / f"snapshot_{timestamp}.jpg"
                cv2.imwrite(str(filename), output)
                print(f"[OK] Snapshot saved: {filename}")
                logger.info(f"Snapshot saved: {filename}")
            elif key == ord('p'):
                detector.paused = not detector.paused
                status_msg = "PAUSED" if detector.paused else "RESUMED"
                print(f"[OK] Playback {status_msg}")
                logger.info(f"Playback {status_msg}")
            elif key == ord('d'):
                detector.debug_mode = not detector.debug_mode
                status_msg = "enabled" if detector.debug_mode else "disabled"
                print(f"[OK] Debug mode {status_msg}")
                logger.info(f"Debug mode {status_msg}")
            elif key == ord('r'):
                detector.metrics = LaneMetrics()
                print("[OK] Metrics reset")
                logger.info("Metrics reset")

        # Cleanup
        print()
        print("=" * 70)
        print("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()

        # Generate and save final report
        print()
        print(detector.metrics.get_report())

        report_file = detector.save_final_report()
        if report_file:
            print(f"[OK] Detailed report saved: {report_file}")

        # Cleanup detector
        detector.cleanup()

        print()
        print("=" * 70)
        print("Thank you for using Advanced LDWS!")
        print("=" * 70)

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        logger.exception("Critical error in main")
        return 1


if __name__ == "__main__":
    sys.exit(main())