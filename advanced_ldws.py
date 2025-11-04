import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from collections import deque
import pygame

class LaneDetectionConfig:
    """Configuration class for lane detection parameters"""
    def __init__(self, config_file="ldws_config.json"):
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or use defaults"""
        default_config = {
            "canny_low": 50,
            "canny_high": 150,
            "hough_threshold": 50,
            "hough_min_line_length": 50,
            "hough_max_line_gap": 150,
            "roi_top_factor": 0.6,
            "roi_bottom_left": 0.1,
            "roi_bottom_right": 0.9,
            "roi_top_left": 0.45,
            "roi_top_right": 0.55,
            "slope_threshold": 0.5,
            "warning_threshold": 0.15,
            "critical_threshold": 0.25,
            "frame_history": 10,
            "video_duration": 30,
            "resolution_width": 1280,
            "resolution_height": 720,
            "enable_audio": True,
            "save_warnings": True
        }
        
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        else:
            self.save_config(default_config)
        
        for key, value in default_config.items():
            setattr(self, key, value)
    
    def save_config(self, config=None):
        """Save current configuration to file"""
        if config is None:
            config = {k: v for k, v in self.__dict__.items() if k != 'config_file'}
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

class AudioAlert:
    """Handle audio alerts for lane departure warnings"""
    def __init__(self, enabled=True):
        self.enabled = enabled
        if enabled:
            try:
                pygame.mixer.init()
                self.warning_played = False
                self.critical_played = False
                self.last_alert_time = 0
                self.alert_cooldown = 2  # seconds
            except:
                print("Warning: Audio system not available")
                self.enabled = False
    
    def play_beep(self, frequency=1000, duration=200, volume=0.5):
        """Generate and play a beep sound"""
        if not self.enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        try:
            sample_rate = 22050
            samples = int(sample_rate * duration / 1000)
            wave = np.sin(2 * np.pi * frequency * np.linspace(0, duration/1000, samples))
            wave = (wave * volume * 32767).astype(np.int16)
            stereo_wave = np.column_stack((wave, wave))
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.play()
            self.last_alert_time = current_time
        except:
            pass
    
    def warning_alert(self):
        """Play warning level alert"""
        self.play_beep(frequency=800, duration=150, volume=0.3)
    
    def critical_alert(self):
        """Play critical level alert"""
        self.play_beep(frequency=1200, duration=300, volume=0.5)

class LaneMetrics:
    """Track and calculate lane detection metrics"""
    def __init__(self):
        self.total_frames = 0
        self.center_frames = 0
        self.warning_frames = 0
        self.critical_frames = 0
        self.lane_positions = []
        self.start_time = time.time()
    
    def update(self, status, lateral_offset=0):
        """Update metrics with current frame data"""
        self.total_frames += 1
        self.lane_positions.append(lateral_offset)
        
        if status == "CENTER":
            self.center_frames += 1
        elif status == "WARNING":
            self.warning_frames += 1
        elif status == "CRITICAL":
            self.critical_frames += 1
    
    def get_report(self):
        """Generate comprehensive metrics report"""
        if self.total_frames == 0:
            return "Insufficient data for analysis"
        
        elapsed_time = time.time() - self.start_time
        center_percentage = (self.center_frames / self.total_frames) * 100
        warning_percentage = (self.warning_frames / self.total_frames) * 100
        critical_percentage = (self.critical_frames / self.total_frames) * 100
        
        avg_offset = np.mean(self.lane_positions) if self.lane_positions else 0
        max_offset = np.max(np.abs(self.lane_positions)) if self.lane_positions else 0
        
        score = center_percentage - (warning_percentage * 0.5) - (critical_percentage * 2)
        score = max(0, min(100, score))
        
        report = f"""
{'='*60}
LANE DEPARTURE WARNING SYSTEM - FINAL REPORT
{'='*60}
Test Duration: {elapsed_time:.2f} seconds
Total Frames Analyzed: {self.total_frames}

LANE KEEPING PERFORMANCE:
  [OK] Center Lane:     {self.center_frames:4d} frames ({center_percentage:5.1f}%)
  [!!] Warning Level:   {self.warning_frames:4d} frames ({warning_percentage:5.1f}%)
  [XX] Critical Level:  {self.critical_frames:4d} frames ({critical_percentage:5.1f}%)

LATERAL POSITION ANALYSIS:
  Average Offset:    {avg_offset:+.3f} (negative = left, positive = right)
  Maximum Offset:    {max_offset:.3f}

OVERALL SCORE: {score:.1f}/100

RATING: """
        
        if score >= 90:
            report += "EXCELLENT [*****]"
        elif score >= 75:
            report += "GOOD [****]"
        elif score >= 60:
            report += "ACCEPTABLE [***]"
        elif score >= 40:
            report += "NEEDS IMPROVEMENT [**]"
        else:
            report += "POOR [*]"
        
        report += f"\n{'='*60}\n"
        return report

class AdvancedLaneDetector:
    """Advanced lane detection with temporal smoothing and robust tracking"""
    def __init__(self, config):
        self.config = config
        self.left_lane_history = deque(maxlen=config.frame_history)
        self.right_lane_history = deque(maxlen=config.frame_history)
        self.audio = AudioAlert(config.enable_audio)
        self.metrics = LaneMetrics()
        
        # Create output directory for warnings
        self.warning_dir = "lane_warnings"
        if config.save_warnings and not os.path.exists(self.warning_dir):
            os.makedirs(self.warning_dir)
    
    def region_of_interest(self, img):
        """Create region of interest mask"""
        height, width = img.shape
        mask = np.zeros_like(img)
        
        polygon = np.array([[
            (int(self.config.roi_bottom_left * width), height),
            (int(self.config.roi_bottom_right * width), height),
            (int(self.config.roi_top_right * width), int(self.config.roi_top_factor * height)),
            (int(self.config.roi_top_left * width), int(self.config.roi_top_factor * height)),
        ]], np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image, polygon
    
    def fit_lane_line(self, lines):
        """Fit a line through detected lane segments"""
        if not lines:
            return None
        
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])
        
        if len(points) < 2:
            return None
        
        points = np.array(points)
        # Fit polynomial (degree 1 for straight line, 2 for curves)
        try:
            z = np.polyfit(points[:, 1], points[:, 0], 1)
            return z
        except:
            return None
    
    def smooth_lane_lines(self, left_fit, right_fit):
        """Apply temporal smoothing to reduce jitter"""
        if left_fit is not None:
            self.left_lane_history.append(left_fit)
        if right_fit is not None:
            self.right_lane_history.append(right_fit)
        
        left_smooth = np.mean(self.left_lane_history, axis=0) if self.left_lane_history else None
        right_smooth = np.mean(self.right_lane_history, axis=0) if self.right_lane_history else None
        
        return left_smooth, right_smooth
    
    def calculate_lateral_offset(self, left_fit, right_fit, img_width, img_height):
        """Calculate vehicle's lateral offset from lane center"""
        if left_fit is None or right_fit is None:
            return None
        
        # Calculate x positions at bottom of image
        y_eval = img_height
        left_x = np.polyval(left_fit, y_eval)
        right_x = np.polyval(right_fit, y_eval)
        
        lane_center = (left_x + right_x) / 2
        vehicle_center = img_width / 2
        
        # Normalize offset (-1 to 1, where 0 is center)
        lane_width = right_x - left_x
        if lane_width > 0:
            offset = (vehicle_center - lane_center) / lane_width
            return offset
        return None
    
    def draw_advanced_lanes(self, img, left_fit, right_fit, status, offset):
        """Draw lanes with enhanced visualization"""
        height, width, _ = img.shape
        
        # Create overlay for lane area
        overlay = img.copy()
        lane_overlay = np.zeros_like(img)
        
        if left_fit is not None and right_fit is not None:
            # Generate points for lane lines
            y_points = np.linspace(int(height * self.config.roi_top_factor), height, 50)
            left_x = np.polyval(left_fit, y_points)
            right_x = np.polyval(right_fit, y_points)
            
            # Draw lane area
            left_points = np.array([np.column_stack((left_x, y_points))], dtype=np.int32)
            right_points = np.array([np.flipud(np.column_stack((right_x, y_points)))], dtype=np.int32)
            lane_points = np.hstack((left_points, right_points))
            
            if status == "CENTER":
                cv2.fillPoly(lane_overlay, lane_points, (0, 255, 0))
            elif status == "WARNING":
                cv2.fillPoly(lane_overlay, lane_points, (0, 255, 255))
            else:
                cv2.fillPoly(lane_overlay, lane_points, (0, 0, 255))
            
            cv2.addWeighted(overlay, 0.7, lane_overlay, 0.3, 0, overlay)
            
            # Draw lane lines
            for i in range(len(y_points) - 1):
                cv2.line(overlay, (int(left_x[i]), int(y_points[i])),
                        (int(left_x[i+1]), int(y_points[i+1])), (255, 0, 0), 3)
                cv2.line(overlay, (int(right_x[i]), int(y_points[i])),
                        (int(right_x[i+1]), int(y_points[i+1])), (255, 0, 0), 3)
        
        # Draw status information
        if status == "CENTER":
            color = (0, 255, 0)
            text = "✓ LANE KEEPING: GOOD"
        elif status == "WARNING":
            color = (0, 255, 255)
            text = "⚠ WARNING: Drifting"
        else:
            color = (0, 0, 255)
            text = "✗ CRITICAL: Lane Departure!"
        
        # Create status panel
        cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (500, 150), color, 2)
        
        cv2.putText(overlay, text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if offset is not None:
            offset_text = f"Lateral Offset: {offset:+.3f}"
            direction = "← LEFT" if offset < 0 else "RIGHT →"
            cv2.putText(overlay, offset_text, (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, direction, (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw center line
        cv2.line(overlay, (width//2, height-100), (width//2, height-20), (255, 255, 0), 2)
        
        return overlay
    
    def process_frame(self, frame):
        """Process a single frame with advanced lane detection using optimized methods"""
        height, width, _ = frame.shape
        
        # Preprocessing with optimized methods
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0, borderType=cv2.BORDER_REPLICATE)
        
        # Enhanced edge detection with dynamic thresholds
        median = np.median(blur)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edges = cv2.Canny(blur, lower, upper, apertureSize=3, L2gradient=True)
        
        # Apply ROI with optimized numpy operations
        cropped_edges, roi_polygon = self.region_of_interest(edges)
        
        # Detect lines with optimized parameters
        lines = cv2.HoughLinesP(cropped_edges, 
                               rho=1,
                               theta=np.pi/180,
                               threshold=self.config.hough_threshold,
                               maxLineGap=self.config.hough_max_line_gap,
                               minLineLength=self.config.hough_min_line_length)
        
        # Efficient line classification
        if lines is not None:
            # Vectorized operations for better performance
            slopes = np.zeros(len(lines))
            left_lines = []
            right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                if abs(x2 - x1) < 1:
                    continue
                
                slope = (y2 - y1) / (x2 - x1)
                if slope < -self.config.slope_threshold:
                    left_lines.append(line)
                elif slope > self.config.slope_threshold:
                    right_lines.append(line)
        
        # Fit lane lines
        left_fit = self.fit_lane_line(left_lines)
        right_fit = self.fit_lane_line(right_lines)
        
        # Apply temporal smoothing
        left_fit, right_fit = self.smooth_lane_lines(left_fit, right_fit)
        
        # Calculate lateral offset
        offset = self.calculate_lateral_offset(left_fit, right_fit, width, height)
        
        # Determine status
        if left_fit is None or right_fit is None:
            status = "NO_LANES"
            offset = None
        elif offset is None:
            status = "UNCERTAIN"
        elif abs(offset) < self.config.warning_threshold:
            status = "CENTER"
        elif abs(offset) < self.config.critical_threshold:
            status = "WARNING"
            self.audio.warning_alert()
        else:
            status = "CRITICAL"
            self.audio.critical_alert()
        
        # Draw visualization
        output = self.draw_advanced_lanes(frame, left_fit, right_fit, status, offset)
        
        # Update metrics
        self.metrics.update(status, offset if offset else 0)
        
        # Draw ROI polygon for reference
        cv2.polylines(output, [roi_polygon], True, (255, 0, 255), 2)
        
        return output, status, offset

def main():
    """Main function to run the advanced lane departure warning system"""
    try:
        # Ensure UTF-8 encoding for output
        import sys
        import codecs
        import io
        if sys.version_info[0] < 3:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
        else:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
        # Load configuration with error handling
        try:
            config = LaneDetectionConfig()
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration...")
            config = LaneDetectionConfig()
            config.load_config()
        
        # Initialize detector with performance monitoring
        detector = AdvancedLaneDetector(config)
        
        # Open video source with enhanced error handling and buffer optimization
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        except Exception:
            # Fallback to default backend if DirectShow fails
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Webcam not detected. Please check your camera connection.")
            print("Tip: You can also use a video file by changing VideoCapture(0) to VideoCapture('video.mp4')")
            return
            
        # Optimize video capture settings for performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set target FPS
    except Exception as e:
        print(f"Critical error during initialization: {e}")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution_height)
    
    # Create window
    window_name = "Advanced Lane Departure Warning System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print("="*60)
    print("ADVANCED LANE DEPARTURE WARNING SYSTEM")
    print("="*60)
    print(f"System started successfully!")
    print(f"Duration: {config.video_duration} seconds")
    print(f"Controls: Press 'q' to quit, 's' to save current frame")
    print("="*60)
    
    start_time = time.time()
    frame_count = 0
    fps_display = 0
    fps_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        frame = cv2.resize(frame, (config.resolution_width, config.resolution_height))
        
        # Process frame
        output, status, offset = detector.process_frame(frame)
        
        # Calculate and display FPS
        frame_count += 1
        if time.time() - fps_time > 1:
            fps_display = frame_count / (time.time() - fps_time)
            frame_count = 0
            fps_time = time.time()
        
        cv2.putText(output, f"FPS: {fps_display:.1f}", (config.resolution_width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save warning frames
        if config.save_warnings and status in ["WARNING", "CRITICAL"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(detector.warning_dir, f"warning_{status}_{timestamp}.jpg")
            cv2.imwrite(filename, output)
        
        cv2.imshow(window_name, output)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or (time.time() - start_time > config.video_duration):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, output)
            print(f"Snapshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final report
    print(detector.metrics.get_report())
    
    try:
        # Save report to file with UTF-8 encoding
        report_filename = f"lane_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(detector.metrics.get_report())
        print(f"Detailed report saved: {report_filename}")
    except Exception as e:
        print(f"Warning: Could not save report file: {e}")
        # Fallback to console output
        print("\nReport:\n")
        print(detector.metrics.get_report())

if __name__ == "__main__":
    main()