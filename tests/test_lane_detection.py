"""
Unit tests for AdvancedLaneDetector class
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_ldws import LaneDetectionConfig, AdvancedLaneDetector, LaneMetrics, AudioAlert


class TestAdvancedLaneDetector:
    """Test cases for lane detection"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return LaneDetectionConfig()
    
    @pytest.fixture
    def detector(self, config):
        """Create detector instance"""
        with patch('advanced_ldws.AudioAlert'):
            return AdvancedLaneDetector(config)
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample test frame"""
        # Create a blank 720x1280 frame with lane markings
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Draw white lane lines
        # Left lane
        cv2.line(frame, (400, 720), (550, 400), (255, 255, 255), 10)
        # Right lane
        cv2.line(frame, (880, 720), (730, 400), (255, 255, 255), 10)
        
        return frame
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector is not None
        assert detector.config is not None
        assert detector.metrics is not None
        assert isinstance(detector.left_lane_history, object)
        assert isinstance(detector.right_lane_history, object)
    
    def test_region_of_interest(self, detector, sample_frame):
        """Test ROI masking"""
        gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
        masked, polygon = detector.region_of_interest(gray)
        
        assert masked.shape == gray.shape
        assert polygon is not None
        assert len(polygon[0]) == 4  # 4 vertices
    
    def test_adaptive_canny(self, detector, sample_frame):
        """Test adaptive Canny edge detection"""
        gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
        edges = detector.adaptive_canny(gray)
        
        assert edges.shape == gray.shape
        assert edges.dtype == np.uint8
        # Should detect some edges
        assert np.count_nonzero(edges) > 0
    
    def test_process_frame(self, detector, sample_frame):
        """Test frame processing"""
        output, status, offset = detector.process_frame(sample_frame)
        
        assert output.shape == sample_frame.shape
        assert status in ["CENTER", "WARNING", "CRITICAL", "NO_LANES", "UNCERTAIN"]
        assert offset is None or isinstance(offset, float)
    
    def test_fit_lane_line(self, detector):
        """Test lane line fitting"""
        # Create sample line segments
        lines = [
            np.array([[100, 500, 150, 400]]),
            np.array([[160, 390, 200, 300]]),
            np.array([[210, 290, 250, 200]])
        ]
        
        fit = detector.fit_lane_line(lines)
        
        assert fit is not None
        assert len(fit) == 2  # slope and intercept
    
    def test_fit_lane_line_insufficient_points(self, detector):
        """Test lane fitting with insufficient points"""
        lines = []
        fit = detector.fit_lane_line(lines)
        assert fit is None
        
        lines = [np.array([[100, 500, 101, 499]])]
        fit = detector.fit_lane_line(lines)
        # Should handle gracefully
    
    def test_smooth_lane_lines(self, detector):
        """Test temporal smoothing"""
        left_fit = np.array([1.5, 200])
        right_fit = np.array([-1.5, 1000])
        
        # First call - should add to history
        left_smooth, right_smooth = detector.smooth_lane_lines(left_fit, right_fit)
        
        assert left_smooth is not None
        assert right_smooth is not None
        assert len(detector.left_lane_history) == 1
        assert len(detector.right_lane_history) == 1
        
        # Second call - should average
        left_smooth2, right_smooth2 = detector.smooth_lane_lines(left_fit, right_fit)
        assert len(detector.left_lane_history) == 2
    
    def test_calculate_lateral_offset(self, detector):
        """Test lateral offset calculation"""
        left_fit = np.array([1.0, 200.0])
        right_fit = np.array([-1.0, 1000.0])
        
        offset, confidence = detector.calculate_lateral_offset(left_fit, right_fit, 1280, 720)
        
        assert offset is not None
        assert -1.0 <= offset <= 1.0
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_lateral_offset_invalid(self, detector):
        """Test offset calculation with invalid inputs"""
        # None inputs
        offset, confidence = detector.calculate_lateral_offset(None, None, 1280, 720)
        assert offset is None
        assert confidence == 0.0
        
        # Invalid lane positions (right < left)
        left_fit = np.array([0.0, 800.0])
        right_fit = np.array([0.0, 400.0])
        offset, confidence = detector.calculate_lateral_offset(left_fit, right_fit, 1280, 720)
        assert offset is None
    
    def test_determine_status(self, detector):
        """Test status determination"""
        # Center status
        status = detector._determine_status(
            np.array([1.0, 200.0]),
            np.array([-1.0, 1000.0]),
            0.05,  # Small offset
            0.9    # High confidence
        )
        assert status == "CENTER"
        
        # Warning status
        status = detector._determine_status(
            np.array([1.0, 200.0]),
            np.array([-1.0, 1000.0]),
            0.20,  # Medium offset
            0.9
        )
        assert status == "WARNING"
        
        # Critical status
        status = detector._determine_status(
            np.array([1.0, 200.0]),
            np.array([-1.0, 1000.0]),
            0.30,  # Large offset
            0.9
        )
        assert status == "CRITICAL"
        
        # No lanes
        status = detector._determine_status(None, None, None, 0.0)
        assert status == "NO_LANES"
        
        # Uncertain
        status = detector._determine_status(
            np.array([1.0, 200.0]),
            np.array([-1.0, 1000.0]),
            0.05,
            0.2    # Low confidence
        )
        assert status == "UNCERTAIN"
    
    def test_draw_advanced_lanes(self, detector, sample_frame):
        """Test lane visualization"""
        left_fit = np.array([1.0, 200.0])
        right_fit = np.array([-1.0, 1000.0])
        
        output = detector.draw_advanced_lanes(
            sample_frame, left_fit, right_fit,
            "CENTER", 0.05, 0.9
        )
        
        assert output.shape == sample_frame.shape
        # Check that something was drawn (frame changed)
        assert not np.array_equal(output, sample_frame)


class TestLaneMetrics:
    """Test cases for metrics tracking"""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics instance"""
        return LaneMetrics()
    
    def test_metrics_initialization(self, metrics):
        """Test metrics initialization"""
        assert metrics.total_frames == 0
        assert metrics.center_frames == 0
        assert metrics.warning_frames == 0
        assert metrics.critical_frames == 0
    
    def test_metrics_update(self, metrics):
        """Test metrics update"""
        metrics.update("CENTER", 0.05, 0.9)
        
        assert metrics.total_frames == 1
        assert metrics.center_frames == 1
        assert len(metrics.lane_positions) == 1
    
    def test_metrics_statistics(self, metrics):
        """Test statistics calculation"""
        # Add some test data
        for i in range(100):
            if i < 70:
                metrics.update("CENTER", 0.05, 0.9)
            elif i < 90:
                metrics.update("WARNING", 0.18, 0.8)
            else:
                metrics.update("CRITICAL", 0.28, 0.7)
        
        stats = metrics.get_statistics()
        
        assert stats['center_percentage'] == 70.0
        assert stats['warning_percentage'] == 20.0
        assert stats['critical_percentage'] == 10.0
        assert 'avg_offset' in stats
        assert 'max_offset' in stats
    
    def test_metrics_score_calculation(self, metrics):
        """Test score calculation"""
        # Perfect score scenario
        for _ in range(100):
            metrics.update("CENTER", 0.01, 1.0)
        
        score = metrics.calculate_score()
        assert 90 <= score <= 100
        
        # Poor score scenario
        metrics2 = LaneMetrics()
        for _ in range(100):
            metrics2.update("CRITICAL", 0.35, 0.5)
        
        score2 = metrics2.calculate_score()
        assert score2 < 50
    
    def test_metrics_report_generation(self, metrics):
        """Test report generation"""
        # Add some data
        for _ in range(50):
            metrics.update("CENTER", 0.05, 0.9)
        
        report = metrics.get_report()
        
        assert isinstance(report, str)
        assert "LANE DEPARTURE WARNING SYSTEM" in report
        assert "Total Frames" in report
        assert "Overall Performance" in report
    
    def test_metrics_fps_calculation(self, metrics):
        """Test FPS calculation"""
        # Simulate frame processing times
        for _ in range(10):
            metrics.update_frame_time(0.033)  # ~30 FPS
        
        fps = metrics.get_fps()
        assert 25 <= fps <= 35  # Allow some tolerance


class TestAudioAlert:
    """Test cases for audio alerts"""
    
    @pytest.fixture
    def audio(self):
        """Create audio alert instance"""
        with patch('pygame.mixer.init'):
            return AudioAlert(enabled=False)  # Disable actual audio
    
    def test_audio_initialization(self, audio):
        """Test audio initialization"""
        assert audio.volume >= 0.0
        assert audio.volume <= 1.0
    
    def test_audio_cooldown(self, audio):
        """Test alert cooldown mechanism"""
        audio.enabled = True
        
        # First alert should be allowed
        assert audio._can_play_alert()
        
        # Immediate second alert should be blocked
        assert not audio._can_play_alert()
    
    @patch('pygame.sndarray.make_sound')
    @patch('pygame.mixer.init')
    def test_warning_alert(self, mock_init, mock_sound):
        """Test warning alert generation"""
        audio = AudioAlert(enabled=True)
        audio.warning_alert()
        # Should not raise exception
    
    @patch('pygame.sndarray.make_sound')
    @patch('pygame.mixer.init')
    def test_critical_alert(self, mock_init, mock_sound):
        """Test critical alert generation"""
        audio = AudioAlert(enabled=True)
        audio.critical_alert()
        # Should not raise exception


class TestIntegration:
    """Integration tests"""
    
    def test_full_processing_pipeline(self):
        """Test complete processing pipeline"""
        config = LaneDetectionConfig()
        
        with patch('advanced_ldws.AudioAlert'):
            detector = AdvancedLaneDetector(config)
        
        # Create test frame with lanes
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.line(frame, (400, 720), (550, 400), (255, 255, 255), 10)
        cv2.line(frame, (880, 720), (730, 400), (255, 255, 255), 10)
        
        # Process multiple frames
        for _ in range(10):
            output, status, offset = detector.process_frame(frame)
            assert output is not None
            assert status is not None
        
        # Check metrics were updated
        assert detector.metrics.total_frames == 10
        
        # Get report
        report = detector.metrics.get_report()
        assert isinstance(report, str)
        
        # Cleanup
        detector.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, '-v'])