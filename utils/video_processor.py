#!/usr/bin/env python3
"""
Video Processing Utilities
Helper functions for video I/O, manipulation, and conversion

Usage:
    from utils.video_processor import VideoProcessor
    processor = VideoProcessor()
    processor.extract_frames('video.mp4', 'output_dir/')
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Generator
from datetime import timedelta


class VideoProcessor:
    """Utility class for video processing operations"""
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        return info
    
    @staticmethod
    def extract_frames(video_path: str, output_dir: str, 
                      interval: int = 30, prefix: str = 'frame') -> List[str]:
        """Extract frames from video at specified interval"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        saved_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                filename = output_path / f"{prefix}_{frame_count:06d}.jpg"
                cv2.imwrite(str(filename), frame)
                saved_frames.append(str(filename))
            
            frame_count += 1
        
        cap.release()
        print(f"✓ Extracted {len(saved_frames)} frames to {output_dir}")
        return saved_frames
    
    @staticmethod
    def resize_video(input_path: str, output_path: str, 
                    width: int, height: int) -> None:
        """Resize video to specified dimensions"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            resized = cv2.resize(frame, (width, height))
            out.write(resized)
        
        cap.release()
        out.release()
        print(f"✓ Resized video saved: {output_path}")
    
    @staticmethod
    def trim_video(input_path: str, output_path: str,
                   start_time: float, end_time: float) -> None:
        """Trim video to specified time range (in seconds)"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            current_frame += 1
        
        cap.release()
        out.release()
        print(f"✓ Trimmed video saved: {output_path}")
    
    @staticmethod
    def create_video_from_frames(frame_dir: str, output_path: str,
                                fps: int = 30, pattern: str = '*.jpg') -> None:
        """Create video from image frames"""
        frame_path = Path(frame_dir)
        frames = sorted(frame_path.glob(pattern))
        
        if not frames:
            raise ValueError(f"No frames found in {frame_dir}")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frames[0]))
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_file in frames:
            frame = cv2.imread(str(frame_file))
            out.write(frame)
        
        out.release()
        print(f"✓ Video created from {len(frames)} frames: {output_path}")
    
    @staticmethod
    def concatenate_videos(video_paths: List[str], output_path: str) -> None:
        """Concatenate multiple videos into one"""
        if not video_paths:
            raise ValueError("No videos to concatenate")
        
        # Get properties from first video
        cap = cv2.VideoCapture(video_paths[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize if dimensions don't match
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                out.write(frame)
            
            cap.release()
        
        out.release()
        print(f"✓ Concatenated {len(video_paths)} videos: {output_path}")
    
    @staticmethod
    def apply_stabilization(input_path: str, output_path: str,
                          smoothing_radius: int = 30) -> None:
        """Apply basic video stabilization"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Read first frame
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        transforms = []
        
        # Calculate transforms between frames
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect feature points
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                               qualityLevel=0.01, minDistance=30)
            
            if prev_pts is None:
                transforms.append([0, 0, 0])
                prev_gray = curr_gray
                continue
            
            # Calculate optical flow
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            
            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]
            
            # Estimate transform
            if len(prev_pts) >= 3:
                m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
                if m is not None:
                    dx = m[0, 2]
                    dy = m[1, 2]
                    da = np.arctan2(m[1, 0], m[0, 0])
                    transforms.append([dx, dy, da])
                else:
                    transforms.append([0, 0, 0])
            else:
                transforms.append([0, 0, 0])
            
            prev_gray = curr_gray
        
        cap.release()
        
        # Calculate smooth trajectory
        transforms = np.array(transforms)
        trajectory = np.cumsum(transforms, axis=0)
        
        # Smooth trajectory
        smoothed_trajectory = np.copy(trajectory)
        for i in range(3):
            smoothed_trajectory[:, i] = VideoProcessor._smooth_signal(trajectory[:, i], smoothing_radius)
        
        # Calculate smooth transforms
        smooth_transforms = smoothed_trajectory - trajectory
        smooth_transforms = transforms + smooth_transforms
        
        # Apply stabilization
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i, transform in enumerate(smooth_transforms):
            ret, frame = cap.read()
            if not ret:
                break
            
            dx, dy, da = transform
            m = np.array([[np.cos(da), -np.sin(da), dx],
                         [np.sin(da), np.cos(da), dy]])
            
            frame_stabilized = cv2.warpAffine(frame, m, (width, height))
            out.write(frame_stabilized)
        
        cap.release()
        out.release()
        print(f"✓ Stabilized video saved: {output_path}")
    
    @staticmethod
    def _smooth_signal(signal: np.ndarray, radius: int) -> np.ndarray:
        """Smooth signal using moving average"""
        window_size = 2 * radius + 1
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(signal, kernel, mode='same')
        return smoothed
    
    @staticmethod
    def frame_iterator(video_path: str, skip_frames: int = 0) -> Generator:
        """Iterate through video frames"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if skip_frames == 0 or frame_count % (skip_frames + 1) == 0:
                    yield frame_count, frame
                
                frame_count += 1
        finally:
            cap.release()
    
    @staticmethod
    def add_timestamp_overlay(input_path: str, output_path: str,
                            position: Tuple[int, int] = (10, 30)) -> None:
        """Add timestamp overlay to video"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate timestamp
            timestamp = timedelta(seconds=frame_count/fps)
            timestamp_str = str(timestamp).split('.')[0]  # Remove microseconds
            
            # Add text overlay
            cv2.putText(frame, timestamp_str, position,
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        print(f"✓ Video with timestamp saved: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Example: Get video info
    try:
        info = VideoProcessor.get_video_info('test_video.mp4')
        print("\nVideo Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Example skipped: {e}")
    
    print("\n✓ VideoProcessor utility loaded")