#!/usr/bin/env python3
"""
Video Processing Utilities for LDWS
Helper functions for video I/O, frame extraction, and manipulation

Usage:
    python utils/video_processor.py --extract video.mp4 --output frames/
    python utils/video_processor.py --create frames/ --output video.mp4
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Generator, Tuple, Optional, List
from datetime import timedelta
import json


class VideoProcessor:
    """Video processing utilities"""
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'path': video_path,
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'duration': 0,
        }
        
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        cap.release()
        return info
    
    @staticmethod
    def print_video_info(video_path: str):
        """Print video information"""
        info = VideoProcessor.get_video_info(video_path)
        
        print("="*60)
        print(f"Video Information: {Path(video_path).name}")
        print("="*60)
        print(f"Resolution:    {info['width']}x{info['height']}")
        print(f"Frame Count:   {info['frame_count']:,}")
        print(f"FPS:           {info['fps']:.2f}")
        print(f"Duration:      {timedelta(seconds=int(info['duration']))}")
        print(f"Codec:         {info['codec']}")
        print("="*60)
    
    @staticmethod
    def extract_frames(video_path: str, output_dir: Path, 
                      frame_interval: int = 1, 
                      start_frame: int = 0,
                      end_frame: Optional[int] = None) -> int:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            frame_interval: Extract every Nth frame
            start_frame: Starting frame number
            end_frame: Ending frame number (None = all)
        
        Returns:
            Number of frames extracted
        """
        cap = cv2.VideoCapture(video_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        extracted_count = 0
        
        print(f"Extracting frames from: {video_path}")
        print(f"Output directory: {output_dir}")
        print(f"Frame interval: {frame_interval}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count < start_frame:
                frame_count += 1
                continue
            
            if end_frame is not None and frame_count >= end_frame:
                break
            
            if frame_count % frame_interval == 0:
                output_file = output_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(output_file), frame, 
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_count += 1
                
                if extracted_count % 100 == 0:
                    print(f"Extracted {extracted_count} frames...")
            
            frame_count += 1
        
        cap.release()
        print(f"✓ Extraction complete: {extracted_count} frames saved")
        return extracted_count
    
    @staticmethod
    def create_video_from_frames(frame_dir: Path, output_path: str,
                                 fps: float = 30.0,
                                 pattern: str = "*.jpg") -> bool:
        """
        Create video from image frames
        
        Args:
            frame_dir: Directory containing frames
            output_path: Output video path
            fps: Frames per second
            pattern: Frame file pattern
        
        Returns:
            True if successful
        """
        # Get all frame files
        frame_files = sorted(frame_dir.glob(pattern))
        
        if not frame_files:
            print(f"No frames found in {frame_dir} with pattern {pattern}")
            return False
        
        print(f"Creating video from {len(frame_files)} frames...")
        print(f"FPS: {fps}")
        print(f"Output: {output_path}")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                out.write(frame)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(frame_files)} frames...")
        
        out.release()
        print(f"✓ Video created: {output_path}")
        return True
    
    @staticmethod
    def resize_video(input_path: str, output_path: str,
                    width: int, height: int) -> bool:
        """
        Resize video to new dimensions
        
        Args:
            input_path: Input video path
            output_path: Output video path
            width: Target width
            height: Target height
        
        Returns:
            True if successful
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Could not open: {input_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Resizing video to {width}x{height}...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            resized = cv2.resize(frame, (width, height))
            out.write(resized)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        print(f"✓ Resized video saved: {output_path}")
        return True
    
    @staticmethod
    def trim_video(input_path: str, output_path: str,
                  start_time: float = 0.0, end_time: Optional[float] = None) -> bool:
        """
        Trim video to specific time range
        
        Args:
            input_path: Input video path
            output_path: Output video path
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of video)
        
        Returns:
            True if successful
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Could not open: {input_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else None
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Trimming video from {start_time}s to {end_time if end_time else 'end'}...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame = start_frame + frame_count
            if end_frame and current_frame >= end_frame:
                break
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        print(f"✓ Trimmed video saved: {output_path}")
        return True
    
    @staticmethod
    def concatenate_videos(video_paths: List[str], output_path: str) -> bool:
        """
        Concatenate multiple videos into one
        
        Args:
            video_paths: List of input video paths
            output_path: Output video path
        
        Returns:
            True if successful
        """
        if not video_paths:
            print("No videos to concatenate")
            return False
        
        # Get properties from first video
        cap = cv2.VideoCapture(video_paths[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Concatenating {len(video_paths)} videos...")
        
        for i, video_path in enumerate(video_paths):
            print(f"Processing video {i+1}/{len(video_paths)}: {Path(video_path).name}")
            
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
        print(f"✓ Concatenated video saved: {output_path}")
        return True
    
    @staticmethod
    def apply_filter(input_path: str, output_path: str, 
                    filter_type: str = 'grayscale') -> bool:
        """
        Apply filter to video
        
        Args:
            input_path: Input video path
            output_path: Output video path
            filter_type: Filter type ('grayscale', 'blur', 'edge')
        
        Returns:
            True if successful
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Could not open: {input_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Applying {filter_type} filter...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply filter
            if filter_type == 'grayscale':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                filtered = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif filter_type == 'blur':
                filtered = cv2.GaussianBlur(frame, (15, 15), 0)
            elif filter_type == 'edge':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                filtered = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            else:
                filtered = frame
            
            out.write(filtered)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        print(f"✓ Filtered video saved: {output_path}")
        return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Video processing utilities for LDWS',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get video information')
    info_parser.add_argument('video', type=str, help='Video file path')
    
    # Extract frames command
    extract_parser = subparsers.add_parser('extract', help='Extract frames from video')
    extract_parser.add_argument('video', type=str, help='Video file path')
    extract_parser.add_argument('--output', '-o', type=str, required=True,
                               help='Output directory for frames')
    extract_parser.add_argument('--interval', '-i', type=int, default=1,
                               help='Extract every Nth frame')
    extract_parser.add_argument('--start', type=int, default=0,
                               help='Start frame')
    extract_parser.add_argument('--end', type=int, default=None,
                               help='End frame')
    
    # Create video command
    create_parser = subparsers.add_parser('create', help='Create video from frames')
    create_parser.add_argument('frames', type=str, help='Directory with frames')
    create_parser.add_argument('--output', '-o', type=str, required=True,
                              help='Output video path')
    create_parser.add_argument('--fps', type=float, default=30.0,
                              help='Frames per second')
    
    # Resize command
    resize_parser = subparsers.add_parser('resize', help='Resize video')
    resize_parser.add_argument('video', type=str, help='Input video path')
    resize_parser.add_argument('--output', '-o', type=str, required=True,
                              help='Output video path')
    resize_parser.add_argument('--width', '-w', type=int, required=True,
                              help='Target width')
    resize_parser.add_argument('--height', '-h', type=int, required=True,
                              help='Target height')
    
    # Trim command
    trim_parser = subparsers.add_parser('trim', help='Trim video')
    trim_parser.add_argument('video', type=str, help='Input video path')
    trim_parser.add_argument('--output', '-o', type=str, required=True,
                            help='Output video path')
    trim_parser.add_argument('--start', '-s', type=float, default=0.0,
                            help='Start time (seconds)')
    trim_parser.add_argument('--end', '-e', type=float, default=None,
                            help='End time (seconds)')
    
    # Concatenate command
    concat_parser = subparsers.add_parser('concat', help='Concatenate videos')
    concat_parser.add_argument('videos', nargs='+', help='Input video paths')
    concat_parser.add_argument('--output', '-o', type=str, required=True,
                              help='Output video path')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Apply filter to video')
    filter_parser.add_argument('video', type=str, help='Input video path')
    filter_parser.add_argument('--output', '-o', type=str, required=True,
                              help='Output video path')
    filter_parser.add_argument('--type', '-t', type=str, default='grayscale',
                              choices=['grayscale', 'blur', 'edge'],
                              help='Filter type')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    if not args.command:
        print("Error: No command specified. Use --help for usage information.")
        return 1
    
    processor = VideoProcessor()
    
    try:
        if args.command == 'info':
            processor.print_video_info(args.video)
        
        elif args.command == 'extract':
            processor.extract_frames(
                args.video,
                Path(args.output),
                frame_interval=args.interval,
                start_frame=args.start,
                end_frame=args.end
            )
        
        elif args.command == 'create':
            processor.create_video_from_frames(
                Path(args.frames),
                args.output,
                fps=args.fps
            )
        
        elif args.command == 'resize':
            processor.resize_video(
                args.video,
                args.output,
                args.width,
                args.height
            )
        
        elif args.command == 'trim':
            processor.trim_video(
                args.video,
                args.output,
                start_time=args.start,
                end_time=args.end
            )
        
        elif args.command == 'concat':
            processor.concatenate_videos(
                args.videos,
                args.output
            )
        
        elif args.command == 'filter':
            processor.apply_filter(
                args.video,
                args.output,
                filter_type=args.type
            )
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())