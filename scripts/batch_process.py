#!/usr/bin/env python3
"""
Batch Video Processing Script
Process multiple video files with Advanced LDWS

Usage:
    python batch_process.py --input data/videos/ --output output/processed/
    python batch_process.py --input data/videos/test.mp4 --config configs/config_night.json
"""

import argparse
import cv2
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_ldws import LaneDetectionConfig, AdvancedLaneDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch video processing with progress tracking"""
    
    def __init__(self, config_file: str = "ldws_config.json"):
        self.config = LaneDetectionConfig(config_file)
        self.results: List[Dict[str, Any]] = []
    
    def process_video(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process a single video file"""
        logger.info(f"Processing: {video_path.name}")
        
        # Initialize detector
        detector = AdvancedLaneDetector(self.config)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return {'status': 'error', 'video': str(video_path), 'message': 'Failed to open'}
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        
        # Create output video writer (optional)
        output_video_path = output_dir / f"{video_path.stem}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        warning_frames = 0
        critical_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                output, status, offset = detector.process_frame(frame)
                
                # Write to output video
                out.write(output)
                
                # Track warnings
                if status == "WARNING":
                    warning_frames += 1
                elif status == "CRITICAL":
                    critical_frames += 1
                
                # Progress indicator
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            return {'status': 'interrupted', 'video': str(video_path)}
        
        finally:
            cap.release()
            out.release()
        
        # Save report
        report_path = output_dir / f"{video_path.stem}_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(detector.metrics.get_report())
        
        # Save statistics JSON
        stats_path = output_dir / f"{video_path.stem}_stats.json"
        stats = detector.metrics.get_statistics()
        stats['video_file'] = str(video_path)
        stats['total_frames'] = frame_count
        stats['warning_frames'] = warning_frames
        stats['critical_frames'] = critical_frames
        stats['score'] = detector.metrics.calculate_score()
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        
        # Cleanup
        detector.cleanup()
        
        logger.info(f"Completed: {video_path.name}")
        logger.info(f"Score: {stats['score']:.1f}/100")
        logger.info(f"Output: {output_video_path}")
        logger.info(f"Report: {report_path}")
        
        return {
            'status': 'success',
            'video': str(video_path),
            'output_video': str(output_video_path),
            'report': str(report_path),
            'stats': stats
        }
    
    def process_directory(self, input_dir: Path, output_dir: Path, 
                         extensions: List[str] = None) -> List[Dict[str, Any]]:
        """Process all videos in a directory"""
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        
        # Find all video files
        video_files = []
        for ext in extensions:
            video_files.extend(input_dir.glob(f'*{ext}'))
            video_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Process each video
        results = []
        for i, video_path in enumerate(video_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing video {i}/{len(video_files)}")
            logger.info(f"{'='*60}\n")
            
            result = self.process_video(video_path, output_dir)
            results.append(result)
        
        self.results = results
        return results
    
    def generate_summary_report(self, output_dir: Path) -> None:
        """Generate summary report for all processed videos"""
        if not self.results:
            logger.warning("No results to generate summary")
            return
        
        summary_path = output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("BATCH PROCESSING SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total Videos Processed: {len(self.results)}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            successful = sum(1 for r in self.results if r['status'] == 'success')
            failed = len(self.results) - successful
            
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n\n")
            
            f.write("="*70 + "\n")
            f.write("INDIVIDUAL VIDEO RESULTS\n")
            f.write("="*70 + "\n\n")
            
            # Individual results
            for i, result in enumerate(self.results, 1):
                f.write(f"{i}. {Path(result['video']).name}\n")
                f.write(f"   Status: {result['status']}\n")
                
                if result['status'] == 'success':
                    stats = result['stats']
                    f.write(f"   Score: {stats['score']:.1f}/100\n")
                    f.write(f"   Center: {stats['center_percentage']:.1f}%\n")
                    f.write(f"   Warnings: {result['stats']['warning_frames']}\n")
                    f.write(f"   Critical: {result['stats']['critical_frames']}\n")
                    f.write(f"   Avg FPS: {stats['avg_fps']:.1f}\n")
                
                f.write("\n")
            
            # Overall statistics
            if successful > 0:
                f.write("="*70 + "\n")
                f.write("OVERALL STATISTICS\n")
                f.write("="*70 + "\n\n")
                
                scores = [r['stats']['score'] for r in self.results if r['status'] == 'success']
                f.write(f"Average Score: {sum(scores)/len(scores):.1f}/100\n")
                f.write(f"Best Score: {max(scores):.1f}/100\n")
                f.write(f"Worst Score: {min(scores):.1f}/100\n")
        
        logger.info(f"Summary report saved: {summary_path}")
        print(f"\n✓ Summary report: {summary_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Batch process videos with Advanced LDWS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/videos/                      # Process directory
  %(prog)s --input data/videos/test.mp4              # Process single file
  %(prog)s --input data/videos/ --config configs/config_night.json
  %(prog)s --input data/videos/ --extensions .mp4 .avi
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input video file or directory')
    parser.add_argument('--output', '-o', type=str, default='output/processed',
                       help='Output directory for processed videos')
    parser.add_argument('--config', '-c', type=str, default='ldws_config.json',
                       help='Configuration file')
    parser.add_argument('--extensions', nargs='+', default=['.mp4', '.avi', '.mov'],
                       help='Video file extensions to process')
    parser.add_argument('--no-summary', action='store_true',
                       help='Do not generate summary report')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # Validate input
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize processor
    try:
        processor = BatchProcessor(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        return 1
    
    # Process
    try:
        if input_path.is_file():
            # Single file
            logger.info("Processing single video file")
            result = processor.process_video(input_path, output_dir)
            processor.results = [result]
        else:
            # Directory
            logger.info("Processing video directory")
            processor.process_directory(input_path, output_dir, args.extensions)
        
        # Generate summary
        if not args.no_summary and processor.results:
            processor.generate_summary_report(output_dir)
        
        # Print final summary
        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)
        successful = sum(1 for r in processor.results if r['status'] == 'success')
        print(f"Total: {len(processor.results)}, Successful: {successful}")
        print(f"Output directory: {output_dir}")
        print("="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())