# Test Videos Directory

This directory contains test video files for the Advanced Lane Departure Warning System.

## Purpose

Store test videos here for:
- Testing lane detection algorithms
- Performance benchmarking
- Batch processing examples
- System validation

## Supported Formats

- `.mp4` (recommended)
- `.avi`
- `.mov`
- `.mkv`
- `.flv`

## Sample Videos

You can download sample driving videos from:

1. **Dashcam Videos**
   - [Pexels Free Videos](https://www.pexels.com/search/videos/driving/)
   - [Pixabay Free Videos](https://pixabay.com/videos/search/driving/)

2. **Dataset Resources**
   - [Udacity Self-Driving Car Dataset](https://github.com/udacity/self-driving-car)
   - [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
   - [BDD100K Dataset](https://bdd-data.berkeley.edu/)

## File Naming Convention

Recommended naming format:
```
{environment}_{location}_{date}_{sequence}.mp4

Examples:
- city_downtown_20240115_001.mp4
- highway_i95_20240115_002.mp4
- night_suburban_20240115_003.mp4
```

## Usage

### Single Video Processing
```bash
python advanced_ldws.py --input data/videos/test_video.mp4
```

### Batch Processing
```bash
python scripts/batch_process.py --input data/videos/ --output output/processed/
```

### With Specific Configuration
```bash
python advanced_ldws.py --input data/videos/night_video.mp4 --config configs/config_night.json
```

## Storage Guidelines

- Keep individual videos under 500MB for optimal processing
- Use compressed formats (H.264 codec recommended)
- Maintain resolution between 720p and 1080p
- Organize by environment (city/, highway/, night/)

## Git Ignore

Video files are ignored by git to keep repository size small. Only README and .gitkeep are tracked.

To share videos, use:
- Cloud storage links
- External dataset repositories
- Separate data packages

## Testing Scenarios

Create or download videos for these scenarios:

1. **Normal Driving**
   - Straight roads
   - Gentle curves
   - Good lighting

2. **Challenging Conditions**
   - Sharp turns
   - Night driving
   - Rain/snow
   - Construction zones
   - Unclear lane markings

3. **Edge Cases**
   - Lane changes
   - Merging
   - Exit ramps
   - Parking lots

## Processing Tips

- Test with short clips (10-30 seconds) first
- Use appropriate config for environment
- Monitor FPS during processing
- Check output reports for accuracy

## Example Workflow

```bash
# 1. Add test video
cp ~/Downloads/dashcam_video.mp4 data/videos/

# 2. Quick test with default config
python advanced_ldws.py --input data/videos/dashcam_video.mp4 --duration 10

# 3. Full processing with reports
python advanced_ldws.py --input data/videos/dashcam_video.mp4

# 4. Check results
ls output/reports/
```

## Notes

- Maximum recommended video duration: 10 minutes
- Longer videos should be split for better processing
- Use calibration tool first for new camera angles
- Verify ROI settings match your video perspective

---

For questions about test videos, see [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md)