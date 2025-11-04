# Advanced Lane Departure Warning System (LDWS)

A real-time lane departure warning system optimized for Python 3.13 that uses computer vision and advanced algorithms to detect and warn about lane departures. The system processes video input from a camera or video file and provides immediate visual and audio feedback about the vehicle's position within the lane.

## Features

- Real-time lane detection and tracking
- Advanced temporal smoothing for stable lane detection
- Multiple warning levels (normal, warning, critical)
- Audio alerts with configurable sounds
- Performance metrics and comprehensive reporting
- Configurable settings for different environments
- Support for different video sources
- Automatic calibration and adaptation
- Performance optimizations for Python 3.13

## Prerequisites

- Python 3.13 or higher
- OpenCV 4.8.0 or higher
- NumPy 1.24.0 or higher
- PyGame 2.6.1 or higher
- Other dependencies as listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Advanced-LDWS.git
cd Advanced-LDWS
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Basic usage with webcam:
```bash
python advanced_ldws.py
```

2. Configure settings in `ldws_config.json` to optimize for your environment:
- Adjust `canny_low` and `canny_high` for edge detection
- Modify `warning_threshold` and `critical_threshold`
- Customize ROI parameters for your camera position

3. Controls:
- Press 'q' to quit
- Press 's' to save the current frame
- The system will automatically run for the duration specified in config

## Configuration Files

- `ldws_config.json`: Main configuration file
- `configs/config_city.json`: Optimized settings for city driving
- `configs/config_highway.json`: Optimized settings for highway driving
- `configs/config_night.json`: Optimized settings for night driving

## Performance Optimization

The system includes several optimizations for Python 3.13:
- Vectorized operations using NumPy
- Efficient frame buffer management
- Multi-threaded processing where applicable
- Memory usage optimization
- Adaptive thresholding for different lighting conditions

## Output

1. Real-time visualization:
- Lane markings with color-coded status
- Lateral position indicator
- Warning overlays
- FPS counter

2. Report generation:
- Test duration and frames analyzed
- Lane keeping performance statistics
- Lateral position analysis
- Overall score and rating

## Project Structure

```
Advanced-LDWS/
├── advanced_ldws.py       # Main application file
├── ldws_config.json      # Configuration file
├── requirements.txt      # Dependencies
├── configs/             # Environment-specific configs
├── data/               # Calibration and test data
├── docs/              # Documentation
├── models/           # ML models (if used)
├── output/          # Generated outputs
├── scripts/        # Utility scripts
├── tests/         # Test files
└── utils/        # Utility modules
```

## Known Issues

1. Unicode characters in reports may not display correctly in some terminals
2. Performance may vary based on hardware capabilities
3. Lighting conditions can affect detection accuracy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
