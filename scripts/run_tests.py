#!/usr/bin/env python3
"""
Test Runner Script
Runs all tests with coverage reporting

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --verbose          # Verbose output
    python run_tests.py --coverage         # With coverage report
    python run_tests.py --module config    # Run specific module tests
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

# Colors for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str) -> None:
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(text: str) -> None:
    """Print success message"""
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {text}")


def print_error(text: str) -> None:
    """Print error message"""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {text}")


def print_info(text: str) -> None:
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ{Colors.ENDC} {text}")


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status"""
    print_info(f"{description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print_success(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def check_pytest_installed() -> bool:
    """Check if pytest is installed"""
    try:
        import pytest
        print_success(f"pytest {pytest.__version__} found")
        return True
    except ImportError:
        print_error("pytest not found")
        print_info("Install with: pip install pytest pytest-cov pytest-mock")
        return False


def run_unit_tests(verbose: bool = False, module: Optional[str] = None) -> bool:
    """Run unit tests"""
    print_header("Running Unit Tests")
    
    cmd = ['pytest', 'tests/']
    
    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
    
    if module:
        cmd[1] = f'tests/test_{module}.py'
        print_info(f"Running tests for module: {module}")
    
    cmd.extend(['--tb=short', '--color=yes'])
    
    return run_command(cmd, "Unit tests")


def run_coverage_tests() -> bool:
    """Run tests with coverage"""
    print_header("Running Tests with Coverage")
    
    cmd = [
        'pytest',
        'tests/',
        '--cov=.',
        '--cov-report=term-missing',
        '--cov-report=html',
        '--cov-config=.coveragerc',
        '--color=yes'
    ]
    
    success = run_command(cmd, "Coverage tests")
    
    if success:
        print_info("Coverage report generated in htmlcov/index.html")
    
    return success


def run_style_checks() -> bool:
    """Run code style checks"""
    print_header("Running Style Checks")
    
    # Check if tools are available
    try:
        import flake8
        has_flake8 = True
    except ImportError:
        has_flake8 = False
        print_info("flake8 not installed (optional)")
    
    try:
        import black
        has_black = True
    except ImportError:
        has_black = False
        print_info("black not installed (optional)")
    
    success = True
    
    if has_flake8:
        cmd = ['flake8', 'advanced_ldws.py', 'tests/', 'utils/', 'scripts/', 
               '--max-line-length=100', '--ignore=E501,W503']
        if not run_command(cmd, "Flake8 style check"):
            success = False
    
    if has_black:
        cmd = ['black', '--check', '--line-length=100', 
               'advanced_ldws.py', 'tests/', 'utils/', 'scripts/']
        if not run_command(cmd, "Black format check"):
            success = False
    
    return success


def run_type_checks() -> bool:
    """Run type checking with mypy"""
    print_header("Running Type Checks")
    
    try:
        import mypy
    except ImportError:
        print_info("mypy not installed (optional)")
        return True
    
    cmd = ['mypy', 'advanced_ldws.py', '--ignore-missing-imports']
    return run_command(cmd, "MyPy type check")


def run_import_checks() -> bool:
    """Check if all imports work"""
    print_header("Checking Imports")
    
    imports_to_check = [
        'cv2',
        'numpy',
        'pygame',
        'json',
        'pathlib'
    ]
    
    all_ok = True
    for module in imports_to_check:
        try:
            __import__(module)
            print_success(f"{module} import OK")
        except ImportError as e:
            print_error(f"{module} import failed: {e}")
            all_ok = False
    
    return all_ok


def run_quick_tests() -> bool:
    """Run quick smoke tests"""
    print_header("Running Quick Tests")
    
    # Test configuration loading
    print_info("Testing configuration loading...")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from advanced_ldws import LaneDetectionConfig
        config = LaneDetectionConfig()
        print_success("Configuration loaded successfully")
    except Exception as e:
        print_error(f"Configuration test failed: {e}")
        return False
    
    # Test detector initialization
    print_info("Testing detector initialization...")
    try:
        from advanced_ldws import AdvancedLaneDetector
        detector = AdvancedLaneDetector(config)
        detector.cleanup()
        print_success("Detector initialized successfully")
    except Exception as e:
        print_error(f"Detector test failed: {e}")
        return False
    
    return True


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run tests for Advanced LDWS',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose test output')
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Run with coverage report')
    parser.add_argument('--module', '-m', type=str,
                       help='Run tests for specific module (e.g., config, lane_detection)')
    parser.add_argument('--style', '-s', action='store_true',
                       help='Run style checks (flake8, black)')
    parser.add_argument('--types', '-t', action='store_true',
                       help='Run type checks (mypy)')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run quick smoke tests only')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all checks (tests, style, types)')
    
    return parser.parse_args()


def main() -> int:
    """Main entry point"""
    args = parse_arguments()
    
    print_header("Advanced LDWS Test Runner")
    
    # Check if pytest is installed
    if not args.quick and not check_pytest_installed():
        return 1
    
    # Check imports
    if not run_import_checks():
        print_error("Import checks failed")
        return 1
    
    # Track overall success
    all_success = True
    
    # Quick tests
    if args.quick:
        if not run_quick_tests():
            return 1
        print_header("Quick Tests Complete")
        return 0
    
    # Run tests based on arguments
    if args.all or (not args.style and not args.types):
        # Run unit tests
        if args.coverage:
            if not run_coverage_tests():
                all_success = False
        else:
            if not run_unit_tests(args.verbose, args.module):
                all_success = False
    
    # Style checks
    if args.all or args.style:
        if not run_style_checks():
            all_success = False
    
    # Type checks
    if args.all or args.types:
        if not run_type_checks():
            all_success = False
    
    # Final summary
    print_header("Test Summary")
    
    if all_success:
        print_success("All checks passed!")
        print("\n" + Colors.OKGREEN + Colors.BOLD + "✓ SUCCESS" + Colors.ENDC + "\n")
        return 0
    else:
        print_error("Some checks failed")
        print("\n" + Colors.FAIL + Colors.BOLD + "✗ FAILURE" + Colors.ENDC + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())