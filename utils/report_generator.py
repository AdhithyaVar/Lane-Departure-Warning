#!/usr/bin/env python3
"""
Report Generator
Generate detailed HTML and PDF reports from LDWS metrics

Usage:
    from utils.report_generator import ReportGenerator
    generator = ReportGenerator(metrics)
    generator.generate_html_report('report.html')
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class ReportGenerator:
    """Generate comprehensive reports from lane detection metrics"""
    
    def __init__(self, metrics_data: Dict[str, Any]):
        self.metrics = metrics_data
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_html_report(self, output_path: str) -> None:
        """Generate HTML report with visualizations"""
        html = self._generate_html_content()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ HTML report generated: {output_path}")
    
    def _generate_html_content(self) -> str:
        """Generate HTML content"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LDWS Performance Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .score-circle {{
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: conic-gradient(
                #667eea 0% {self.metrics.get('score', 0)}%,
                #e0e0e0 {self.metrics.get('score', 0)}% 100%
            );
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px auto;
            position: relative;
        }}
        
        .score-circle::before {{
            content: '';
            position: absolute;
            width: 160px;
            height: 160px;
            background: white;
            border-radius: 50%;
        }}
        
        .score-text {{
            position: relative;
            z-index: 1;
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 30px;
            margin: 10px 0;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-weight: bold;
            transition: width 1s ease;
        }}
        
        .status-good {{ color: #4caf50; }}
        .status-warning {{ color: #ff9800; }}
        .status-critical {{ color: #f44336; }}
        
        .footer {{
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 Lane Departure Warning System</h1>
            <p>Performance Analysis Report</p>
            <p style="font-size: 0.9em; margin-top: 10px;">{self.timestamp}</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2 class="section-title">Overall Performance Score</h2>
                <div class="score-circle">
                    <span class="score-text">{self.metrics.get('score', 0):.0f}</span>
                </div>
                <p style="text-align: center; font-size: 1.2em; color: #666;">
                    {self._get_rating(self.metrics.get('score', 0))}
                </p>
            </div>
            
            <div class="section">
                <h2 class="section-title">Key Metrics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Frames</div>
                        <div class="stat-value">{self.metrics.get('total_frames', 0):,}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Average FPS</div>
                        <div class="stat-value">{self.metrics.get('avg_fps', 0):.1f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Detection Rate</div>
                        <div class="stat-value">{self.metrics.get('detection_rate', 0):.1f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Avg Confidence</div>
                        <div class="stat-value">{self.metrics.get('avg_confidence', 0):.0%}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Lane Keeping Performance</h2>
                
                <div style="margin: 20px 0;">
                    <div style="margin-bottom: 15px;">
                        <strong class="status-good">✓ Center Lane</strong>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {self.metrics.get('center_percentage', 0)}%;">
                                {self.metrics.get('center_percentage', 0):.1f}%
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <strong class="status-warning">⚠ Warning Level</strong>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {self.metrics.get('warning_percentage', 0)}%; background: #ff9800;">
                                {self.metrics.get('warning_percentage', 0):.1f}%
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <strong class="status-critical">✗ Critical Level</strong>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {self.metrics.get('critical_percentage', 0)}%; background: #f44336;">
                                {self.metrics.get('critical_percentage', 0):.1f}%
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Lateral Position Analysis</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Average Offset</div>
                        <div class="stat-value" style="font-size: 2em;">
                            {self.metrics.get('avg_offset', 0):+.4f}
                        </div>
                        <p style="color: #666; margin-top: 10px;">
                            {self._interpret_offset(self.metrics.get('avg_offset', 0))}
                        </p>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Maximum Offset</div>
                        <div class="stat-value" style="font-size: 2em;">
                            {self.metrics.get('max_offset', 0):.4f}
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Stability Index</div>
                        <div class="stat-value" style="font-size: 2em;">
                            {(1 - self.metrics.get('std_offset', 0)) * 100:.1f}%
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Recommendations</h2>
                <div style="background: #f5f7fa; padding: 20px; border-radius: 10px;">
                    {self._generate_recommendations()}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by Advanced LDWS v2.0</p>
            <p style="margin-top: 5px; font-size: 0.9em;">
                For more information, visit the documentation
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    def _get_rating(self, score: float) -> str:
        """Get rating text based on score"""
        if score >= 95:
            return "⭐⭐⭐⭐⭐ Exceptional"
        elif score >= 85:
            return "⭐⭐⭐⭐☆ Excellent"
        elif score >= 75:
            return "⭐⭐⭐☆☆ Good"
        elif score >= 60:
            return "⭐⭐☆☆☆ Acceptable"
        elif score >= 40:
            return "⭐☆☆☆☆ Needs Improvement"
        else:
            return "☆☆☆☆☆ Poor"
    
    def _interpret_offset(self, offset: float) -> str:
        """Interpret lateral offset"""
        if abs(offset) < 0.05:
            return "Perfect centering"
        elif offset < 0:
            return f"Slight left bias"
        else:
            return f"Slight right bias"
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        score = self.metrics.get('score', 0)
        center_pct = self.metrics.get('center_percentage', 0)
        warning_pct = self.metrics.get('warning_percentage', 0)
        critical_pct = self.metrics.get('critical_percentage', 0)
        detection_rate = self.metrics.get('detection_rate', 0)
        
        if score >= 90:
            recommendations.append("✓ <strong>Excellent performance!</strong> Continue current driving practices.")
        elif score >= 75:
            recommendations.append("✓ <strong>Good performance.</strong> Minor improvements possible.")
        else:
            recommendations.append("⚠ <strong>Performance needs attention.</strong>")
        
        if center_pct < 70:
            recommendations.append("• Consider improving lane centering techniques")
        
        if warning_pct > 15:
            recommendations.append("• High warning rate detected - review lane keeping consistency")
        
        if critical_pct > 5:
            recommendations.append("• ⚠ <strong>Critical departures detected</strong> - immediate attention required")
        
        if detection_rate < 90:
            recommendations.append("• Lane detection rate low - check camera positioning and road conditions")
        
        if self.metrics.get('avg_fps', 0) < 20:
            recommendations.append("• Low FPS detected - consider reducing resolution or optimizing performance")
        
        return "<br>".join(recommendations) if recommendations else "✓ No specific recommendations - excellent performance!"
    
    def generate_plots(self, output_dir: str) -> Dict[str, str]:
        """Generate visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # Performance distribution pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = [
            self.metrics.get('center_percentage', 0),
            self.metrics.get('warning_percentage', 0),
            self.metrics.get('critical_percentage', 0)
        ]
        labels = ['Center', 'Warning', 'Critical']
        colors = ['#4caf50', '#ff9800', '#f44336']
        explode = (0.05, 0, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')
        plt.title('Lane Keeping Performance Distribution', fontsize=14, fontweight='bold')
        
        plot_path = output_path / 'performance_distribution.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plots['performance_distribution'] = str(plot_path)
        
        print(f"✓ Generated plot: {plot_path}")
        
        return plots
    
    def export_json(self, output_path: str) -> None:
        """Export metrics as JSON"""
        data = {
            'timestamp': self.timestamp,
            'metrics': self.metrics,
            'rating': self._get_rating(self.metrics.get('score', 0))
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        print(f"✓ JSON export: {output_path}")


# Example usage
if __name__ == "__main__":
    # Sample metrics data
    sample_metrics = {
        'total_frames': 1800,
        'center_percentage': 85.5,
        'warning_percentage': 10.2,
        'critical_percentage': 4.3,
        'detection_rate': 96.5,
        'avg_offset': -0.02,
        'max_offset': 0.18,
        'std_offset': 0.08,
        'avg_confidence': 0.92,
        'avg_fps': 29.8,
        'score': 87.3
    }
    
    generator = ReportGenerator(sample_metrics)
    generator.generate_html_report('sample_report.html')
    generator.generate_plots('output/plots')
    generator.export_json('sample_metrics.json')
    
    print("\n✓ Sample reports generated!")