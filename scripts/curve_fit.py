from svgpathtools import svg2paths, Path, Line, CubicBezier, wsvg
import numpy as np
from scipy.optimize import curve_fit
import argparse
# Function to fit a cubic Bezier curve to a set of points
def fit_cubic_bezier(points):
    def bezier_curve(t, p0, p1, p2, p3):
        return (1-t)**3 * p0 + 3 * (1-t)**2 * t * p1 + 3 * (1-t) * t**2 * p2 + t**3 * p3

    # Fit separate curves for x and y coordinates
    t_values = np.linspace(0, 1, len(points))
    x_points = [p.real for p in points]
    y_points = [p.imag for p in points]

    popt_x, _ = curve_fit(bezier_curve, t_values, x_points)
    popt_y, _ = curve_fit(bezier_curve, t_values, y_points)

    # Create a cubic Bezier curve from the fitted parameters
    p0 = complex(popt_x[0], popt_y[0])
    p1 = complex(popt_x[1], popt_y[1])
    p2 = complex(popt_x[2], popt_y[2])
    p3 = complex(popt_x[3], popt_y[3])

    return CubicBezier(p0, p1, p2, p3)

# Read the SVG file

parser = argparse.ArgumentParser(description='Fit cubic Bezier curves to line segments in an SVG file')
parser.add_argument('--input', '-i', dest='input', metavar='FILE', help='input SVG file', required=True)
args = parser.parse_args()

MAX_PATHS = 50

input_svg = args.input
paths, attributes = svg2paths(input_svg)

# Iterate through each path and simplify line segments
new_paths = []
for path in paths:
    simplified_path = Path()

    # Collect consecutive line segments
    line_segments = []
    for segment in path:
        if isinstance(segment, Line):
            line_segments.append(segment.start)
            line_segments.append(segment.end)
        
        if len(line_segments) > MAX_PATHS or not isinstance(segment, Line):
            # If we encounter a non-line segment, fit the previous lines and add the segment
            if len(line_segments) > 2:
                bezier = fit_cubic_bezier(line_segments)
                simplified_path.append(bezier)
                line_segments = []

            simplified_path.append(segment)  # Add the non-line segment

    # If any remaining line segments exist, fit them as well
    if len(line_segments) > 2:
        bezier = fit_cubic_bezier(line_segments)
        simplified_path.append(bezier)

    new_paths.append(simplified_path)

# Write the simplified paths to a new SVG file
output_svg = 'output.svg'
wsvg(new_paths, filename=output_svg)
