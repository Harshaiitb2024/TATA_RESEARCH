import numpy as np
import os
from xml.etree.ElementTree import Element, SubElement, ElementTree

def create_svg_root(width=512, height=512):
    return Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'width': str(width),
        'height': str(height),
        'viewBox': f'0 0 {width} {height}',
        'version': '1.1'
    })

def color_to_svg_rgb(color):
    r, g, b = (color * 255).astype(int)
    return f'rgb({r},{g},{b})'

def save_svg(root, filepath):
    tree = ElementTree(root)
    tree.write(filepath)

def draw_strokes_to_combined_svgs(npz_path, output_dir, canvas_width=512, canvas_height=512):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path)

    s = data['start']
    e = data['end']
    c = data['center']
    w = data['width']
    color = data['color']

    print("\n🔹 First 5 stroke parameters:")
    for name, array in [('Start', s), ('End', e), ('Control', c), ('Width', w), ('Color', color)]:
        print(f"{name}:\n{array[:5]}\n")

    # Create SVG roots for 5 shape styles
    svg_line     = create_svg_root(canvas_width, canvas_height)
    svg_bezier   = create_svg_root(canvas_width, canvas_height)
    svg_circle   = create_svg_root(canvas_width, canvas_height)
    svg_ellipse  = create_svg_root(canvas_width, canvas_height)
    svg_triangle = create_svg_root(canvas_width, canvas_height)

    for i in range(len(s)):
        stroke_color = color_to_svg_rgb(color[i])
        width_px = str(max(w[i], 1.0))  # to avoid invisible strokes

        # LINE
        SubElement(svg_line, 'line', {
            'x1': str(s[i][0]), 'y1': str(s[i][1]),
            'x2': str(e[i][0]), 'y2': str(e[i][1]),
            'stroke': stroke_color,
            'stroke-width': width_px
        })

        # QUADRATIC BEZIER
        path_str = f"M {s[i][0]},{s[i][1]} Q {c[i][0]},{c[i][1]} {e[i][0]},{e[i][1]}"
        SubElement(svg_bezier, 'path', {
            'd': path_str,
            'stroke': stroke_color,
            'stroke-width': width_px,
            'fill': 'none'
        })

        # CIRCLE
        SubElement(svg_circle, 'circle', {
            'cx': str(c[i][0]),
            'cy': str(c[i][1]),
            'r': str(w[i] / 2),
            'stroke': stroke_color,
            'stroke-width': '1',
            'fill': stroke_color
        })

        # ELLIPSE
        rx = np.linalg.norm(s[i] - e[i]) / 2
        ry = w[i] / 2
        center_x = (s[i][0] + e[i][0]) / 2
        center_y = (s[i][1] + e[i][1]) / 2
        SubElement(svg_ellipse, 'ellipse', {
            'cx': str(center_x),
            'cy': str(center_y),
            'rx': str(rx),
            'ry': str(ry),
            'fill': stroke_color,
            'stroke': stroke_color
        })

        # TRIANGLE
        points = f"{s[i][0]},{s[i][1]} {c[i][0]},{c[i][1]} {e[i][0]},{e[i][1]}"
        SubElement(svg_triangle, 'polygon', {
            'points': points,
            'stroke': stroke_color,
            'stroke-width': '1',
            'fill': stroke_color
        })

    # Save all 5 SVG files
    save_svg(svg_line,     os.path.join(output_dir, "all_strokes_line.svg"))
    save_svg(svg_bezier,   os.path.join(output_dir, "all_strokes_bezier.svg"))
    save_svg(svg_circle,   os.path.join(output_dir, "all_strokes_circle.svg"))
    save_svg(svg_ellipse,  os.path.join(output_dir, "all_strokes_ellipse.svg"))
    save_svg(svg_triangle, os.path.join(output_dir, "all_strokes_triangle.svg"))

    print(f"\n SVG files saved to: {output_dir}")

# Example usage
if __name__ == "__main__":
    draw_strokes_to_combined_svgs("/content/TATA_RESEARCH/output/strokes.npz", output_dir="/content/TATA_RESEARCH/output/svg_shapes")
