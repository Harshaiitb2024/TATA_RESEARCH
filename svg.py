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

def draw_shapes_from_strokes(npz_path, output_dir):
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

    for i in range(len(s)):
        stroke_color = color_to_svg_rgb(color[i])
        width = str(max(w[i], 1.0))  # avoid zero width

        # LINE
        svg = create_svg_root()
        SubElement(svg, 'line', {
            'x1': str(s[i][0]), 'y1': str(s[i][1]),
            'x2': str(e[i][0]), 'y2': str(e[i][1]),
            'stroke': stroke_color,
            'stroke-width': width
        })
        save_svg(svg, f"{output_dir}/stroke_{i:03d}_line.svg")

        # QUADRATIC BEZIER
        svg = create_svg_root()
        path_str = f"M {s[i][0]},{s[i][1]} Q {c[i][0]},{c[i][1]} {e[i][0]},{e[i][1]}"
        SubElement(svg, 'path', {
            'd': path_str,
            'stroke': stroke_color,
            'stroke-width': width,
            'fill': 'none'
        })
        save_svg(svg, f"{output_dir}/stroke_{i:03d}_bezier.svg")

        # CIRCLE at center c[i] with radius ~ width
        svg = create_svg_root()
        SubElement(svg, 'circle', {
            'cx': str(c[i][0]),
            'cy': str(c[i][1]),
            'r': str(w[i] / 2),
            'stroke': stroke_color,
            'stroke-width': '1',
            'fill': stroke_color
        })
        save_svg(svg, f"{output_dir}/stroke_{i:03d}_circle.svg")

        # ELLIPSE from s→e as axis
        svg = create_svg_root()
        rx = np.linalg.norm(s[i] - e[i]) / 2
        ry = w[i] / 2
        center_x = (s[i][0] + e[i][0]) / 2
        center_y = (s[i][1] + e[i][1]) / 2
        SubElement(svg, 'ellipse', {
            'cx': str(center_x),
            'cy': str(center_y),
            'rx': str(rx),
            'ry': str(ry),
            'fill': stroke_color,
            'stroke': stroke_color
        })
        save_svg(svg, f"{output_dir}/stroke_{i:03d}_ellipse.svg")

        # TRIANGLE (polygon using s, c, e)
        svg = create_svg_root()
        points = f"{s[i][0]},{s[i][1]} {c[i][0]},{c[i][1]} {e[i][0]},{e[i][1]}"
        SubElement(svg, 'polygon', {
            'points': points,
            'stroke': stroke_color,
            'stroke-width': '1',
            'fill': stroke_color
        })
        save_svg(svg, f"{output_dir}/stroke_{i:03d}_triangle.svg")

# Example usage
if __name__ == "__main__":
    draw_shapes_from_strokes("/content/TATA_RESEARCH/output/strokes.npz", output_dir="/content/TATA_RESEARCH/output/svg_shapes")
