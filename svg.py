import numpy as np
import os
import math
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

def stroke_length(s, e):
    return np.linalg.norm(np.array(s) - np.array(e))

def draw_bezier_and_polygon_svg(npz_path, output_dir, canvas_width=512, canvas_height=512, min_len_thresh=5):
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

    svg_bezier = create_svg_root(canvas_width, canvas_height)
    svg_polygon = create_svg_root(canvas_width, canvas_height)

    for i in range(len(s)):
        length = stroke_length(s[i], e[i])
        if length < min_len_thresh:
            continue  # skip small strokes

        stroke_color = color_to_svg_rgb(color[i])
        width_px = str(max(w[i], 1.0))

        # --- Bézier Curve (Q) ---
        path_d = f"M {s[i][0]},{s[i][1]} Q {c[i][0]},{c[i][1]} {e[i][0]},{e[i][1]}"
        SubElement(svg_bezier, 'path', {
            'd': path_d,
            'stroke': stroke_color,
            'stroke-width': width_px,
            'fill': 'none'
        })

        # --- Triangle Polygon (s, c, e) ---
        points = f"{s[i][0]},{s[i][1]} {c[i][0]},{c[i][1]} {e[i][0]},{e[i][1]}"
        SubElement(svg_polygon, 'polygon', {
            'points': points,
            'fill': stroke_color,
            'stroke': stroke_color,
            'stroke-width': '0.5'
        })

    # Save output files
    save_svg(svg_bezier, os.path.join(output_dir, "all_strokes_bezier_fixed.svg"))
    save_svg(svg_polygon, os.path.join(output_dir, "all_strokes_polygon_fixed.svg"))

    print(f"\n✅ Saved cleaned SVGs to {output_dir}")

# Example usage
if __name__ == "__main__":
    draw_bezier_and_polygon_svg("/content/TATA_RESEARCH/output/strokes.npz", output_dir="/content/TATA_RESEARCH/output/svg_shapes")
