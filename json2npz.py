# json_to_npz.py -
import json, argparse, math, numpy as np
from shapely import wkt
from shapely.geometry import Polygon

def rgba_hex_to_float3(hexstr):
    # '#RRGGBB'  ➜  (r,g,b) in [0,1]
    hexstr = hexstr.lstrip('#')
    r, g, b = tuple(int(hexstr[i:i+2], 16) for i in (0,2,4))
    return np.array([r,g,b], np.float32) / 255.

def angle_0to1(rad):
    # –π…π ➜ 0…1  (paper stores θ that way)
    deg = math.degrees(rad)
    return (deg + 180.0) / 360.0

def row_to_vector(row, canvas_w, canvas_h):
    # shape
    poly = wkt.loads(row['wkt_string'])          # shapely Polygon
    rect = poly.minimum_rotated_rectangle
    (x0,y0),(x1,y1),(x2,y2),(x3,y3) = rect.exterior.coords[:4]
    w = math.hypot(x1-x0, y1-y0)
    h = math.hypot(x2-x1, y2-y1)
    xc = (x0+x2) * 0.5 / canvas_w
    yc = (y0+y2) * 0.5 / canvas_h
    theta = angle_0to1(math.atan2(y1-y0, x1-x0))

    # colour
    head_rgb = rgba_hex_to_float3(row.get('fill_colour', '#000000'))
    tail_rgb = head_rgb.copy()                   # flat colour – duplicate

    # alpha
    alpha = np.float32(row.get('opacity', 1.0))  # already 0‑1? else normalise

    return np.concatenate([[xc,yc,w/canvas_w,h/canvas_h,theta],
                           head_rgb, tail_rgb,
                           [alpha]])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--out',  default='strokes.npz')
    args = p.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    # assume the first entry stores canvas size
    canvas_w = float(data[0]['canvas_width'])
    canvas_h = float(data[0]['canvas_height'])

    vectors = [row_to_vector(r, canvas_w, canvas_h) for r in data]
    V = np.stack(vectors, axis=0)                # [N,12]

    # split into three arrays exactly like the paper
    x_ctt   = V[:,  0:5]                         # [N,5]
    x_color = V[:, 5:11]                         # [N,6]
    x_alpha = V[:, 11:12]                        # [N,1]

    np.savez(args.out, x_ctt=x_ctt, x_color=x_color, x_alpha=x_alpha)
    print(f'✓ Saved {args.out}  ({len(V)} strokes)')

if __name__ == '__main__':
    main()
