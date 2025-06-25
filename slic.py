import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread, imsave
from scipy.spatial import ConvexHull
import os

def segment_style(image, num_paths, output_dir):
    segments = slic(image, n_segments=num_paths, compactness=2, sigma=1, start_label=0)
    segmented_image = mark_boundaries(image, segments) * 255
    imsave(os.path.join(output_dir, 'segmented.png'), segmented_image.astype(np.uint8))
    return segments

def clusters_to_strokes(segments, img, H, W):
    segments += np.abs(np.min(segments))
    num_clusters = np.max(segments)
    clusters_params = {'s': [], 'e': [], 'width': [], 'color_rgb': []}

    print('Extracting stroke parameters...')
    for cluster_idx in range(num_clusters + 1):
        cluster_mask = segments == cluster_idx
        if np.sum(cluster_mask) < 5:
            continue
        cluster_mask_nonzeros = np.nonzero(cluster_mask)
        cluster_points = np.stack((cluster_mask_nonzeros[0], cluster_mask_nonzeros[1]), axis=-1)
        try:
            convex_hull = ConvexHull(cluster_points)
        except:
            continue

        border_points = cluster_points[convex_hull.simplices.reshape(-1)]
        dist = np.sum((np.expand_dims(border_points, axis=1) - border_points) ** 2, axis=-1)
        max_idx_a, max_idx_b = np.nonzero(dist == np.max(dist))
        point_a = border_points[max_idx_a[0]]
        point_b = border_points[max_idx_b[0]]
        v_ab = point_b - point_a

        distances = np.zeros(len(border_points))
        for i, point in enumerate(border_points):
            v_ap = point - point_a
            distance = np.abs(np.cross(v_ab, v_ap)) / np.linalg.norm(v_ab)
            distances[i] = distance
        average_width = np.mean(distances)
        if average_width == 0.0:
            continue

        clusters_params['s'].append(point_a / img.shape[:2])
        clusters_params['e'].append(point_b / img.shape[:2])
        clusters_params['width'].append(average_width)
        clusters_params['color_rgb'].append(np.mean(img[cluster_mask], axis=0))

    for key in clusters_params:
        clusters_params[key] = np.array(clusters_params[key])

    s = clusters_params['s']
    e = clusters_params['e']
    width = clusters_params['width']
    color = clusters_params['color_rgb']

    s[..., 0] *= H
    s[..., 1] *= W
    e[..., 0] *= H
    e[..., 1] *= W
    c = (s + e) / 2.

    s, e, c, width, color = [x.astype(np.float32) for x in [s, e, c, width, color]]
    s = s[..., ::-1]
    e = e[..., ::-1]
    c = c[..., ::-1]

    return s, e, c, width, color

def save_stroke_params(s, e, c, width, color, output_path):
    np.savez(output_path, start=s, end=e, center=c, width=width, color=color)
    print(f"Saved stroke parameters to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--num_strokes', type=int, default=3000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image = imread(args.image_path)
    if image.dtype != np.float32:
        image = image / 255.0
    H, W = image.shape[:2]

    segments = segment_style(image, args.num_strokes, args.output_dir)
    s, e, c, width, color = clusters_to_strokes(segments, image, H, W)

    save_stroke_params(s, e, c, width, color, os.path.join(args.output_dir, "strokes.npz"))
