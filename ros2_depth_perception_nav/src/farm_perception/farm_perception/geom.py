"""Shared numpy geometry helpers for depth-geometric perception.

No PCL / sklearn dependency: everything here is plain numpy so the Docker image
stays light and the math is fully transparent for the hardcoded-vs-learned writeup.
"""
import numpy as np


def tf_to_matrix(t):
    """geometry_msgs/TransformStamped -> 4x4 homogeneous transform (numpy)."""
    q = t.transform.rotation
    tr = t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    # rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tr.x, tr.y, tr.z]
    return M


def transform_points(points, M):
    """Apply 4x4 transform M to Nx3 points -> Nx3."""
    if points.shape[0] == 0:
        return points
    homo = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homo @ M.T)[:, :3]


def ransac_ground_plane(points, n_iter=40, tol=0.03, sample_cap=2500, rng=None):
    """RANSAC dominant (near-horizontal) plane fit.

    Returns (normal[3], d, inlier_mask). Plane: normal . p + d = 0, |normal|=1.
    Biased toward horizontal planes (the ground), so bed walls / plants do not win.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = points.shape[0]
    if n < 50:
        # fall back to "z = min" horizontal plane
        return np.array([0.0, 0.0, 1.0]), -np.min(points[:, 2]), np.zeros(n, dtype=bool)

    idx = rng.choice(n, size=min(n, sample_cap), replace=False)
    P = points[idx]
    best_inliers = None
    best_count = -1
    best_plane = (np.array([0.0, 0.0, 1.0]), 0.0)
    for _ in range(n_iter):
        s = P[rng.choice(P.shape[0], size=3, replace=False)]
        v1, v2 = s[1] - s[0], s[2] - s[0]
        nrm = np.cross(v1, v2)
        nn = np.linalg.norm(nrm)
        if nn < 1e-6:
            continue
        nrm = nrm / nn
        # only accept near-horizontal candidate planes (|nz| large)
        if abs(nrm[2]) < 0.85:
            continue
        if nrm[2] < 0:
            nrm = -nrm
        d = -nrm @ s[0]
        dist = np.abs(P @ nrm + d)
        cnt = int(np.count_nonzero(dist < tol))
        if cnt > best_count:
            best_count = cnt
            best_plane = (nrm, d)
    nrm, d = best_plane
    dist_all = np.abs(points @ nrm + d)
    return nrm, d, dist_all < tol


def cluster_2d(xy, cell=0.30, min_pts=8):
    """Grid connected-components clustering of Nx2 points.

    Returns list of (centroid_xy, count). Plants are ~2 m apart so a coarse grid
    with 8-connectivity cleanly separates them without sklearn.
    """
    if xy.shape[0] == 0:
        return []
    keys = np.floor(xy / cell).astype(np.int64)
    from collections import defaultdict
    cell_pts = defaultdict(list)
    for i, k in enumerate(map(tuple, keys)):
        cell_pts[k].append(i)

    visited = set()
    clusters = []
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for start in cell_pts:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        members = []
        while stack:
            c = stack.pop()
            members.extend(cell_pts[c])
            for dx, dy in neigh:
                nb = (c[0] + dx, c[1] + dy)
                if nb in cell_pts and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        if len(members) >= min_pts:
            pts = xy[members]
            clusters.append((pts.mean(axis=0), len(members)))
    return clusters


def find_row_gaps(y_values, weights, bed_halfwidth=0.30, bin_size=0.10,
                  min_weight_frac=0.04):
    """From lateral (y) positions of obstacle points, find bed-row centers and
    the navigable gaps (lane centers) between them.

    Returns (bed_centers, lane_centers) as sorted lists of y.
    """
    if len(y_values) < 30:
        return [], []
    y_values = np.asarray(y_values)
    weights = np.asarray(weights)
    lo, hi = y_values.min(), y_values.max()
    if hi - lo < 0.3:
        return [], []
    bins = np.arange(lo - bin_size, hi + bin_size, bin_size)
    hist, edges = np.histogram(y_values, bins=bins, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    thr = max(hist.max() * min_weight_frac, 1.0)
    occupied = hist > thr

    # group contiguous occupied bins into bed rows
    beds = []
    i = 0
    while i < len(occupied):
        if occupied[i]:
            j = i
            while j < len(occupied) and occupied[j]:
                j += 1
            seg_c = centers[i:j]
            seg_w = hist[i:j]
            beds.append(float(np.average(seg_c, weights=seg_w)))
            i = j
        else:
            i += 1
    # lane centers = midpoints between consecutive bed rows
    lanes = [0.5 * (beds[k] + beds[k + 1]) for k in range(len(beds) - 1)]
    return beds, lanes
