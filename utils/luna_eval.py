"""Coordinate and FROC helpers; no model/GPU dependencies."""

import numpy as np

FROC_POINTS = np.array([.125, .25, .5, 1., 2., 4., 8.])


def match_distance_squared(distance_squared, radius_squared, matching_rule='strict'):
    """Strict LUNA radius criterion or explicitly requested historical tolerance."""
    if matching_rule not in ('strict', 'legacy'):
        raise ValueError('matching_rule must be strict or legacy')
    return distance_squared < radius_squared + (5. if matching_rule == 'legacy' else 0.)


def cropped_xyz_to_world(points_xyz, meta):
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N, 3) in x,y,z order')
    crop = np.asarray(meta['crop_start_zyx'], dtype=np.float64)
    origin = np.asarray(meta['resampled_origin_xyz'], dtype=np.float64)
    spacing = np.asarray(meta['resampled_spacing_xyz'], dtype=np.float64)
    direction = np.asarray(meta['resampled_direction'], dtype=np.float64).reshape(3, 3)
    if any(a.shape != (3,) for a in (crop, origin, spacing)):
        raise ValueError('crop, origin and spacing must each contain three values')
    if not all(np.isfinite(a).all() for a in (points, crop, origin, spacing, direction)):
        raise ValueError('non-finite coordinate metadata or predictions')
    if np.any(spacing <= 0) or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError('invalid spacing or direction matrix')
    return ((points + crop[::-1]) * spacing) @ direction.T + origin


def sample_froc(fps, sensitivity):
    """Interpolate at the seven targets, with endpoint extension as in np.interp."""
    fps = np.asarray(fps, dtype=float)
    sensitivity = np.asarray(sensitivity, dtype=float)
    if (fps.ndim != 1 or not len(fps) or fps.shape != sensitivity.shape
            or not np.isfinite(fps).all() or not np.isfinite(sensitivity).all()
            or np.any(np.diff(fps) < 0)):
        raise ValueError('FROC must be finite, nonempty and sorted by FP/scan')
    return np.interp(FROC_POINTS, fps, sensitivity).tolist()
