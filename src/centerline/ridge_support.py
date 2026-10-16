"""Signal-based main-stripe support and boundary-aware local polynomial guidance.

No geometry ground truth is used by these estimators. The support mask describes
observed signal; interpolated guides are kept separate from measured centers.
"""
from __future__ import annotations

import numpy as np


def continuous_ridge_peaks(smooth: np.ndarray, max_step: int = 8, penalty: float = .5) -> np.ndarray:
    """Track up to five observed peaks with a second-order continuity cost.

    Curvature cost avoids the flattening bias of a first-order path length cost.
    max_step controls the extra penalty for large inter-column jumps. A path
    through a gap is only a guide; signal tests still decide measurement validity.
    """
    width, height = smooth.shape
    if width < 3 or height < 3:
        return np.argmax(smooth, axis=1)
    peaks_mask = np.zeros_like(smooth, dtype=bool)
    peaks_mask[:, 1:-1] = (smooth[:, 1:-1] >= smooth[:, :-2]) & (smooth[:, 1:-1] >= smooth[:, 2:]) & ((smooth[:, 1:-1] > smooth[:, :-2]) | (smooth[:, 1:-1] > smooth[:, 2:]))
    ranked = np.where(peaks_mask, smooth, -np.inf)
    count = min(5, height)
    candidates = np.argsort(ranked, axis=1)[:, -count:]
    scores = np.take_along_axis(ranked, candidates, axis=1)
    # Keep a defined fallback for entirely absent or monotonic profiles.
    fallback = np.argmax(smooth, axis=1)
    candidates = np.where(np.isfinite(scores), candidates, fallback[:, None])
    scores = np.take_along_axis(smooth, candidates, axis=1)
    scale = np.maximum(np.max(np.where(np.isfinite(smooth), smooth, 0), axis=1), 1e-6)
    likelihood = np.maximum(np.where(np.isfinite(scores), scores, 0), 0) / scale[:, None]
    unary = -np.log(np.maximum(likelihood, .01)).astype(np.float32)
    back = np.empty((width, count, count), dtype=np.int8)
    initial_motion = np.abs(candidates[1][None, :] - candidates[0][:, None])
    cost = unary[0][:, None] + unary[1][None, :] + .02 * initial_motion + .1 * np.maximum(initial_motion - max_step, 0)**2
    for column in range(2, width):
        motion = candidates[column][None, :] - candidates[column - 1][:, None]
        curvature = np.abs(candidates[column][None, None, :] - 2 * candidates[column - 1][None, :, None] + candidates[column - 2][:, None, None])
        alternatives = cost[:, :, None] + penalty * curvature
        back[column] = np.argmin(alternatives, axis=0)
        cost = np.min(alternatives, axis=0) + unary[column][None, :] + .02 * np.abs(motion) + .1 * np.maximum(np.abs(motion) - max_step, 0)**2
        cost -= np.min(cost)
    peaks = np.empty(width, dtype=np.int32)
    previous, current = np.unravel_index(np.argmin(cost), cost.shape)
    peaks[-1], peaks[-2] = candidates[-1, current], candidates[-2, previous]
    for column in range(width - 1, 1, -1):
        earlier = back[column, previous, current]
        peaks[column - 2] = candidates[column - 2, earlier]
        previous, current = earlier, previous
    return peaks


def localize_main_stripe(image, y_min, y_max, roi_mask, threshold_ratio, half_width, min_snr,
                         min_relative_signal=.35, max_weak_deviation=3.0,
                         track_max_step=8, track_penalty=.5):
    height, width = image.shape
    start = max(0, min(height, int(y_min)))
    stop = max(start, min(height, int(y_max)))
    centers = np.full(width, np.nan, dtype=np.float32)
    valid = np.zeros(width, dtype=bool)
    snr = np.zeros(width, dtype=np.float32)
    if stop == start:
        return {"estimated_y": centers, "valid": valid, "signal_snr": snr}
    profiles = np.asarray(image[start:stop], dtype=np.float32).T
    mask = np.ones_like(profiles, dtype=bool) if roi_mask is None else np.asarray(roi_mask[start:stop], dtype=bool).T
    count = mask.sum(axis=1)
    masked = np.where(mask, profiles, np.nan)
    floors = np.zeros(width, dtype=np.float32)
    active = count > 0
    floors[active] = np.nanpercentile(masked[active], 30, axis=1)
    signal = np.where(mask, np.maximum(profiles - floors[:, None], 0), 0)
    padded = np.pad(signal, ((0, 0), (2, 2)), mode="edge")
    smooth = sum(weight * padded[:, k:k + profiles.shape[1]] for k, weight in enumerate([1, 4, 6, 4, 1])) / 16
    if roi_mask is not None:
        padded_mask = np.pad(mask.astype(np.float32), ((0,0),(2,2)), mode='edge')
        support = sum(weight * padded_mask[:, k:k + profiles.shape[1]] for k,weight in enumerate([1,4,6,4,1])) / 16
        smooth /= np.maximum(support, 1e-6)
    smooth = np.where(mask, smooth, -np.inf)
    track_profiles = smooth
    strengths = np.max(np.where(np.isfinite(smooth), smooth, 0), axis=1)
    reference = float(np.median(strengths[active])) if np.any(active) else 0
    trusted_columns = active & (strengths >= reference * min_relative_signal) & (strengths > 0)
    trusted_x = np.flatnonzero(trusted_columns)
    if track_max_step > 0 and trusted_x.size > 1 and np.any(~trusted_columns):
        # Bridge dim/absent columns for path guidance only, using observed
        # neighboring profiles. Actual signal and masks still govern validity.
        x = np.arange(width)
        pos = np.searchsorted(trusted_x, x)
        left = trusted_x[np.clip(pos - 1, 0, trusted_x.size - 1)]
        right = trusted_x[np.clip(pos, 0, trusted_x.size - 1)]
        alpha = (x - left) / np.maximum(right - left, 1)
        track_profiles = smooth.copy()
        filled = (1 - alpha[:, None]) * np.where(np.isfinite(smooth[left]), smooth[left], 0) + alpha[:, None] * np.where(np.isfinite(smooth[right]), smooth[right], 0)
        track_profiles[~trusted_columns] = filled[~trusted_columns]
    peaks = continuous_ridge_peaks(track_profiles, track_max_step, track_penalty) if track_max_step > 0 else np.argmax(smooth, axis=1)
    amplitude = smooth[np.arange(width), peaks]
    # Vertical first differences estimate sensor noise without counting smooth background gradients.
    noise = np.ones(width, dtype=np.float32) * 1e-6
    if profiles.shape[1] > 1:
        diff = np.diff(profiles, axis=1)
        pair_mask = mask[:, 1:] & mask[:, :-1]
        for_noise = np.where(pair_mask, np.abs(diff), np.nan)
        has_pairs = pair_mask.any(axis=1)
        noise[has_pairs] = np.nanmedian(for_noise[has_pairs], axis=1) / .9538725524
    snr[active] = amplitude[active] / np.maximum(noise[active], 1e-6)
    offsets = np.arange(-half_width, half_width + 1)
    indices = peaks[:, None] + offsets
    inside = (indices >= 0) & (indices < profiles.shape[1])
    indices = np.clip(indices, 0, profiles.shape[1] - 1)
    rows = np.arange(width)[:, None]
    local_smooth = smooth[rows, indices]
    above = inside & mask[rows, indices] & (local_smooth > threshold_ratio * amplitude[:, None])
    # Keep only the contiguous support containing the dominant peak.
    connected = above.copy()
    connected[:, :half_width + 1] &= np.minimum.accumulate(above[:, :half_width + 1][:, ::-1], axis=1)[:, ::-1]
    connected[:, half_width:] &= np.minimum.accumulate(above[:, half_width:], axis=1)
    weights = np.where(connected, np.maximum(signal[rows, indices] - threshold_ratio * amplitude[:, None], 0), 0)
    sums = weights.sum(axis=1)
    # A peak clipped by the search boundary cannot distinguish a ridge from a
    # monotonic background ramp; require an observed peak with both flanks.
    interior_peak = (peaks > 0) & (peaks < profiles.shape[1] - 1)
    rows = np.arange(width)
    left = np.maximum(peaks - 1, 0)
    right = np.minimum(peaks + 1, profiles.shape[1] - 1)
    # Ridge evidence must exist in the original observed profile. Interpolated
    # tracking profiles and ROI boundaries cannot supply either flank.
    interior_peak &= mask[rows,left] & mask[rows,right]
    interior_peak &= (amplitude >= smooth[rows,left]) & (amplitude >= smooth[rows,right])
    interior_peak &= (amplitude > smooth[rows,left]) | (amplitude > smooth[rows,right])
    valid = active & interior_peak & (amplitude > 0) & (snr >= min_snr) & (sums > 1e-6)
    centers[valid] = ((weights * (indices + start)).sum(axis=1)[valid] / sums[valid]).astype(np.float32)
    if np.any(valid):
        reference = float(np.median(amplitude[valid]))
        trusted = valid & (amplitude >= reference * min_relative_signal)
        if np.count_nonzero(trusted) >= 3:
            x = np.arange(width)
            predicted = np.interp(x, x[trusted], centers[trusted])
            # A dim but geometrically consistent stripe remains measurable. A weak,
            # displaced peak is not allowed to silently replace the main stripe.
            inconsistent = valid & ~trusted & (np.abs(centers - predicted) > max_weak_deviation)
            valid[inconsistent] = False
            centers[inconsistent] = np.nan
    return {"estimated_y": centers, "valid": valid, "signal_snr": snr}


def polynomial_guide(center_y, window_size=9):
    """Quadratic moving least squares with real, one-sided coordinates at edges."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("guide window must be odd and at least 3")
    values = np.asarray(center_y, dtype=np.float64)
    width = values.size
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.full(width, np.nan, dtype=np.float32)
    x = np.arange(width)
    filled = np.interp(x, x[valid], values[valid])
    offsets = np.arange(-window_size // 2 + 1, window_size // 2 + 1)
    indices = x[:, None] + offsets
    inside = (indices >= 0) & (indices < width)
    clipped = np.clip(indices, 0, width - 1)
    weights = inside * valid[clipped] * np.exp(-.5 * (offsets / max(window_size / 3, 1)) ** 2)
    design = np.stack([np.ones_like(offsets), offsets, offsets ** 2], axis=1).astype(np.float64)
    lhs = np.einsum("wi,ij,ik->wjk", weights, design, design)
    rhs = np.einsum("wi,ij,wi->wj", weights, design, filled[clipped])
    fitted = np.linalg.solve(lhs + np.eye(3)[None] * 1e-8, rhs[..., None])[..., 0]
    enough = np.count_nonzero(weights, axis=1) >= 3
    result = np.where(enough, fitted[:, 0], filled)
    return result.astype(np.float32)
