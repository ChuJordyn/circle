from __future__ import annotations

import numpy as np


def gray_centroid_on_profile(
    coordinates: np.ndarray,
    gray_values: np.ndarray,
    threshold_ratio: float,
) -> float | None:
    coordinates = np.asarray(coordinates, dtype=np.float32)
    gray_values = np.asarray(gray_values, dtype=np.float32)
    if coordinates.size == 0 or gray_values.size == 0 or coordinates.shape != gray_values.shape:
        return None

    peak = float(np.max(gray_values))
    if peak <= 0:
        return None

    local_min = float(np.percentile(gray_values, 15))
    cutoff = local_min + threshold_ratio * (peak - local_min)
    weights = gray_values.astype(np.float32) - cutoff
    weights[weights < 0] = 0

    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-6:
        return None

    return float(np.sum(coordinates * weights) / weight_sum)


def extract_vertical_peak_centerline(
    image: np.ndarray,
    y_min: int = 120,
    y_max: int = 390,
    roi_mask: np.ndarray | None = None,
    vectorized: bool = False,
) -> dict[str, np.ndarray]:
    if vectorized:
        return extract_vertical_peak_centerline_vectorized(
            image,
            y_min=y_min,
            y_max=y_max,
            roi_mask=roi_mask,
        )

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array")
    height, width = image.shape
    if roi_mask is not None and np.asarray(roi_mask).shape != image.shape:
        raise ValueError("roi_mask must have the same shape as image")

    y_start = max(0, int(y_min))
    y_stop = min(max(y_start, int(y_max)), height)
    estimated_y = np.full(width, np.nan, dtype=np.float32)
    valid = np.zeros(width, dtype=bool)

    for x in range(width):
        profile = image[y_start:y_stop, x]
        if roi_mask is not None:
            profile_mask = np.asarray(roi_mask[y_start:y_stop, x], dtype=bool)
            if not np.any(profile_mask):
                continue
            masked_profile = np.where(profile_mask, profile, -np.inf)
        else:
            masked_profile = profile
        if masked_profile.size == 0 or not np.any(np.isfinite(masked_profile)):
            continue
        peak_index = int(np.argmax(masked_profile))
        if not np.isfinite(masked_profile[peak_index]):
            continue
        estimated_y[x] = float(y_start + peak_index)
        valid[x] = True

    return {"estimated_y": estimated_y, "valid": valid}


def extract_vertical_peak_centerline_vectorized(
    image: np.ndarray,
    y_min: int = 120,
    y_max: int = 390,
    roi_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array")
    height, width = image.shape
    if roi_mask is not None and np.asarray(roi_mask).shape != image.shape:
        raise ValueError("roi_mask must have the same shape as image")

    y_start = max(0, int(y_min))
    y_stop = min(max(y_start, int(y_max)), height)
    profiles = image[y_start:y_stop, :].T
    if profiles.shape[1] == 0:
        return {"estimated_y": np.full(width, np.nan, dtype=np.float32), "valid": np.zeros(width, dtype=bool)}
    if roi_mask is not None:
        mask = np.asarray(roi_mask[y_start:y_stop, :], dtype=bool).T
        profiles = np.where(mask, profiles, -np.inf)

    peak_indices = np.argmax(profiles, axis=1)
    peak_values = np.max(profiles, axis=1)
    valid = np.isfinite(peak_values)
    estimated_y = np.full(width, np.nan, dtype=np.float32)
    estimated_y[valid] = y_start + peak_indices[valid].astype(np.float32)
    return {"estimated_y": estimated_y, "valid": valid}


def extract_vertical_gray_centroid(
    image: np.ndarray,
    y_min: int = 120,
    y_max: int = 390,
    threshold_ratio: float = 0.35,
    roi_mask: np.ndarray | None = None,
    vectorized: bool = False,
) -> dict[str, np.ndarray]:
    if vectorized:
        return extract_vertical_gray_centroid_vectorized(
            image,
            y_min=y_min,
            y_max=y_max,
            threshold_ratio=threshold_ratio,
            roi_mask=roi_mask,
        )

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array")
    height, width = image.shape
    if roi_mask is not None and np.asarray(roi_mask).shape != image.shape:
        raise ValueError("roi_mask must have the same shape as image")

    y_start = max(0, int(y_min))
    y_stop = min(max(y_start, int(y_max)), height)
    ys = np.arange(y_start, y_stop, dtype=np.float32)
    estimated_y = np.full(width, np.nan, dtype=np.float32)
    valid = np.zeros(width, dtype=bool)
    peaks = np.zeros(width, dtype=np.float32)
    weight_sums = np.zeros(width, dtype=np.float32)

    for x in range(width):
        profile = image[y_start:y_stop, x].astype(np.float32)
        if roi_mask is not None:
            profile_mask = np.asarray(roi_mask[y_start:y_stop, x], dtype=bool)
            if not np.any(profile_mask):
                continue
            profile = np.where(profile_mask, profile, 0.0)
        center = gray_centroid_on_profile(ys, profile, threshold_ratio)
        if center is not None:
            estimated_y[x] = center
            valid[x] = True
            peaks[x] = float(np.max(profile))
            cutoff = float(np.percentile(profile, 15)) + threshold_ratio * (
                peaks[x] - float(np.percentile(profile, 15))
            )
            weights = np.maximum(profile - cutoff, 0.0)
            weight_sums[x] = float(np.sum(weights))

    return {
        "estimated_y": estimated_y,
        "valid": valid,
        "peak": peaks,
        "weight_sum": weight_sums,
    }


def extract_vertical_gray_centroid_vectorized(
    image: np.ndarray,
    y_min: int = 120,
    y_max: int = 390,
    threshold_ratio: float = 0.35,
    roi_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array")
    height, width = image.shape
    if roi_mask is not None and np.asarray(roi_mask).shape != image.shape:
        raise ValueError("roi_mask must have the same shape as image")

    y_start = max(0, int(y_min))
    y_stop = min(max(y_start, int(y_max)), height)
    ys = np.arange(y_start, y_stop, dtype=np.float32)
    profiles = image[y_start:y_stop, :].T
    if profiles.shape[1] == 0:
        return {
            "estimated_y": np.full(width, np.nan, dtype=np.float32),
            "valid": np.zeros(width, dtype=bool),
            "peak": np.zeros(width, dtype=np.float32),
            "weight_sum": np.zeros(width, dtype=np.float32),
        }
    if roi_mask is not None:
        mask = np.asarray(roi_mask[y_start:y_stop, :], dtype=bool).T
        profiles = np.where(mask, profiles, 0.0)

    peaks = np.max(profiles, axis=1).astype(np.float32)
    profile_floors = np.percentile(profiles, 15, axis=1).astype(np.float32)
    cutoffs = profile_floors + threshold_ratio * (peaks - profile_floors)
    weights = np.maximum(profiles - cutoffs[:, np.newaxis], 0.0)
    weight_sums = np.sum(weights, axis=1, dtype=np.float32)
    numerator = np.sum(weights * ys[np.newaxis, :], axis=1, dtype=np.float32)

    estimated_y = np.full(width, np.nan, dtype=np.float32)
    valid = (peaks > 0.0) & (weight_sums > 1e-6)
    estimated_y[valid] = numerator[valid] / weight_sums[valid]
    return {
        "estimated_y": estimated_y,
        "valid": valid,
        "peak": peaks,
        "weight_sum": weight_sums,
    }
