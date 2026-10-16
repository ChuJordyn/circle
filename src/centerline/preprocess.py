from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ROI = tuple[int, int, int, int]


@dataclass(frozen=True)
class PreprocessResult:
    image: np.ndarray
    roi_mask: np.ndarray
    valid_mask: np.ndarray
    background_level: float
    valid_threshold: float


def to_grayscale_float32(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3:
        if arr.shape[2] < 3:
            gray = np.mean(arr.astype(np.float32), axis=2)
        else:
            rgb = arr.astype(np.float32)
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError("image must be a 2-D grayscale or 3-D color array")
    return np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def make_roi_mask(shape: tuple[int, int], roi: ROI | None = None) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    if roi is None:
        mask[:] = True
        return mask

    x_min, y_min, x_max, y_max = roi
    x_min = max(0, min(width, int(x_min)))
    x_max = max(x_min, min(width, int(x_max)))
    y_min = max(0, min(height, int(y_min)))
    y_max = max(y_min, min(height, int(y_max)))
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def preprocess_laser_image(
    image: np.ndarray,
    roi: ROI | None = None,
    background_percentile: float = 15.0,
    valid_threshold_ratio: float = 0.05,
    normalize: bool = False,
) -> PreprocessResult:
    gray = to_grayscale_float32(image)
    roi_mask = make_roi_mask(gray.shape, roi)
    roi_values = gray[roi_mask]
    if roi_values.size == 0:
        raise ValueError("roi must contain at least one pixel")

    background_level = float(np.percentile(roi_values, background_percentile))
    processed = np.maximum(gray - background_level, 0.0).astype(np.float32)

    roi_signal = processed[roi_mask]
    signal_peak = float(np.max(roi_signal)) if roi_signal.size else 0.0
    if normalize and signal_peak > 1e-6:
        processed = (processed / signal_peak * 255.0).astype(np.float32)
        signal_peak = 255.0

    valid_threshold = max(0.0, float(valid_threshold_ratio)) * signal_peak
    valid_mask = roi_mask & (processed >= valid_threshold)
    return PreprocessResult(
        image=processed,
        roi_mask=roi_mask,
        valid_mask=valid_mask,
        background_level=background_level,
        valid_threshold=valid_threshold,
    )
