from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from centerline.gray_centroid import gray_centroid_on_profile, extract_vertical_gray_centroid
from centerline.ridge_support import localize_main_stripe, polynomial_guide


@dataclass(frozen=True)
class NormalGuidedConfig:
    y_min: int = 120
    y_max: int = 390
    threshold_ratio: float = 0.35
    normal_half_width: float = 14.0
    normal_step: float = 0.25
    median_window: int = 9
    mean_window: int = 13
    outlier_sigma: float = 3.0
    min_sample_count: int = 5
    use_robust_support: bool = True
    min_signal_snr: float = 5.0
    coarse_half_width: int = 14
    min_relative_signal: float = 0.35
    max_weak_deviation: float = 3.0
    track_max_step: int = 8
    track_penalty: float = .5


def bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = image.shape
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    finite = np.isfinite(xs) & np.isfinite(ys)
    safe_xs = np.where(finite, xs, 0.0)
    safe_ys = np.where(finite, ys, 0.0)
    x0 = np.floor(safe_xs).astype(np.int32)
    y0 = np.floor(safe_ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    valid = finite & (xs >= 0) & (ys >= 0) & (xs <= w - 1) & (ys <= h - 1)
    values = np.zeros(xs.shape, dtype=np.float32)
    if not np.any(valid):
        return values

    xv = xs[valid]
    yv = ys[valid]
    x0v = x0[valid]
    y0v = y0[valid]
    x1v = x1[valid]
    y1v = y1[valid]

    dx = xv - x0v
    dy = yv - y0v
    top = image[y0v, x0v] * (1.0 - dx) + image[y0v, x1v] * dx
    bottom = image[y1v, x0v] * (1.0 - dx) + image[y1v, x1v] * dx
    values[valid] = top * (1.0 - dy) + bottom * dy
    return values


def fill_nan_by_interp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    xs = np.arange(values.size, dtype=np.float32)
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.zeros_like(values)
    return np.interp(xs, xs[valid], values[valid]).astype(np.float32)


def median_filter_1d(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    radius = window_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    filtered = np.empty_like(values, dtype=np.float32)
    for i in range(values.size):
        filtered[i] = np.median(padded[i : i + window_size])
    return filtered


def moving_average_1d(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    radius = window_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def robust_smooth_centerline(
    center_y: np.ndarray,
    median_window: int = 9,
    mean_window: int = 9,
    outlier_sigma: float = 3.0,
) -> np.ndarray:
    filled = fill_nan_by_interp(center_y)
    median_y = median_filter_1d(filled, median_window)
    residual = filled - median_y
    mad = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826
    threshold = max(5.0, outlier_sigma * mad)
    repaired = filled.copy()
    outliers = np.abs(residual) > threshold
    repaired[outliers] = median_y[outliers]
    return moving_average_1d(repaired, mean_window)


def estimate_tangent_normal(center_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    slope = np.gradient(center_y.astype(np.float32)) if center_y.size > 1 else np.zeros_like(center_y)
    denom = np.sqrt(1.0 + slope * slope)
    tangent_x = 1.0 / denom
    tangent_y = slope / denom
    normal_x = -slope / denom
    normal_y = 1.0 / denom
    return tangent_x, tangent_y, normal_x, normal_y


def extract_normal_guided_gray_centroid_vectorized(
    image: np.ndarray,
    y_min: int = 120,
    y_max: int = 390,
    threshold_ratio: float = 0.35,
    normal_half_width: float = 14.0,
    normal_step: float = 0.25,
    roi_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    config: NormalGuidedConfig | None = None,
) -> dict[str, np.ndarray]:
    if config is not None:
        y_min = config.y_min
        y_max = config.y_max
        threshold_ratio = config.threshold_ratio
        normal_half_width = config.normal_half_width
        normal_step = config.normal_step
        median_window = config.median_window
        mean_window = config.mean_window
        outlier_sigma = config.outlier_sigma
        min_sample_count = config.min_sample_count
    else:
        median_window = 9
        mean_window = 9
        outlier_sigma = 3.0
        min_sample_count = 5

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array")
    if roi_mask is not None and np.asarray(roi_mask).shape != image.shape:
        raise ValueError("roi_mask must have the same shape as image")
    if valid_mask is not None and np.asarray(valid_mask).shape != image.shape:
        raise ValueError("valid_mask must have the same shape as image")

    robust_config = config or NormalGuidedConfig()
    if robust_config.use_robust_support:
        coarse = localize_main_stripe(
            image, y_min, y_max, roi_mask, threshold_ratio,
            robust_config.coarse_half_width, robust_config.min_signal_snr,
            robust_config.min_relative_signal, robust_config.max_weak_deviation,
            robust_config.track_max_step, robust_config.track_penalty,
        )
    else:
        coarse = extract_vertical_gray_centroid(
            image, y_min=y_min, y_max=y_max, threshold_ratio=threshold_ratio,
            roi_mask=roi_mask, vectorized=True,
        )
    coarse_y = coarse["estimated_y"]
    coarse_valid = coarse["valid"]
    if np.any(coarse_valid):
        if robust_config.use_robust_support:
            guide_y = polynomial_guide(coarse_y, mean_window)
        else:
            guide_y = robust_smooth_centerline(
                coarse_y, median_window=median_window, mean_window=mean_window,
                outlier_sigma=outlier_sigma,
            )
    else:
        guide_y = np.full_like(coarse_y, np.nan, dtype=np.float32)
    _, _, normal_x, normal_y = estimate_tangent_normal(guide_y)

    height, width = image.shape
    offsets = np.arange(
        -normal_half_width,
        normal_half_width + normal_step,
        normal_step,
        dtype=np.float32,
    )
    base_x = np.arange(width, dtype=np.float32)
    guide_valid = np.isfinite(guide_y)
    safe_guide_y = np.nan_to_num(guide_y, nan=0.0)
    safe_normal_x = np.nan_to_num(normal_x, nan=0.0)
    safe_normal_y = np.nan_to_num(normal_y, nan=1.0)

    sample_x = base_x[:, np.newaxis] + safe_normal_x[:, np.newaxis] * offsets[np.newaxis, :]
    sample_y = safe_guide_y[:, np.newaxis] + safe_normal_y[:, np.newaxis] * offsets[np.newaxis, :]
    inside = (
        guide_valid[:, np.newaxis]
        & (sample_x >= 0.0)
        & (sample_x <= width - 1)
        & (sample_y >= 0.0)
        & (sample_y <= height - 1)
        & (sample_y >= y_min)
        & (sample_y < y_max)
    )
    if roi_mask is not None:
        roi_values = bilinear_sample(
            np.asarray(roi_mask, dtype=np.float32),
            sample_x,
            sample_y,
        )
        inside &= roi_values >= 0.5

    profiles = bilinear_sample(image, sample_x, sample_y)
    sample_counts = np.count_nonzero(inside, axis=1).astype(np.int32)
    has_samples = sample_counts > 0
    profile_for_peak = np.where(inside, profiles, -np.inf)
    sample_peaks = np.max(profile_for_peak, axis=1).astype(np.float32)
    sample_peaks[~has_samples] = 0.0

    masked_profiles = np.where(inside, profiles, np.nan)
    profile_floors = np.zeros(width, dtype=np.float32)
    if np.any(has_samples):
        profile_floors[has_samples] = np.nanpercentile(
            masked_profiles[has_samples],
            15,
            axis=1,
        ).astype(np.float32)
    cutoffs = profile_floors + threshold_ratio * (sample_peaks - profile_floors)
    weights = np.where(
        inside,
        np.maximum(profiles - cutoffs[:, np.newaxis], 0.0),
        0.0,
    )
    weight_sums = np.sum(weights, axis=1, dtype=np.float32)
    offset_numerators = np.sum(
        weights * offsets[np.newaxis, :],
        axis=1,
        dtype=np.float32,
    )
    has_weight = weight_sums > 1e-6
    normal_offsets = np.full(width, np.nan, dtype=np.float32)
    normal_offsets[has_weight] = offset_numerators[has_weight] / weight_sums[has_weight]

    candidate_x = base_x + normal_offsets * safe_normal_x
    candidate_y = safe_guide_y + normal_offsets * safe_normal_y
    refined = guide_valid & (sample_counts >= min_sample_count) & has_weight
    refined &= (candidate_y >= y_min) & (candidate_y < y_max)
    if roi_mask is not None:
        refined &= bilinear_sample(np.asarray(roi_mask, dtype=np.float32), candidate_x, candidate_y) >= .5
    if valid_mask is not None:
        safe_candidate_x = np.nan_to_num(candidate_x, nan=0.0)
        safe_candidate_y = np.nan_to_num(candidate_y, nan=0.0)
        candidate_valid = bilinear_sample(
            np.asarray(valid_mask, dtype=np.float32),
            safe_candidate_x,
            safe_candidate_y,
        ) >= 0.5
        refined &= candidate_valid

    estimated_x = np.full(width, np.nan, dtype=np.float32)
    estimated_y = np.full(width, np.nan, dtype=np.float32)
    estimated_x[refined] = candidate_x[refined]
    estimated_y[refined] = candidate_y[refined]

    valid = refined.copy()
    if robust_config.use_robust_support:
        # An interpolated guide is never evidence that a missing stripe was measured.
        refined &= coarse_valid
        first_inside = np.argmax(inside, axis=1)
        last_inside = inside.shape[1] - 1 - np.argmax(inside[:, ::-1], axis=1)
        rows = np.arange(width)
        truncated = has_samples & (
            (profiles[rows, first_inside] >= cutoffs)
            | (profiles[rows, last_inside] >= cutoffs)
        )
        fallback = coarse_valid & (truncated | ~refined)
        if roi_mask is not None:
            fallback &= bilinear_sample(np.asarray(roi_mask, dtype=np.float32), base_x, coarse_y) >= .5
        if valid_mask is not None:
            coarse_allowed = bilinear_sample(
                np.asarray(valid_mask, dtype=np.float32), base_x,
                np.nan_to_num(coarse_y, nan=0.0),
            ) >= .5
            fallback &= coarse_allowed
        estimated_x[fallback] = base_x[fallback]
        estimated_y[fallback] = coarse_y[fallback]
        refined[fallback] = False
        valid = refined | fallback
        estimated_x[~valid] = np.nan
        estimated_y[~valid] = np.nan

    support_width = np.count_nonzero(
        inside & (profiles >= cutoffs[:, np.newaxis]),
        axis=1,
    ).astype(np.float32) * normal_step
    return {
        "coarse_y": coarse_y,
        "coarse_valid": coarse_valid,
        "guide_y": guide_y,
        "normal_x": normal_x,
        "normal_y": normal_y,
        "estimated_x": estimated_x,
        "estimated_y": estimated_y,
        "normal_offsets": normal_offsets,
        "refined": refined,
        "valid": valid,
        "coarse_signal_snr": coarse.get("signal_snr", np.full(width, np.nan, dtype=np.float32)),
        "search_range": np.array([y_min, y_max], dtype=np.float32),
        "sample_count": sample_counts,
        "sample_peak": sample_peaks,
        "support_width": support_width,
    }


def extract_normal_guided_gray_centroid(
    image: np.ndarray,
    y_min: int = 120,
    y_max: int = 390,
    threshold_ratio: float = 0.35,
    normal_half_width: float = 14.0,
    normal_step: float = 0.25,
    roi_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    config: NormalGuidedConfig | None = None,
    vectorized: bool = False,
) -> dict[str, np.ndarray]:
    if vectorized or (config or NormalGuidedConfig()).use_robust_support:
        return extract_normal_guided_gray_centroid_vectorized(
            image=image,
            y_min=y_min,
            y_max=y_max,
            threshold_ratio=threshold_ratio,
            normal_half_width=normal_half_width,
            normal_step=normal_step,
            roi_mask=roi_mask,
            valid_mask=valid_mask,
            config=config,
        )

    if config is not None:
        y_min = config.y_min
        y_max = config.y_max
        threshold_ratio = config.threshold_ratio
        normal_half_width = config.normal_half_width
        normal_step = config.normal_step
        median_window = config.median_window
        mean_window = config.mean_window
        outlier_sigma = config.outlier_sigma
        min_sample_count = config.min_sample_count
    else:
        median_window = 9
        mean_window = 9
        outlier_sigma = 3.0
        min_sample_count = 5

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array")
    if roi_mask is not None and np.asarray(roi_mask).shape != image.shape:
        raise ValueError("roi_mask must have the same shape as image")
    if valid_mask is not None and np.asarray(valid_mask).shape != image.shape:
        raise ValueError("valid_mask must have the same shape as image")

    coarse = extract_vertical_gray_centroid(
        image,
        y_min,
        y_max,
        threshold_ratio,
        roi_mask=roi_mask,
    )
    coarse_y = coarse["estimated_y"]
    coarse_valid = coarse["valid"]
    if np.any(coarse_valid):
        guide_y = robust_smooth_centerline(
            coarse_y,
            median_window=median_window,
            mean_window=mean_window,
            outlier_sigma=outlier_sigma,
        )
    else:
        guide_y = np.full_like(coarse_y, np.nan, dtype=np.float32)
    _, _, normal_x, normal_y = estimate_tangent_normal(guide_y)

    h, w = image.shape
    offsets = np.arange(-normal_half_width, normal_half_width + normal_step, normal_step, dtype=np.float32)
    estimated_x = np.full(w, np.nan, dtype=np.float32)
    estimated_y = np.full(w, np.nan, dtype=np.float32)
    normal_offsets = np.full(w, np.nan, dtype=np.float32)
    refined = np.zeros(w, dtype=bool)
    sample_counts = np.zeros(w, dtype=np.int32)
    sample_peaks = np.zeros(w, dtype=np.float32)
    support_width = np.zeros(w, dtype=np.float32)

    for x in range(w):
        base_x = float(x)
        base_y = float(guide_y[x])
        if not np.isfinite(base_y):
            continue
        nx = float(normal_x[x])
        ny = float(normal_y[x])

        sample_x = base_x + offsets * nx
        sample_y = base_y + offsets * ny
        inside = (sample_x >= 0) & (sample_x <= w - 1) & (sample_y >= 0) & (sample_y <= h - 1) & (sample_y >= y_min) & (sample_y < y_max)
        if roi_mask is not None:
            roi_values = bilinear_sample(
                np.asarray(roi_mask, dtype=np.float32),
                sample_x,
                sample_y,
            )
            inside &= roi_values >= 0.5
        sample_counts[x] = int(np.count_nonzero(inside))
        if sample_counts[x] < min_sample_count:
            continue

        valid_offsets = offsets[inside]
        profile = bilinear_sample(image, sample_x[inside], sample_y[inside])
        sample_peaks[x] = float(np.max(profile)) if profile.size else 0.0
        if profile.size:
            profile_floor = float(np.percentile(profile, 15))
            profile_cutoff = profile_floor + threshold_ratio * (sample_peaks[x] - profile_floor)
            support_width[x] = float(np.count_nonzero(profile >= profile_cutoff) * normal_step)
        offset_centroid = gray_centroid_on_profile(valid_offsets, profile, threshold_ratio)
        if offset_centroid is None:
            continue

        candidate_x = base_x + offset_centroid * nx
        candidate_y = base_y + offset_centroid * ny
        if not y_min <= candidate_y < y_max:
            continue
        if roi_mask is not None and bilinear_sample(np.asarray(roi_mask, dtype=np.float32), np.array([candidate_x]), np.array([candidate_y]))[0] < .5:
            continue
        if valid_mask is not None:
            candidate_valid = bilinear_sample(
                np.asarray(valid_mask, dtype=np.float32),
                np.asarray([candidate_x], dtype=np.float32),
                np.asarray([candidate_y], dtype=np.float32),
            )[0] >= 0.5
            if not candidate_valid:
                continue

        estimated_x[x] = candidate_x
        estimated_y[x] = candidate_y
        normal_offsets[x] = offset_centroid
        refined[x] = True

    return {
        "coarse_y": coarse_y,
        "coarse_valid": coarse_valid,
        "guide_y": guide_y,
        "normal_x": normal_x,
        "normal_y": normal_y,
        "estimated_x": estimated_x,
        "estimated_y": estimated_y,
        "normal_offsets": normal_offsets,
        "refined": refined,
        "valid": refined.copy(),
        "sample_count": sample_counts,
        "sample_peak": sample_peaks,
        "support_width": support_width,
        "search_range": np.array([y_min,y_max],dtype=np.float32),
    }


def errors_against_truth_normal_distance(
    estimated_x: np.ndarray,
    estimated_y: np.ndarray,
    truth_y: np.ndarray,
) -> np.ndarray:
    truth_x = np.arange(truth_y.size, dtype=np.float32)
    truth_slope = np.gradient(truth_y.astype(np.float32))
    valid = np.isfinite(estimated_x) & np.isfinite(estimated_y)
    errors = np.full_like(estimated_y, np.nan, dtype=np.float32)
    truth_at_x = np.interp(estimated_x[valid], truth_x, truth_y)
    slope_at_x = np.interp(estimated_x[valid], truth_x, truth_slope)
    vertical_error = estimated_y[valid] - truth_at_x
    errors[valid] = vertical_error / np.sqrt(1.0 + slope_at_x * slope_at_x)
    return errors


def summarize_errors(errors: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(errors)
    e = errors[valid]
    if e.size == 0:
        return {
            "valid_count": 0,
            "mean_abs_px": float("nan"),
            "rmse_px": float("nan"),
            "max_abs_px": float("nan"),
        }
    return {
        "valid_count": int(e.size),
        "mean_abs_px": float(np.mean(np.abs(e))),
        "rmse_px": float(np.sqrt(np.mean(e * e))),
        "max_abs_px": float(np.max(np.abs(e))),
    }
