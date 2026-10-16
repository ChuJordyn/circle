from __future__ import annotations

from dataclasses import dataclass
from math import acos, ceil, floor, pi

import numpy as np

from centerline.normal_guided_centroid import (
    bilinear_sample,
    extract_normal_guided_gray_centroid,
    errors_against_truth_normal_distance,
    summarize_errors,
)
from centerline.preprocess import ROI


@dataclass(frozen=True)
class StegerConfig:
    sigma: float = 1.0
    local_half_width: float = 4.0
    normal_step: float = 0.25
    max_offset: float = 0.75
    hessian_response_threshold: float = 0.5
    eigenvalue_ratio_threshold: float = 1.5
    max_normal_angle_deg: float = 35.0
    denominator_epsilon: float = 1e-3
    min_valid_fraction: float = 0.5
    adaptive_sigma: bool = False
    sigma_min: float = 0.75
    sigma_max: float = 2.0
    sigma_scale: float = 0.35
    sigma_quantization: float = 0.25
    ridge_sign: int = -1
    use_confidence_checks: bool = True


@dataclass(frozen=True)
class DerivativeFields:
    smoothed: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    ixx: np.ndarray
    iyy: np.ndarray
    ixy: np.ndarray
    origin_x: int = 0
    origin_y: int = 0

    def sample(self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        local_xs = np.asarray(xs, dtype=np.float32) - self.origin_x
        local_ys = np.asarray(ys, dtype=np.float32) - self.origin_y
        return bilinear_sample(field, local_xs, local_ys)


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def gaussian_derivative_kernels(
    sigma: float,
    radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    gaussian = np.exp(-0.5 * (xs / sigma) ** 2)
    gaussian /= np.sum(gaussian)

    first = -(xs / (sigma * sigma)) * gaussian
    first -= np.mean(first)
    second = ((xs * xs - sigma * sigma) / (sigma**4)) * gaussian
    second -= np.mean(second)
    return (
        gaussian.astype(np.float32),
        first.astype(np.float32),
        second.astype(np.float32),
    )


def convolve_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    radius = len(kernel) // 2
    if axis == 1:
        padded = np.pad(image, ((0, 0), (radius, radius)), mode="edge")
        output = np.empty_like(image, dtype=np.float32)
        for row in range(image.shape[0]):
            output[row] = np.convolve(padded[row], kernel, mode="valid")
        return output
    if axis == 0:
        padded = np.pad(image, ((radius, radius), (0, 0)), mode="edge")
        output = np.empty_like(image, dtype=np.float32)
        for col in range(image.shape[1]):
            output[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
        return output
    raise ValueError("axis must be 0 or 1")


def convolve_separable(
    image: np.ndarray,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray | None = None,
) -> np.ndarray:
    if kernel_y is None:
        kernel_y = kernel_x
    horizontal = convolve_axis(image, kernel_x, axis=1)
    return convolve_axis(horizontal, kernel_y, axis=0)


def build_derivative_fields(
    image: np.ndarray,
    sigma: float,
    roi: ROI | None = None,
    padding: int | None = None,
) -> DerivativeFields:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    height, width = image.shape
    if roi is None:
        x_min, y_min, x_max, y_max = 0, 0, width, height
    else:
        x_min, y_min, x_max, y_max = roi
        x_min = max(0, min(width, int(x_min)))
        y_min = max(0, min(height, int(y_min)))
        x_max = max(x_min + 1, min(width, int(x_max)))
        y_max = max(y_min + 1, min(height, int(y_max)))

    support = max(1, int(ceil(3.0 * sigma)) + 1)
    if padding is None:
        padding = support
    x0 = max(0, x_min - int(padding))
    y0 = max(0, y_min - int(padding))
    x1 = min(width, x_max + int(padding))
    y1 = min(height, y_max + int(padding))
    crop = image[y0:y1, x0:x1]

    gaussian, first, second = gaussian_derivative_kernels(sigma)
    return DerivativeFields(
        smoothed=convolve_separable(crop, gaussian),
        ix=convolve_separable(crop, first, gaussian),
        iy=convolve_separable(crop, gaussian, first),
        ixx=convolve_separable(crop, second, gaussian),
        iyy=convolve_separable(crop, gaussian, second),
        ixy=convolve_separable(crop, first, first),
        origin_x=x0,
        origin_y=y0,
    )


def gradient_and_hessian_at(
    fields: DerivativeFields,
    x: float,
    y: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    xs = np.asarray([x], dtype=np.float32)
    ys = np.asarray([y], dtype=np.float32)
    gradient = np.array(
        [
            fields.sample(fields.ix, xs, ys)[0],
            fields.sample(fields.iy, xs, ys)[0],
        ],
        dtype=np.float32,
    )
    hxx = float(fields.sample(fields.ixx, xs, ys)[0])
    hyy = float(fields.sample(fields.iyy, xs, ys)[0])
    hxy = float(fields.sample(fields.ixy, xs, ys)[0])
    hessian = np.array([[hxx, hxy], [hxy, hyy]], dtype=np.float32)
    intensity = float(fields.sample(fields.smoothed, xs, ys)[0])
    return gradient, hessian, intensity


def hessian_principal_direction(
    hessian: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    matrix = np.asarray(hessian, dtype=np.float32)
    if matrix.shape != (2, 2):
        raise ValueError("hessian must have shape (2, 2)")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    main_index = int(np.argmax(np.abs(eigenvalues)))
    other_index = 1 - main_index
    normal = eigenvectors[:, main_index].astype(np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return (
        float(eigenvalues[main_index]),
        float(eigenvalues[other_index]),
        normal,
    )


def _normal_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    dot = float(np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12))
    dot = min(1.0, max(-1.0, abs(dot)))
    return float(acos(dot) * 180.0 / pi)


def _point_in_image(shape: tuple[int, int], x: float, y: float, margin: float = 1.5) -> bool:
    height, width = shape
    return margin <= x < width - margin and margin <= y < height - margin


def _point_is_valid(valid_mask: np.ndarray | None, x: float, y: float) -> bool:
    if valid_mask is None:
        return True
    value = bilinear_sample(
        np.asarray(valid_mask, dtype=np.float32),
        np.asarray([x], dtype=np.float32),
        np.asarray([y], dtype=np.float32),
    )[0]
    return bool(value >= 0.5)


def _confidence(
    response: float,
    ratio: float,
    angle_deg: float,
    offset: float,
    config: StegerConfig,
) -> float:
    response_score = min(1.0, response / max(config.hessian_response_threshold * 4.0, 1e-6))
    ratio_score = min(1.0, ratio / max(config.eigenvalue_ratio_threshold * 2.0, 1e-6))
    angle_score = max(0.0, 1.0 - angle_deg / max(config.max_normal_angle_deg, 1e-6))
    offset_score = max(0.0, 1.0 - abs(offset) / max(config.max_offset, 1e-6))
    return float(response_score * ratio_score * angle_score * offset_score)


def steger_refine_at_point(
    fields: DerivativeFields,
    base_x: float,
    base_y: float,
    guided_normal_x: float,
    guided_normal_y: float,
    image_shape: tuple[int, int],
    roi_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    config: StegerConfig | None = None,
) -> dict[str, float | bool | str]:
    config = config or StegerConfig()
    result: dict[str, float | bool | str] = {
        "accepted": False,
        "x": float(base_x),
        "y": float(base_y),
        "offset": float("nan"),
        "normal_x": float("nan"),
        "normal_y": float("nan"),
        "lambda_main": float("nan"),
        "lambda_other": float("nan"),
        "eigen_ratio": float("nan"),
        "normal_angle_deg": float("nan"),
        "response": float("nan"),
        "denominator": float("nan"),
        "confidence": 0.0,
        "reason": "not_attempted",
    }

    if not _point_in_image(image_shape, base_x, base_y):
        result["reason"] = "center_outside_image"
        return result
    offsets = np.arange(
        -config.local_half_width,
        config.local_half_width + config.normal_step,
        config.normal_step,
        dtype=np.float32,
    )
    roi_xs = base_x + offsets * guided_normal_x
    roi_ys = base_y + offsets * guided_normal_y
    roi_inside = (
        (roi_xs >= 0)
        & (roi_xs < image_shape[1] - 1)
        & (roi_ys >= 0)
        & (roi_ys < image_shape[0] - 1)
    )
    if roi_mask is not None and np.any(roi_inside):
        roi_values = bilinear_sample(
            np.asarray(roi_mask, dtype=np.float32),
            roi_xs[roi_inside],
            roi_ys[roi_inside],
        )
        roi_fraction = float(np.mean(roi_values >= 0.5))
        if roi_fraction < config.min_valid_fraction:
            result["reason"] = "low_local_roi_fraction"
            return result
    if not _point_is_valid(valid_mask, base_x, base_y):
        result["reason"] = "center_outside_valid_region"
        return result

    gradient, hessian, _ = gradient_and_hessian_at(fields, base_x, base_y)
    lambda_main, lambda_other, hessian_normal = hessian_principal_direction(hessian)
    if np.dot(hessian_normal, np.asarray([guided_normal_x, guided_normal_y], dtype=np.float32)) < 0:
        hessian_normal = -hessian_normal

    response = abs(lambda_main)
    eigen_ratio = response / max(abs(lambda_other), 1e-6)
    angle_deg = _normal_angle_deg(
        np.asarray([guided_normal_x, guided_normal_y], dtype=np.float32),
        hessian_normal,
    )
    denominator = float(hessian_normal @ hessian @ hessian_normal)

    result.update(
        {
            "normal_x": float(hessian_normal[0]),
            "normal_y": float(hessian_normal[1]),
            "lambda_main": lambda_main,
            "lambda_other": lambda_other,
            "eigen_ratio": eigen_ratio,
            "normal_angle_deg": angle_deg,
            "response": response,
            "denominator": denominator,
        }
    )

    if config.use_confidence_checks:
        if response < config.hessian_response_threshold:
            result["reason"] = "weak_hessian_response"
            return result
        if config.ridge_sign < 0 and lambda_main >= -config.hessian_response_threshold:
            result["reason"] = "wrong_ridge_sign"
            return result
        if eigen_ratio < config.eigenvalue_ratio_threshold:
            result["reason"] = "weak_eigenvalue_separation"
            return result
        if angle_deg > config.max_normal_angle_deg:
            result["reason"] = "normal_angle_mismatch"
            return result
    if abs(denominator) <= config.denominator_epsilon:
        result["reason"] = "small_taylor_denominator"
        return result

    offset = float(-(hessian_normal @ gradient) / denominator)
    result["offset"] = offset
    result["confidence"] = _confidence(response, eigen_ratio, angle_deg, offset, config)
    if abs(offset) > config.max_offset:
        result["reason"] = "offset_too_large"
        return result

    refined_x = float(base_x + offset * hessian_normal[0])
    refined_y = float(base_y + offset * hessian_normal[1])
    if not _point_in_image(image_shape, refined_x, refined_y):
        result["reason"] = "refined_point_outside_image"
        return result
    if not _point_is_valid(valid_mask, refined_x, refined_y):
        result["reason"] = "refined_point_outside_valid_region"
        return result
    if not _point_is_valid(roi_mask, refined_x, refined_y):
        result["reason"] = "refined_point_outside_roi"
        return result
    if abs(refined_x - base_x) > config.local_half_width or abs(refined_y - base_y) > config.local_half_width:
        result["reason"] = "refined_point_outside_local_roi"
        return result

    result.update(
        {
            "accepted": True,
            "x": refined_x,
            "y": refined_y,
            "reason": "accepted",
        }
    )
    return result


def _make_derivative_roi(
    image_shape: tuple[int, int],
    base_x: np.ndarray,
    base_y: np.ndarray,
    config: StegerConfig,
    sigma: float,
) -> ROI | None:
    valid = np.isfinite(base_x) & np.isfinite(base_y)
    if not np.any(valid):
        return None
    height, width = image_shape
    margin = int(ceil(config.local_half_width + 3.0 * sigma + 2.0))
    x_min = max(0, int(floor(float(np.min(base_x[valid]))) - margin))
    y_min = max(0, int(floor(float(np.min(base_y[valid]))) - margin))
    x_max = min(width, int(ceil(float(np.max(base_x[valid]))) + margin + 1))
    y_max = min(height, int(ceil(float(np.max(base_y[valid]))) + margin + 1))
    return x_min, y_min, x_max, y_max


def _point_sigma(
    guided: dict[str, np.ndarray],
    index: int,
    config: StegerConfig,
) -> float:
    if not config.adaptive_sigma:
        return float(config.sigma)
    support_values = guided.get(
        "support_width",
        np.zeros(guided["estimated_x"].shape[0], dtype=np.float32),
    )
    support_width = float(support_values[index])
    sigma = config.sigma_scale * max(support_width, 1.0)
    sigma = min(config.sigma_max, max(config.sigma_min, sigma))
    step = max(config.sigma_quantization, 1e-3)
    return float(round(sigma / step) * step)


def _enforce_output_region(result, guided, roi_mask, valid_mask):
    x, y = result['estimated_x'], result['estimated_y']
    allowed = np.isfinite(x) & np.isfinite(y)
    if 'valid' in guided:
        allowed &= guided['valid']
    if 'search_range' in guided:
        lo, hi = guided['search_range']
        allowed &= (y >= lo) & (y < hi)
    for mask in (roi_mask, valid_mask):
        if mask is not None:
            allowed &= bilinear_sample(np.asarray(mask, dtype=np.float32), x, y) >= .5
    excluded = result['valid'] & ~allowed
    x[~allowed] = np.nan
    y[~allowed] = np.nan
    result['valid'] = allowed
    result['steger_refined'][~allowed] = False
    result['steger_fallback'][~allowed] = False
    result['confidence'][~allowed] = 0
    result['failure_reason'][excluded] = 'output_outside_allowed_region'
    return result


def extract_normal_guided_steger(
    image: np.ndarray,
    y_min: int = 120,
    y_max: int = 390,
    threshold_ratio: float = 0.35,
    sigma: float = 1.0,
    half_width: float = 3.0,
    roi_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    steger_config: StegerConfig | None = None,
    guided: dict[str, np.ndarray] | None = None,
    vectorized: bool = True,
) -> dict[str, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D grayscale array")
    if roi_mask is not None and np.asarray(roi_mask).shape != image.shape:
        raise ValueError("roi_mask must have the same shape as image")
    if valid_mask is not None and np.asarray(valid_mask).shape != image.shape:
        raise ValueError("valid_mask must have the same shape as image")

    config = steger_config or StegerConfig(
        sigma=sigma,
        local_half_width=half_width,
        normal_step=0.25,
    )
    if guided is None:
        guided = extract_normal_guided_gray_centroid(
            image=image,
            y_min=y_min,
            y_max=y_max,
            threshold_ratio=threshold_ratio,
            roi_mask=roi_mask,
            valid_mask=valid_mask,
        )

    if 'search_range' in guided:
        lo, hi = guided['search_range']
        rows = np.arange(image.shape[0])[:, None]
        search_mask = np.broadcast_to((rows >= lo) & (rows < hi), image.shape)
        roi_mask = search_mask if roi_mask is None else np.asarray(roi_mask, dtype=bool) & search_mask
    if vectorized:
        from centerline.steger_batch import extract_batch
        return _enforce_output_region(extract_batch(image, config, guided, roi_mask, valid_mask), guided, roi_mask, valid_mask)

    # Cache image-sized conversions once even on the scalar reference path.
    roi_mask = None if roi_mask is None else np.asarray(roi_mask, dtype=np.float32)
    valid_mask = None if valid_mask is None else np.asarray(valid_mask, dtype=np.float32)

    normal_x = guided["normal_x"]
    normal_y = guided["normal_y"]
    base_x = guided["estimated_x"]
    base_y = guided["estimated_y"]
    point_count = base_x.shape[0]
    refined_x = base_x.copy()
    refined_y = base_y.copy()
    steger_offsets = np.full(point_count, np.nan, dtype=np.float32)
    steger_refined = np.zeros(point_count, dtype=bool)
    steger_fallback = np.zeros(point_count, dtype=bool)
    refine_attempted = np.zeros(point_count, dtype=bool)
    hessian_normal_x = np.full(point_count, np.nan, dtype=np.float32)
    hessian_normal_y = np.full(point_count, np.nan, dtype=np.float32)
    lambda_main = np.full(point_count, np.nan, dtype=np.float32)
    lambda_other = np.full(point_count, np.nan, dtype=np.float32)
    eigen_ratio = np.full(point_count, np.nan, dtype=np.float32)
    normal_angle_deg = np.full(point_count, np.nan, dtype=np.float32)
    hessian_response = np.full(point_count, np.nan, dtype=np.float32)
    taylor_denominator = np.full(point_count, np.nan, dtype=np.float32)
    confidence = np.zeros(point_count, dtype=np.float32)
    sigma_used = np.full(point_count, np.nan, dtype=np.float32)
    failure_reason = np.full(point_count, "no_guided_center", dtype=object)

    finite_guided = np.isfinite(base_x) & np.isfinite(base_y)
    if np.any(finite_guided):
        sigma_values = np.array(
            [_point_sigma(guided, i, config) for i in range(point_count)],
            dtype=np.float32,
        )
        derivative_cache: dict[float, DerivativeFields] = {}
        roi_cache: dict[float, ROI | None] = {}

        for i in np.flatnonzero(finite_guided):
            current_sigma = float(sigma_values[i])
            sigma_used[i] = current_sigma
            if current_sigma not in derivative_cache:
                derivative_roi = _make_derivative_roi(
                    image.shape,
                    base_x,
                    base_y,
                    config,
                    current_sigma,
                )
                roi_cache[current_sigma] = derivative_roi
                derivative_cache[current_sigma] = build_derivative_fields(
                    image,
                    current_sigma,
                    roi=derivative_roi,
                    padding=int(ceil(3.0 * current_sigma)) + 2,
                )

            refine_attempted[i] = True
            point = steger_refine_at_point(
                fields=derivative_cache[current_sigma],
                base_x=float(base_x[i]),
                base_y=float(base_y[i]),
                guided_normal_x=float(normal_x[i]),
                guided_normal_y=float(normal_y[i]),
                image_shape=image.shape,
                roi_mask=roi_mask,
                valid_mask=valid_mask,
                config=config,
            )
            failure_reason[i] = point["reason"]
            hessian_normal_x[i] = float(point["normal_x"])
            hessian_normal_y[i] = float(point["normal_y"])
            lambda_main[i] = float(point["lambda_main"])
            lambda_other[i] = float(point["lambda_other"])
            eigen_ratio[i] = float(point["eigen_ratio"])
            normal_angle_deg[i] = float(point["normal_angle_deg"])
            hessian_response[i] = float(point["response"])
            taylor_denominator[i] = float(point["denominator"])
            confidence[i] = float(point["confidence"])
            if bool(point["accepted"]):
                refined_x[i] = float(point["x"])
                refined_y[i] = float(point["y"])
                steger_offsets[i] = float(point["offset"])
                steger_refined[i] = True
            else:
                steger_fallback[i] = True

    result = {
        "gray_guided_x": base_x,
        "gray_guided_y": base_y,
        "normal_x": normal_x,
        "normal_y": normal_y,
        "estimated_x": refined_x,
        "estimated_y": refined_y,
        "steger_offsets": steger_offsets,
        "steger_refined": steger_refined,
        "steger_fallback": steger_fallback,
        "refine_attempted": refine_attempted,
        "hessian_normal_x": hessian_normal_x,
        "hessian_normal_y": hessian_normal_y,
        "lambda_main": lambda_main,
        "lambda_other": lambda_other,
        "eigen_ratio": eigen_ratio,
        "normal_angle_deg": normal_angle_deg,
        "hessian_response": hessian_response,
        "taylor_denominator": taylor_denominator,
        "confidence": confidence,
        "sigma_used": sigma_used,
        "failure_reason": failure_reason,
        "valid": np.isfinite(refined_x) & np.isfinite(refined_y),
        "guided": guided,
    }
    return _enforce_output_region(result, guided, roi_mask, valid_mask)


__all__ = [
    "DerivativeFields",
    "StegerConfig",
    "build_derivative_fields",
    "convolve_separable",
    "errors_against_truth_normal_distance",
    "gaussian_derivative_kernels",
    "gaussian_kernel1d",
    "gradient_and_hessian_at",
    "hessian_principal_direction",
    "extract_normal_guided_steger",
    "steger_refine_at_point",
    "summarize_errors",
]
