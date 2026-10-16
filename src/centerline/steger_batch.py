"""Batch Hessian/Taylor evaluation, grouped by locally estimated stripe scale."""
from __future__ import annotations

import numpy as np

from centerline.normal_guided_centroid import bilinear_sample
from centerline.normal_guided_steger import build_derivative_fields, _make_derivative_roi, _point_sigma


def extract_batch(image, config, guided, roi_mask=None, valid_mask=None):
    base_x = np.asarray(guided["estimated_x"], dtype=np.float32)
    base_y = np.asarray(guided["estimated_y"], dtype=np.float32)
    nx = guided["normal_x"]
    ny = guided["normal_y"]
    count = base_x.size
    roi_float = None if roi_mask is None else np.asarray(roi_mask, dtype=np.float32)
    valid_float = None if valid_mask is None else np.asarray(valid_mask, dtype=np.float32)
    output = {
        "gray_guided_x": base_x, "gray_guided_y": base_y,
        "normal_x": nx, "normal_y": ny,
        "estimated_x": base_x.copy(), "estimated_y": base_y.copy(),
        "guided": guided,
    }
    for key in ["steger_offsets", "hessian_normal_x", "hessian_normal_y", "lambda_main", "lambda_other", "eigen_ratio", "normal_angle_deg", "hessian_response", "taylor_denominator", "sigma_used"]:
        output[key] = np.full(count, np.nan, dtype=np.float32)
    for key in ["steger_refined", "steger_fallback", "refine_attempted"]:
        output[key] = np.zeros(count, dtype=bool)
    output["confidence"] = np.zeros(count, dtype=np.float32)
    output["failure_reason"] = np.full(count, "no_guided_center", dtype=object)
    finite = np.isfinite(base_x) & np.isfinite(base_y)
    scales = np.array([_point_sigma(guided, i, config) for i in range(count)], dtype=np.float32)
    height, width = image.shape

    for sigma in np.unique(scales[finite]):
        indices = np.flatnonzero(finite & (scales == sigma))
        x = base_x[indices]
        y = base_y[indices]
        normal = np.stack([nx[indices], ny[indices]], axis=1)
        reasons = np.full(indices.size, "accepted", dtype=object)

        def reject(condition, reason):
            reasons[(reasons == "accepted") & condition] = reason

        reject((x < 1.5) | (x >= width - 1.5) | (y < 1.5) | (y >= height - 1.5), "center_outside_image")
        offsets = np.arange(-config.local_half_width, config.local_half_width + config.normal_step, config.normal_step, dtype=np.float32)
        sx = x[:, None] + normal[:, 0, None] * offsets
        sy = y[:, None] + normal[:, 1, None] * offsets
        inside = (sx >= 0) & (sx < width - 1) & (sy >= 0) & (sy < height - 1)
        if roi_float is not None:
            roi_values = bilinear_sample(roi_float, sx, sy)
            fraction = np.sum(inside & (roi_values >= .5), axis=1) / np.maximum(inside.sum(axis=1), 1)
            reject(fraction < config.min_valid_fraction, "low_local_roi_fraction")
        if valid_float is not None:
            reject(bilinear_sample(valid_float, x, y) < .5, "center_outside_valid_region")

        roi = _make_derivative_roi(image.shape, base_x, base_y, config, float(sigma))
        fields = build_derivative_fields(image, float(sigma), roi=roi)
        gradient = np.stack([fields.sample(fields.ix, x, y), fields.sample(fields.iy, x, y)], axis=1)
        hessian = np.empty((indices.size, 2, 2), dtype=np.float32)
        hessian[:, 0, 0] = fields.sample(fields.ixx, x, y)
        hessian[:, 1, 1] = fields.sample(fields.iyy, x, y)
        hessian[:, 0, 1] = hessian[:, 1, 0] = fields.sample(fields.ixy, x, y)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        main_index = np.argmax(np.abs(eigenvalues), axis=1)
        rows = np.arange(indices.size)
        main = eigenvalues[rows, main_index]
        other = eigenvalues[rows, 1 - main_index]
        principal = eigenvectors[rows, :, main_index]
        dots = np.sum(principal * normal, axis=1)
        principal[dots < 0] *= -1
        response = np.abs(main)
        ratio = response / np.maximum(np.abs(other), 1e-6)
        normal_length = np.linalg.norm(normal, axis=1)
        cosine = np.abs(np.sum(principal * normal, axis=1)) / np.maximum(normal_length, 1e-12)
        angles = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
        denominator = np.einsum("wi,wij,wj->w", principal, hessian, principal)
        if config.use_confidence_checks:
            reject(response < config.hessian_response_threshold, "weak_hessian_response")
            if config.ridge_sign < 0:
                reject(main >= -config.hessian_response_threshold, "wrong_ridge_sign")
            reject(ratio < config.eigenvalue_ratio_threshold, "weak_eigenvalue_separation")
            reject(angles > config.max_normal_angle_deg, "normal_angle_mismatch")
        reject(np.abs(denominator) <= config.denominator_epsilon, "small_taylor_denominator")
        offset = np.full(indices.size, np.nan, dtype=np.float32)
        nonzero = np.abs(denominator) > config.denominator_epsilon
        offset[nonzero] = -np.sum(principal[nonzero] * gradient[nonzero], axis=1) / denominator[nonzero]
        scored = reasons == "accepted"
        confidence = (
            np.minimum(1, response / max(config.hessian_response_threshold * 4, 1e-6))
            * np.minimum(1, ratio / max(config.eigenvalue_ratio_threshold * 2, 1e-6))
            * np.maximum(0, 1 - angles / max(config.max_normal_angle_deg, 1e-6))
            * np.maximum(0, 1 - np.abs(offset) / max(config.max_offset, 1e-6))
        )
        output["confidence"][indices[scored]] = confidence[scored]
        reject(np.abs(offset) > config.max_offset, "offset_too_large")
        cx = x + offset * principal[:, 0]
        cy = y + offset * principal[:, 1]
        reject((cx < 1.5) | (cx >= width - 1.5) | (cy < 1.5) | (cy >= height - 1.5), "refined_point_outside_image")
        if valid_float is not None:
            reject(bilinear_sample(valid_float, cx, cy) < .5, "refined_point_outside_valid_region")
        if roi_float is not None:
            reject(bilinear_sample(roi_float, cx, cy) < .5, "refined_point_outside_roi")
        if 'search_range' in guided:
            lo, hi = guided['search_range']
            reject((cy < lo) | (cy >= hi), 'refined_point_outside_search_range')
        reject((np.abs(cx - x) > config.local_half_width) | (np.abs(cy - y) > config.local_half_width), "refined_point_outside_local_roi")
        accepted = reasons == "accepted"
        chosen = indices[accepted]
        output["estimated_x"][chosen] = cx[accepted]
        output["estimated_y"][chosen] = cy[accepted]
        output["steger_offsets"][chosen] = offset[accepted]
        output["steger_refined"][indices] = accepted
        output["steger_fallback"][indices] = ~accepted
        output["refine_attempted"][indices] = True
        output["failure_reason"][indices] = reasons
        for key, values in [("hessian_normal_x", principal[:, 0]), ("hessian_normal_y", principal[:, 1]), ("lambda_main", main), ("lambda_other", other), ("eigen_ratio", ratio), ("normal_angle_deg", angles), ("hessian_response", response), ("taylor_denominator", denominator)]:
            output[key][indices] = values
        output["sigma_used"][indices] = sigma
    output["valid"] = np.isfinite(output["estimated_x"]) & np.isfinite(output["estimated_y"])
    return output
