from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from centerline.normal_guided_centroid import (  # noqa: E402
    errors_against_truth_normal_distance,
    extract_normal_guided_gray_centroid,
    summarize_errors,
)
from centerline.gray_centroid import extract_vertical_gray_centroid  # noqa: E402

OUT_DIR = PROJECT_ROOT / "result" / "line_normal_guided_centroid"
IMAGE_SIZE = 512
RANDOM_SEED = 7


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = image.shape
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
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


def generate_annular_stripe(size: int = IMAGE_SIZE) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(RANDOM_SEED)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = size / 2 + 4.7
    cy = size / 2 - 6.2
    dx = xx - cx
    dy = yy - cy
    theta = np.arctan2(dy, dx)
    radius = np.sqrt(dx * dx + dy * dy)

    # Slightly non-circular truth to imitate a real measured inner profile.
    r_true = (
        155.0
        + 5.5 * np.sin(3.0 * theta + 0.4)
        + 2.0 * np.cos(7.0 * theta - 0.8)
        + 1.2 * np.sin(11.0 * theta)
    )
    stripe_sigma = 2.7
    stripe = 220.0 * np.exp(-0.5 * ((radius - r_true) / stripe_sigma) ** 2)

    background = 18.0 + 0.025 * yy + 0.018 * xx
    noise = rng.normal(0.0, 5.5, size=(size, size))

    # Add a few bright asymmetric spots to imitate metal reflection.
    reflection = np.zeros((size, size), dtype=np.float32)
    for angle, rr, amp, sx, sy in [
        (0.45, 158.0, 70.0, 5.0, 10.0),
        (2.15, 151.0, 45.0, 7.0, 6.0),
        (-1.2, 160.0, 55.0, 6.0, 8.0),
    ]:
        px = cx + rr * math.cos(angle)
        py = cy + rr * math.sin(angle)
        reflection += amp * np.exp(-0.5 * (((xx - px) / sx) ** 2 + ((yy - py) / sy) ** 2))

    image = normalize_to_uint8(background + stripe + reflection + noise)
    truth = {
        "center": np.array([cx, cy], dtype=np.float32),
    }
    return image, truth


def generate_hard_annular_stripe(size: int = IMAGE_SIZE) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(RANDOM_SEED + 20)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = size / 2 + rng.uniform(-8.0, 8.0)
    cy = size / 2 + rng.uniform(-8.0, 8.0)
    dx = xx - cx
    dy = yy - cy
    theta = np.arctan2(dy, dx)
    radius = np.sqrt(dx * dx + dy * dy)

    r_true = annular_truth_radius(theta)
    stripe_sigma = 2.5 + 1.2 * (0.5 + 0.5 * np.sin(5.0 * theta - 0.3))
    base = np.exp(-0.5 * ((radius - r_true) / stripe_sigma) ** 2)

    # Side bias makes one side of the stripe brighter, which shifts the gray centroid.
    side_bias = 1.0 + 0.45 * np.tanh((radius - r_true) / 2.5) * (0.5 + 0.5 * np.sin(4.0 * theta))
    stripe = 205.0 * base * side_bias

    background = 23.0 + 0.035 * yy + 0.018 * xx + 20.0 * np.sin((xx + yy) / 135.0)
    noise = rng.normal(0.0, 11.5, size=(size, size))

    reflection = np.zeros((size, size), dtype=np.float32)
    for _ in range(9):
        angle = rng.uniform(-math.pi, math.pi)
        rr = 155.0 + rng.normal(0.0, 7.0)
        amp = rng.uniform(65.0, 135.0)
        sx = rng.uniform(4.0, 13.0)
        sy = rng.uniform(4.0, 13.0)
        px = cx + rr * math.cos(angle)
        py = cy + rr * math.sin(angle)
        reflection += amp * np.exp(-0.5 * (((xx - px) / sx) ** 2 + ((yy - py) / sy) ** 2))

    # Local dark occlusion simulates oil/dust/stripe interruption.
    occlusion = np.ones((size, size), dtype=np.float32)
    for angle, width, strength in [
        (0.9, 0.16, 0.55),
        (-2.2, 0.12, 0.45),
    ]:
        angular_distance = np.angle(np.exp(1j * (theta - angle)))
        occlusion *= 1.0 - strength * np.exp(-0.5 * (angular_distance / width) ** 2)

    image = normalize_to_uint8(background + stripe * occlusion + reflection + noise)
    truth = {
        "center": np.array([cx, cy], dtype=np.float32),
    }
    return image, truth


def annular_truth_radius(theta: np.ndarray) -> np.ndarray:
    return (
        155.0
        + 5.5 * np.sin(3.0 * theta + 0.4)
        + 2.0 * np.cos(7.0 * theta - 0.8)
        + 1.2 * np.sin(11.0 * theta)
    )


def extract_annular_centroid(
    image: np.ndarray,
    center: np.ndarray,
    theta_count: int = 720,
    r_min: float = 130.0,
    r_max: float = 180.0,
    r_step: float = 0.25,
    threshold_ratio: float = 0.35,
) -> dict[str, np.ndarray]:
    cx, cy = float(center[0]), float(center[1])
    angles = np.linspace(-math.pi, math.pi, theta_count, endpoint=False, dtype=np.float32)
    radii = np.arange(r_min, r_max + r_step, r_step, dtype=np.float32)

    estimated_r = np.full(theta_count, np.nan, dtype=np.float32)
    estimated_xy = np.full((theta_count, 2), np.nan, dtype=np.float32)

    for i, theta in enumerate(angles):
        xs = cx + radii * math.cos(float(theta))
        ys = cy + radii * math.sin(float(theta))
        profile = bilinear_sample(image, xs, ys)

        peak = float(profile.max())
        if peak <= 0:
            continue

        # Local threshold keeps the centroid focused on the bright stripe body.
        local_min = float(np.percentile(profile, 15))
        cutoff = local_min + threshold_ratio * (peak - local_min)
        weights = profile - cutoff
        weights[weights < 0] = 0

        weight_sum = float(weights.sum())
        if weight_sum <= 1e-6:
            continue

        r_centroid = float((radii * weights).sum() / weight_sum)
        estimated_r[i] = r_centroid
        estimated_xy[i] = [cx + r_centroid * math.cos(float(theta)), cy + r_centroid * math.sin(float(theta))]

    truth_r = annular_truth_radius(angles)
    errors = estimated_r - truth_r
    return {
        "angles": angles,
        "truth_r": truth_r,
        "estimated_r": estimated_r,
        "estimated_xy": estimated_xy,
        "errors": errors,
    }


def generate_curve_stripe(size: int = IMAGE_SIZE, seed: int = RANDOM_SEED + 1) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    x = np.arange(size, dtype=np.float32)
    y_true_1d = 255.0 + 46.0 * np.sin((x - 30.0) / 68.0) + 22.0 * np.sin(x / 23.0)
    y_true = y_true_1d[np.newaxis, :]

    stripe_sigma = 2.6
    stripe = 215.0 * np.exp(-0.5 * ((yy - y_true) / stripe_sigma) ** 2)
    background = 20.0 + 0.04 * yy
    noise = rng.normal(0.0, 5.0, size=(size, size))

    glare = 60.0 * np.exp(-0.5 * (((xx - 335.0) / 25.0) ** 2 + ((yy - 245.0) / 6.0) ** 2))
    image = normalize_to_uint8(background + stripe + glare + noise)
    return image, {"truth_y": y_true_1d}


def generate_hard_curve_stripe(size: int = IMAGE_SIZE, seed: int = RANDOM_SEED + 21) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    x = np.arange(size, dtype=np.float32)
    y_true_1d = (
        255.0
        + 62.0 * np.sin((x - 25.0) / 55.0)
        + 34.0 * np.sin(x / 18.0)
        + 13.0 * np.sin(x / 7.0)
    )
    y_true = y_true_1d[np.newaxis, :]
    stripe_sigma = 2.4 + 1.8 * (0.5 + 0.5 * np.sin(x[np.newaxis, :] / 31.0))
    base = np.exp(-0.5 * ((yy - y_true) / stripe_sigma) ** 2)
    side_bias = 1.0 + 0.55 * np.tanh((yy - y_true) / 2.4) * (0.5 + 0.5 * np.sin(xx / 42.0))
    stripe = 205.0 * base * side_bias

    background = 24.0 + 0.06 * yy + 17.0 * np.sin(xx / 90.0)
    noise = rng.normal(0.0, 12.0, size=(size, size))

    glare = np.zeros((size, size), dtype=np.float32)
    for _ in range(7):
        px = rng.uniform(60.0, size - 60.0)
        py = float(np.interp(px, x, y_true_1d)) + rng.normal(0.0, 8.0)
        amp = rng.uniform(60.0, 130.0)
        sx = rng.uniform(7.0, 25.0)
        sy = rng.uniform(3.0, 10.0)
        glare += amp * np.exp(-0.5 * (((xx - px) / sx) ** 2 + ((yy - py) / sy) ** 2))

    occlusion = np.ones((size, size), dtype=np.float32)
    for px, width, strength in [(145.0, 18.0, 0.55), (380.0, 25.0, 0.45)]:
        occlusion *= 1.0 - strength * np.exp(-0.5 * ((xx - px) / width) ** 2)

    image = normalize_to_uint8(background + stripe * occlusion + glare + noise)
    return image, {"truth_y": y_true_1d}


def extract_curve_centroid(
    image: np.ndarray,
    y_min: int = 120,
    y_max: int = 390,
    threshold_ratio: float = 0.35,
) -> dict[str, np.ndarray]:
    return extract_vertical_gray_centroid(image, y_min, y_max, threshold_ratio)


def draw_annular_overlay(image: np.ndarray, result: dict[str, np.ndarray], path: Path) -> None:
    rgb = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    xy = result["estimated_xy"]

    points = [(float(x), float(y)) for x, y in xy if np.isfinite(x) and np.isfinite(y)]
    for point in points:
        x, y = point
        draw.ellipse((x - 1.1, y - 1.1, x + 1.1, y + 1.1), fill=(255, 40, 40))

    rgb.save(path)


def draw_curve_overlay(image: np.ndarray, result: dict[str, np.ndarray], path: Path) -> None:
    rgb = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    estimated_y = result["estimated_y"]
    for x, y in enumerate(estimated_y):
        if np.isfinite(y):
            draw.ellipse((x - 1, float(y) - 1, x + 1, float(y) + 1), fill=(255, 40, 40))
    rgb.save(path)


def draw_curve_comparison_overlay(
    image: np.ndarray,
    gray_result: dict[str, np.ndarray],
    normal_result: dict[str, np.ndarray],
    path: Path,
) -> None:
    rgb = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(rgb)

    for x, y in enumerate(gray_result["estimated_y"]):
        if np.isfinite(y):
            draw.ellipse((x - 1, float(y) - 1, x + 1, float(y) + 1), fill=(255, 50, 50))

    xs = normal_result["estimated_x"]
    ys = normal_result["estimated_y"]
    for x, y in zip(xs, ys):
        if np.isfinite(x) and np.isfinite(y):
            draw.ellipse((float(x) - 1, float(y) - 1, float(x) + 1, float(y) + 1), fill=(30, 220, 80))

    rgb.save(path)


def save_annular_csv(result: dict[str, np.ndarray], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_rad", "truth_radius_px", "estimated_radius_px", "error_px"])
        for theta, truth, estimate, error in zip(
            result["angles"], result["truth_r"], result["estimated_r"], result["errors"]
        ):
            writer.writerow([float(theta), float(truth), float(estimate), float(error)])


def save_curve_csv(truth_y: np.ndarray, result: dict[str, np.ndarray], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_px", "truth_y_px", "estimated_y_px", "error_px"])
        for x, (truth, estimate) in enumerate(zip(truth_y, result["estimated_y"])):
            writer.writerow([x, float(truth), float(estimate), float(estimate - truth)])


def save_curve_comparison_csv(
    truth_y: np.ndarray,
    gray_result: dict[str, np.ndarray],
    normal_result: dict[str, np.ndarray],
    path: Path,
) -> None:
    gray_x = np.arange(truth_y.size, dtype=np.float32)
    gray_vertical_errors = gray_result["estimated_y"] - truth_y
    gray_normal_errors = errors_against_truth_normal_distance(
        gray_x,
        gray_result["estimated_y"],
        truth_y,
    )
    normal_errors = errors_against_truth_normal_distance(
        normal_result["estimated_x"],
        normal_result["estimated_y"],
        truth_y,
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "x_px",
                "truth_y_px",
                "gray_centroid_y_px",
                "gray_normal_distance_error_px",
                "normal_guided_x_px",
                "normal_guided_y_px",
                "normal_guided_normal_distance_error_px",
                "normal_offset_px",
            ]
        )
        for x in range(truth_y.size):
            gray_y = gray_result["estimated_y"][x]
            writer.writerow(
                [
                    x,
                    float(truth_y[x]),
                    float(gray_y),
                    float(gray_normal_errors[x]),
                    float(normal_result["estimated_x"][x]),
                    float(normal_result["estimated_y"][x]),
                    float(normal_errors[x]),
                    float(normal_result["normal_offsets"][x]),
                ]
            )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    curve_image, curve_meta = generate_curve_stripe()
    curve_result = extract_curve_centroid(curve_image)
    curve_gray_x = np.arange(curve_meta["truth_y"].size, dtype=np.float32)
    curve_errors = summarize_errors(
        errors_against_truth_normal_distance(curve_gray_x, curve_result["estimated_y"], curve_meta["truth_y"])
    )
    curve_normal_result = extract_normal_guided_gray_centroid(curve_image)
    curve_normal_errors = summarize_errors(
        errors_against_truth_normal_distance(
            curve_normal_result["estimated_x"],
            curve_normal_result["estimated_y"],
            curve_meta["truth_y"],
        )
    )

    curve_normal_active = int(np.count_nonzero(curve_normal_result["refined"]))

    Image.fromarray(curve_image).save(OUT_DIR / "line_simulated.png")
    draw_curve_comparison_overlay(
        curve_image,
        curve_result,
        curve_normal_result,
        OUT_DIR / "line_centroid_comparison_overlay.png",
    )
    save_curve_comparison_csv(
        curve_meta["truth_y"],
        curve_result,
        curve_normal_result,
        OUT_DIR / "line_centroid_comparison_errors.csv",
    )

    hard_curve_image, hard_curve_meta = generate_hard_curve_stripe()
    hard_curve_result = extract_curve_centroid(hard_curve_image)
    hard_curve_gray_x = np.arange(hard_curve_meta["truth_y"].size, dtype=np.float32)
    hard_curve_errors = summarize_errors(
        errors_against_truth_normal_distance(
            hard_curve_gray_x,
            hard_curve_result["estimated_y"],
            hard_curve_meta["truth_y"],
        )
    )
    hard_curve_normal_result = extract_normal_guided_gray_centroid(hard_curve_image)
    hard_curve_normal_errors = summarize_errors(
        errors_against_truth_normal_distance(
            hard_curve_normal_result["estimated_x"],
            hard_curve_normal_result["estimated_y"],
            hard_curve_meta["truth_y"],
        )
    )
    hard_curve_normal_active = int(np.count_nonzero(hard_curve_normal_result["refined"]))

    Image.fromarray(hard_curve_image).save(OUT_DIR / "hard_line_simulated.png")
    draw_curve_comparison_overlay(
        hard_curve_image,
        hard_curve_result,
        hard_curve_normal_result,
        OUT_DIR / "hard_line_centroid_comparison_overlay.png",
    )
    save_curve_comparison_csv(
        hard_curve_meta["truth_y"],
        hard_curve_result,
        hard_curve_normal_result,
        OUT_DIR / "hard_line_centroid_comparison_errors.csv",
    )

    print("Gray centroid and normal-guided gray centroid demo for line laser stripe finished.")
    print("Standard line stripe:")
    print("  Gray centroid:")
    for key, value in curve_errors.items():
        print(f"    {key}: {value}")
    print("  Normal-guided gray centroid:")
    for key, value in curve_normal_errors.items():
        print(f"    {key}: {value}")
    print(f"    active_points: {curve_normal_active}")
    print("Hard line stripe:")
    print("  Gray centroid:")
    for key, value in hard_curve_errors.items():
        print(f"    {key}: {value}")
    print("  Normal-guided gray centroid:")
    for key, value in hard_curve_normal_errors.items():
        print(f"    {key}: {value}")
    print(f"    active_points: {hard_curve_normal_active}")
    print(f"Outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
