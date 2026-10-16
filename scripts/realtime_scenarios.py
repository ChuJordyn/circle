from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from line_grayscale_centroid_demo import (
    generate_curve_stripe,
    normalize_to_uint8,
)


IMAGE_SIZE = 512


def build_single_factor_scenarios(seed: int = 7, noise_sigma: float = 5.0) -> list[Scenario]:
    from single_factor_scenarios import build_single_factor_scenarios as build
    return build(seed, noise_sigma=noise_sigma)


@dataclass(frozen=True)
class Scenario:
    name: str
    image: np.ndarray
    truth_y: np.ndarray


def generate_straight_stripe(
    size: int = IMAGE_SIZE,
    seed: int = 101,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = np.arange(size, dtype=np.float32)
    truth_y = 255.0 + 0.12 * (x - size / 2.0)
    stripe = 220.0 * np.exp(
        -0.5 * ((yy - truth_y[np.newaxis, :]) / 2.6) ** 2
    )
    background = 18.0 + 0.02 * yy
    noise = rng.normal(0.0, 3.5, size=(size, size))
    return normalize_to_uint8(background + stripe + noise), {"truth_y": truth_y}


def add_noise(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalize_to_uint8(
        np.asarray(image, dtype=np.float32)
        + rng.normal(0.0, sigma, image.shape)
    )


def add_secondary_peak(
    image: np.ndarray,
    truth_y: np.ndarray,
    offset: float = 5.0,
) -> np.ndarray:
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]].astype(np.float32)
    secondary = 95.0 * np.exp(
        -0.5 * ((yy - (truth_y[np.newaxis, :] + offset)) / 2.0) ** 2
    )
    return normalize_to_uint8(np.asarray(image, dtype=np.float32) + secondary)


def add_broken_segments(image: np.ndarray, truth_y: np.ndarray) -> np.ndarray:
    output = np.asarray(image, dtype=np.float32).copy()
    for x_min, x_max in [(125, 150), (315, 345)]:
        for x in range(x_min, x_max):
            center = int(round(float(truth_y[x])))
            y0 = max(0, center - 8)
            y1 = min(output.shape[0], center + 9)
            output[y0:y1, x] *= 0.08
    return normalize_to_uint8(output)


def build_scenarios(seed: int = 7) -> list[Scenario]:
    """Nine scenarios paired to one baseline; each changes only its named factor."""
    isolated = {s.name: s for s in build_single_factor_scenarios(seed)}
    aliases = [
        ('normal_straight','control'),('curved','curved_only'),
        ('large_curvature','curvature_only'),('weak_light','weak_light_only'),
        ('noise','noise_only'),('reflection','reflection_only'),
        ('saturation','saturation_only'),('double_peak','double_peak_only'),
        ('broken','broken_only'),
    ]
    return [Scenario(name,isolated[key].image,isolated[key].truth_y) for name,key in aliases]
