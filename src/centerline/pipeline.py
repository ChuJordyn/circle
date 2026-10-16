from __future__ import annotations

from dataclasses import dataclass, field, replace
import time

import numpy as np

from centerline.gray_centroid import extract_vertical_gray_centroid, extract_vertical_peak_centerline
from centerline.normal_guided_centroid import NormalGuidedConfig, extract_normal_guided_gray_centroid
from centerline.normal_guided_steger import StegerConfig, extract_normal_guided_steger
from centerline.preprocess import PreprocessResult, ROI, preprocess_laser_image


@dataclass(frozen=True)
class CenterlinePipelineConfig:
    roi: ROI | None = None
    y_min: int = 120
    y_max: int = 390
    background_percentile: float = 15.0
    valid_threshold_ratio: float = 0.05
    normalize: bool = False
    normal: NormalGuidedConfig = field(default_factory=NormalGuidedConfig)
    steger: StegerConfig = field(default_factory=StegerConfig)


@dataclass(frozen=True)
class RealtimePipelineConfig:
    roi: ROI | None = None
    y_min: int = 60
    y_max: int = 450
    background_percentile: float = 15.0
    valid_threshold_ratio: float = 0.02
    normalize: bool = False
    speed_mm_s: float = 50.0
    sample_step_mm: float = 10.0
    normal: NormalGuidedConfig = field(
        default_factory=lambda: NormalGuidedConfig(y_min=60, y_max=450)
    )
    refine_with_steger: bool = True
    steger: StegerConfig = field(default_factory=lambda: StegerConfig(adaptive_sigma=True, sigma_scale=0.2))

    @property
    def required_fps(self) -> float:
        if self.speed_mm_s <= 0 or self.sample_step_mm <= 0:
            raise ValueError("speed_mm_s and sample_step_mm must be positive")
        return float(self.speed_mm_s / self.sample_step_mm)

    @property
    def frame_budget_ms(self) -> float:
        return float(1000.0 / self.required_fps)


def realtime_frame_budget_ms(speed_mm_s: float, sample_step_mm: float) -> float:
    if speed_mm_s <= 0 or sample_step_mm <= 0:
        raise ValueError("speed_mm_s and sample_step_mm must be positive")
    return float(1000.0 * sample_step_mm / speed_mm_s)


def run_centerline_pipeline(
    image: np.ndarray,
    config: CenterlinePipelineConfig | None = None,
) -> dict[str, object]:
    config = config or CenterlinePipelineConfig()
    preprocess: PreprocessResult = preprocess_laser_image(
        image,
        roi=config.roi,
        background_percentile=config.background_percentile,
        valid_threshold_ratio=config.valid_threshold_ratio,
        normalize=config.normalize,
    )

    gray = extract_vertical_gray_centroid(
        preprocess.image,
        y_min=config.y_min,
        y_max=config.y_max,
        threshold_ratio=config.normal.threshold_ratio,
        roi_mask=preprocess.roi_mask,
    )
    peak = extract_vertical_peak_centerline(
        preprocess.image,
        y_min=config.y_min,
        y_max=config.y_max,
        roi_mask=preprocess.roi_mask,
    )
    guided = extract_normal_guided_gray_centroid(
        preprocess.image,
        config=replace(config.normal, y_min=config.y_min, y_max=config.y_max),
        roi_mask=preprocess.roi_mask,
        valid_mask=preprocess.valid_mask,
    )
    steger = extract_normal_guided_steger(
        preprocess.image,
        steger_config=config.steger,
        roi_mask=preprocess.roi_mask,
        valid_mask=preprocess.valid_mask,
        guided=guided,
    )
    return {
        "preprocess": preprocess,
        "peak": peak,
        "gray": gray,
        "guided": guided,
        "steger": steger,
    }


def run_realtime_centerline_pipeline(
    image: np.ndarray,
    config: RealtimePipelineConfig | None = None,
) -> dict[str, object]:
    config = config or RealtimePipelineConfig()
    timings_ms: dict[str, float] = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    preprocess: PreprocessResult = preprocess_laser_image(
        image,
        roi=config.roi,
        background_percentile=config.background_percentile,
        valid_threshold_ratio=config.valid_threshold_ratio,
        normalize=config.normalize,
    )
    timings_ms["preprocess"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    gray = extract_vertical_gray_centroid(
        preprocess.image,
        y_min=config.y_min,
        y_max=config.y_max,
        threshold_ratio=config.normal.threshold_ratio,
        roi_mask=preprocess.roi_mask,
        vectorized=True,
    )
    timings_ms["gray"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    guided = extract_normal_guided_gray_centroid(
        preprocess.image,
        config=replace(config.normal, y_min=config.y_min, y_max=config.y_max),
        roi_mask=preprocess.roi_mask,
        valid_mask=preprocess.valid_mask,
        vectorized=True,
    )
    timings_ms["guided"] = (time.perf_counter() - start) * 1000.0

    steger = None
    selected = guided
    if config.refine_with_steger:
        start = time.perf_counter()
        steger = extract_normal_guided_steger(
            preprocess.image, steger_config=config.steger,
            roi_mask=preprocess.roi_mask, valid_mask=preprocess.valid_mask,
            guided=guided,
        )
        timings_ms["steger"] = (time.perf_counter() - start) * 1000.0
        selected = steger

    timings_ms["total"] = (time.perf_counter() - total_start) * 1000.0
    return {
        "preprocess": preprocess,
        "gray": gray,
        "guided": guided,
        "selected": selected,
        "steger": steger,
        "timings_ms": timings_ms,
        "frame_budget_ms": config.frame_budget_ms,
        "required_fps": config.required_fps,
        "deadline_met": timings_ms["total"] <= config.frame_budget_ms,
    }
