from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from centerline.gray_centroid import (
    extract_vertical_gray_centroid,
    gray_centroid_on_profile,
)
from centerline.normal_guided_centroid import (
    NormalGuidedConfig,
    bilinear_sample,
    estimate_tangent_normal,
    extract_normal_guided_gray_centroid,
    summarize_errors,
)
from centerline.normal_guided_steger import (
    DerivativeFields,
    StegerConfig,
    hessian_principal_direction,
    steger_refine_at_point,
)
from centerline.pipeline import (
    RealtimePipelineConfig,
    realtime_frame_budget_ms,
    run_realtime_centerline_pipeline,
)
from centerline.preprocess import make_roi_mask, preprocess_laser_image


class CenterlineMathTests(unittest.TestCase):
    def test_bilinear_sample_subpixel(self) -> None:
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        value = bilinear_sample(
            image,
            np.asarray([1.5], dtype=np.float32),
            np.asarray([1.5], dtype=np.float32),
        )[0]
        self.assertAlmostEqual(float(value), 7.5, places=5)

    def test_gray_centroid_is_centered_for_symmetric_profile(self) -> None:
        coordinates = np.arange(-4, 5, dtype=np.float32)
        profile = np.exp(-0.5 * (coordinates / 1.4) ** 2).astype(np.float32)
        center = gray_centroid_on_profile(coordinates, profile, threshold_ratio=0.1)
        self.assertIsNotNone(center)
        self.assertAlmostEqual(float(center), 0.0, places=3)

    def test_tangent_and_normal_are_orthogonal(self) -> None:
        y = np.arange(32, dtype=np.float32) * 2.0
        tangent_x, tangent_y, normal_x, normal_y = estimate_tangent_normal(y)
        dot = tangent_x * normal_x + tangent_y * normal_y
        self.assertTrue(np.allclose(dot, 0.0, atol=1e-6))
        self.assertTrue(np.allclose(tangent_x**2 + tangent_y**2, 1.0, atol=1e-6))
        self.assertTrue(np.allclose(normal_x**2 + normal_y**2, 1.0, atol=1e-6))

    def test_hessian_uses_largest_absolute_eigenvalue_direction(self) -> None:
        main, other, normal = hessian_principal_direction(
            np.asarray([[-4.0, 0.0], [0.0, -1.0]], dtype=np.float32)
        )
        self.assertAlmostEqual(main, -4.0, places=5)
        self.assertAlmostEqual(other, -1.0, places=5)
        self.assertTrue(np.allclose(np.abs(normal), np.asarray([1.0, 0.0]), atol=1e-6))

    def test_taylor_refine_accepts_valid_subpixel_shift(self) -> None:
        fields = DerivativeFields(
            smoothed=np.ones((8, 8), dtype=np.float32),
            ix=np.ones((8, 8), dtype=np.float32),
            iy=np.zeros((8, 8), dtype=np.float32),
            ixx=np.full((8, 8), -2.0, dtype=np.float32),
            iyy=np.full((8, 8), -0.5, dtype=np.float32),
            ixy=np.zeros((8, 8), dtype=np.float32),
        )
        result = steger_refine_at_point(
            fields=fields,
            base_x=3.0,
            base_y=3.0,
            guided_normal_x=1.0,
            guided_normal_y=0.0,
            image_shape=(8, 8),
            config=StegerConfig(
                hessian_response_threshold=0.1,
                eigenvalue_ratio_threshold=1.5,
                max_offset=1.0,
            ),
        )
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(float(result["offset"]), 0.5, places=5)
        self.assertAlmostEqual(float(result["x"]), 3.5, places=5)

    def test_taylor_refine_falls_back_when_offset_is_too_large(self) -> None:
        fields = DerivativeFields(
            smoothed=np.ones((8, 8), dtype=np.float32),
            ix=np.full((8, 8), 5.0, dtype=np.float32),
            iy=np.zeros((8, 8), dtype=np.float32),
            ixx=np.full((8, 8), -2.0, dtype=np.float32),
            iyy=np.full((8, 8), -0.5, dtype=np.float32),
            ixy=np.zeros((8, 8), dtype=np.float32),
        )
        result = steger_refine_at_point(
            fields=fields,
            base_x=3.0,
            base_y=3.0,
            guided_normal_x=1.0,
            guided_normal_y=0.0,
            image_shape=(8, 8),
            config=StegerConfig(
                hessian_response_threshold=0.1,
                eigenvalue_ratio_threshold=1.5,
                max_offset=1.0,
            ),
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "offset_too_large")
        self.assertAlmostEqual(float(result["x"]), 3.0, places=5)

    def test_preprocess_builds_roi_and_valid_region(self) -> None:
        image = np.zeros((10, 12), dtype=np.float32)
        image[4:6, 5:7] = 100.0
        roi = make_roi_mask(image.shape, (2, 2, 10, 8))
        self.assertTrue(roi[4, 5])
        self.assertFalse(roi[0, 0])
        result = preprocess_laser_image(
            image,
            roi=(2, 2, 10, 8),
            valid_threshold_ratio=0.5,
        )
        self.assertEqual(result.image.shape, image.shape)
        self.assertTrue(result.valid_mask[4, 5])
        self.assertFalse(result.valid_mask[0, 0])

    def test_empty_error_summary_is_safe(self) -> None:
        summary = summarize_errors(np.asarray([np.nan, np.nan], dtype=np.float32))
        self.assertEqual(summary["valid_count"], 0)
        self.assertTrue(np.isnan(summary["rmse_px"]))

    def test_vectorized_gray_centroid_matches_scalar(self) -> None:
        yy, xx = np.mgrid[0:48, 0:32].astype(np.float32)
        image = np.exp(-0.5 * ((yy - (20.0 + 0.08 * xx)) / 2.0) ** 2).astype(np.float32)
        scalar = extract_vertical_gray_centroid(
            image,
            y_min=5,
            y_max=40,
            threshold_ratio=0.35,
        )
        vectorized = extract_vertical_gray_centroid(
            image,
            y_min=5,
            y_max=40,
            threshold_ratio=0.35,
            vectorized=True,
        )
        self.assertTrue(np.array_equal(scalar["valid"], vectorized["valid"]))
        self.assertTrue(
            np.allclose(
                scalar["estimated_y"],
                vectorized["estimated_y"],
                equal_nan=True,
                atol=1e-5,
            )
        )

    def test_vectorized_guided_centroid_matches_scalar(self) -> None:
        yy, xx = np.mgrid[0:64, 0:40].astype(np.float32)
        truth_y = 30.0 + 0.12 * xx
        image = np.exp(-0.5 * ((yy - truth_y) / 2.0) ** 2).astype(np.float32)
        config = NormalGuidedConfig(
            y_min=8,
            y_max=56,
            normal_half_width=8.0,
            normal_step=0.5,
            use_robust_support=False,
        )
        scalar = extract_normal_guided_gray_centroid(image, config=config)
        vectorized = extract_normal_guided_gray_centroid(
            image,
            config=config,
            vectorized=True,
        )
        self.assertTrue(np.array_equal(scalar["valid"], vectorized["valid"]))
        self.assertTrue(
            np.allclose(
                scalar["estimated_x"],
                vectorized["estimated_x"],
                equal_nan=True,
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                scalar["estimated_y"],
                vectorized["estimated_y"],
                equal_nan=True,
                atol=1e-4,
            )
        )

    def test_realtime_budget_for_project_condition(self) -> None:
        self.assertAlmostEqual(realtime_frame_budget_ms(50.0, 10.0), 200.0)
        config = RealtimePipelineConfig(speed_mm_s=50.0, sample_step_mm=10.0)
        self.assertAlmostEqual(config.required_fps, 5.0)
        self.assertAlmostEqual(config.frame_budget_ms, 200.0)

    def test_realtime_pipeline_returns_fast_path_result(self) -> None:
        yy, xx = np.mgrid[0:64, 0:40].astype(np.float32)
        image = np.exp(-0.5 * ((yy - (30.0 + 0.12 * xx)) / 2.0) ** 2).astype(np.float32)
        config = RealtimePipelineConfig(
            y_min=8,
            y_max=56,
            normal=NormalGuidedConfig(y_min=8, y_max=56),
            refine_with_steger=False,
        )
        result = run_realtime_centerline_pipeline(image, config)
        self.assertIn("timings_ms", result)
        self.assertIn("selected", result)
        self.assertIn("deadline_met", result)
        self.assertIs(result["selected"], result["guided"])


if __name__ == "__main__":
    unittest.main()
