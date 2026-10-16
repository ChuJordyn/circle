from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from centerline.gray_centroid import extract_vertical_gray_centroid
from centerline.normal_guided_centroid import (
    NormalGuidedConfig, bilinear_sample, extract_normal_guided_gray_centroid,
)
from centerline.normal_guided_steger import StegerConfig, extract_normal_guided_steger
from centerline.pipeline import RealtimePipelineConfig, run_realtime_centerline_pipeline


class RobustCenterlineTests(unittest.TestCase):
    def test_gradient_only_image_has_no_stripe(self):
        yy, xx = np.mgrid[:96, :40].astype(np.float32)
        result = extract_normal_guided_gray_centroid(20 + .08 * yy, config=NormalGuidedConfig(y_min=8,y_max=85))
        self.assertFalse(result['valid'].any())

    def test_interpolated_tracking_cannot_measure_noiseless_gradient_gap(self):
        yy, xx = np.mgrid[:96,:80].astype(np.float32)
        background = 20 + .08*yy
        image = background + 180*np.exp(-.5*((yy-42.2)/2.4)**2)
        image[:,30:48] = background[:,30:48]
        result = extract_normal_guided_gray_centroid(image,config=NormalGuidedConfig(y_min=8,y_max=85))
        self.assertFalse(result['valid'][30:48].any())
        self.assertEqual(int(result['valid'][:25].sum()),25)

    def test_roi_boundary_does_not_turn_gradient_into_stripe(self):
        yy, xx = np.mgrid[:96,:40].astype(np.float32)
        mask = np.zeros_like(yy,dtype=bool)
        mask[15:65] = True
        result = extract_normal_guided_gray_centroid(20 + .08*yy, roi_mask=mask, config=NormalGuidedConfig(y_min=8,y_max=85))
        self.assertFalse(result['valid'].any())

    def test_normal_sampling_stays_in_search_range(self):
        yy, xx = np.mgrid[:110, :40].astype(np.float32)
        image = 200 * np.exp(-.5 * ((yy-78)/2)**2) + 1000 * np.exp(-.5*((yy-89)/2)**2)
        result = extract_normal_guided_gray_centroid(image, config=NormalGuidedConfig(y_min=8,y_max=83))
        self.assertTrue(np.all(result['estimated_y'][result['valid']] < 83))

    def test_centroid_does_not_land_in_disjoint_roi_hole(self):
        yy, xx = np.mgrid[:96, :40].astype(np.float32)
        image = 200 * np.exp(-.5 * ((yy-45)/5)**2)
        mask = np.ones_like(image, dtype=bool)
        mask[43:47] = False
        result = extract_normal_guided_gray_centroid(image, roi_mask=mask, config=NormalGuidedConfig(y_min=8,y_max=85))
        samples = bilinear_sample(mask.astype(np.float32), result['estimated_x'][result['valid']],result['estimated_y'][result['valid']])
        self.assertTrue(np.all(samples >= .5))

    def test_scenarios_use_independent_reproducible_noise(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from realtime_scenarios import build_scenarios
        first, replay, alternate = build_scenarios(seed=7), build_scenarios(seed=7), build_scenarios(seed=37)
        for a, b, c in zip(first, replay, alternate):
            np.testing.assert_array_equal(a.image, b.image)
            np.testing.assert_array_equal(a.truth_y, c.truth_y)
            self.assertFalse(np.array_equal(a.image, c.image), a.name)

    def test_sampling_includes_last_row_and_column(self):
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        actual = bilinear_sample(image, np.array([3., 1.5, -0.1]), np.array([3., 3., 1.]))
        np.testing.assert_allclose(actual, [15., 13.5, 0.], atol=1e-6)

    def test_empty_vectorized_search_returns_no_points(self):
        result = extract_vertical_gray_centroid(np.ones((12, 8)), 20, 30, vectorized=True)
        self.assertEqual(int(result["valid"].sum()), 0)
        self.assertTrue(np.isnan(result["estimated_y"]).all())

    def test_normal_centroid_does_not_bias_sloping_stripe_endpoints(self):
        yy, xx = np.mgrid[:80, :48].astype(np.float32)
        image = 200 * np.exp(-0.5 * ((yy - (25.3 + .2 * xx)) / 2.3) ** 2)
        result = extract_normal_guided_gray_centroid(image, config=NormalGuidedConfig(y_min=8, y_max=65), vectorized=True)
        error = result["estimated_y"] - (25.3 + .2 * result["estimated_x"])
        self.assertGreaterEqual(int(result["valid"].sum()), 46)
        self.assertLess(float(np.nanmax(np.abs(error))), .1)

    def test_normal_centroid_rejects_random_noise_in_missing_stripe(self):
        rng = np.random.default_rng(20261011)
        yy, xx = np.mgrid[:96, :80].astype(np.float32)
        image = 20 + 180 * np.exp(-.5 * ((yy - (42.2 + .05 * xx)) / 2.4) ** 2)
        image[:, 30:48] = 20
        image += rng.normal(0, 5, image.shape)
        result = extract_normal_guided_gray_centroid(image, config=NormalGuidedConfig(y_min=8, y_max=85), vectorized=True)
        self.assertLessEqual(int(result["valid"][33:45].sum()), 1)
        self.assertGreaterEqual(int(result["valid"][:25].sum()), 23)

    def test_blank_image_has_no_centers(self):
        result = extract_normal_guided_gray_centroid(np.zeros((40, 20)), config=NormalGuidedConfig(y_min=0, y_max=40), vectorized=True)
        self.assertFalse(result["valid"].any())

    def test_background_gradient_does_not_become_a_missing_stripe(self):
        rng = np.random.default_rng(121)
        yy, xx = np.mgrid[:160, :96].astype(np.float32)
        background = 18 + .08 * yy
        image = background + 180 * np.exp(-.5 * ((yy - (65.2 + .05 * xx)) / 2.4) ** 2)
        image[:, 30:55] = background[:, 30:55]
        image += rng.normal(0, 4, image.shape)
        result = extract_normal_guided_gray_centroid(image, config=NormalGuidedConfig(y_min=5, y_max=150), vectorized=True)
        self.assertLessEqual(int(result["valid"][34:51].sum()), 1)
        self.assertGreaterEqual(int(result["valid"][:25].sum()), 23)

    def test_invalid_mask_is_respected_by_boundary_fallback(self):
        yy, xx = np.mgrid[:80, :48].astype(np.float32)
        image = 200 * np.exp(-.5 * ((yy - (25.3 + .2 * xx)) / 2.3) ** 2)
        result = extract_normal_guided_gray_centroid(image, valid_mask=np.zeros_like(image, dtype=bool), config=NormalGuidedConfig(y_min=8, y_max=65), vectorized=True)
        self.assertFalse(result["valid"].any())

    def test_large_background_gradient_has_no_false_stripe_in_gap(self):
        rng = np.random.default_rng(8)
        yy, xx = np.mgrid[:512, :96].astype(np.float32)
        background = 20 + .04 * yy
        image = background + 215 * np.exp(-.5 * ((yy - 255.3) / 2.6) ** 2)
        image[:, 30:55] = background[:, 30:55]
        image += rng.normal(0, 5, image.shape)
        result = extract_normal_guided_gray_centroid(image, config=NormalGuidedConfig(y_min=60, y_max=450), vectorized=True)
        self.assertLessEqual(int(result["valid"][33:52].sum()), 1)
        self.assertGreaterEqual(int(result["valid"][:25].sum()), 23)

    def test_reflection_does_not_replace_an_attenuated_stripe(self):
        rng = np.random.default_rng(42)
        yy, xx = np.mgrid[:160, :96].astype(np.float32)
        truth = 65.2 + .05 * xx
        stripe = 200 * np.exp(-.5 * ((yy - truth) / 2.4) ** 2)
        stripe[:, 30:55] *= .03
        glare = 55 * np.exp(-.5 * ((yy - truth - 12) / 5.) ** 2)
        image = 20 + .04 * yy + stripe + glare + rng.normal(0, 4, yy.shape)
        result = extract_normal_guided_gray_centroid(image, config=NormalGuidedConfig(y_min=5, y_max=150), vectorized=True)
        errors = result["estimated_y"] - (65.2 + .05 * result["estimated_x"])
        self.assertLessEqual(int((result["valid"][33:52] & (np.abs(errors[33:52]) > 3)).sum()), 1)
        self.assertGreaterEqual(int(result["valid"][:25].sum()), 23)

    def test_dim_continuous_stripe_is_retained(self):
        yy, xx = np.mgrid[:96, :80].astype(np.float32)
        amplitudes = np.full(80, 200., dtype=np.float32)
        amplitudes[25:55] = 30
        image = 20 + amplitudes * np.exp(-.5 * ((yy - (42.2 + .05 * xx)) / 2.4) ** 2)
        result = extract_normal_guided_gray_centroid(image, config=NormalGuidedConfig(y_min=8, y_max=85), vectorized=True)
        self.assertGreaterEqual(int(result["valid"][30:50].sum()), 18)

    def test_continuity_keeps_dim_stripe_under_stronger_detached_glare(self):
        rng = np.random.default_rng(46)
        yy, xx = np.mgrid[:160, :96].astype(np.float32)
        truth = 65.2 + .05 * xx
        amp = np.full(96, 200., dtype=np.float32)
        amp[30:55] = 70
        glare = 170 * np.exp(-.5*((yy - 95)/3)**2) * ((xx >= 30) & (xx < 55))
        image = 20 + amp * np.exp(-.5*((yy-truth)/2.4)**2) + glare + rng.normal(0,4,yy.shape)
        result = extract_normal_guided_gray_centroid(image,config=NormalGuidedConfig(y_min=5,y_max=150))
        error = result['estimated_y'] - (65.2 + .05 * result['estimated_x'])
        self.assertGreaterEqual(int(result['valid'][33:52].sum()), 18)
        self.assertLess(float(np.nanmax(abs(error[33:52]))), 1)

    def test_outer_search_range_reaches_normal_stage(self):
        yy, xx = np.mgrid[:96, :40].astype(np.float32)
        image = 200 * np.exp(-.5 * ((yy - 25.3) / 2.) ** 2)
        result = run_realtime_centerline_pipeline(image, RealtimePipelineConfig(y_min=8, y_max=45))
        self.assertGreaterEqual(int(result["guided"]["valid"].sum()), 38)

    def test_realtime_pipeline_includes_steger_when_enabled(self):
        yy, xx = np.mgrid[:80, :48].astype(np.float32)
        image = 200 * np.exp(-.5 * ((yy - (25.3 + .2 * xx)) / 2.3) ** 2)
        result = run_realtime_centerline_pipeline(image, RealtimePipelineConfig(y_min=8, y_max=65))
        self.assertIs(result["selected"], result["steger"])
        self.assertIn("steger", result["timings_ms"])
        self.assertGreaterEqual(int(result["steger"]["steger_refined"].sum()), 42)

    def test_batch_steger_matches_scalar_reference(self):
        yy, xx = np.mgrid[:80, :48].astype(np.float32)
        image = 200 * np.exp(-.5 * ((yy - (25.3 + .1 * xx)) / 2.3) ** 2)
        guided = extract_normal_guided_gray_centroid(image, config=NormalGuidedConfig(y_min=8, y_max=65), vectorized=True)
        common = dict(image=image, guided=guided, steger_config=StegerConfig())
        reference = extract_normal_guided_steger(**common, vectorized=False)
        batch = extract_normal_guided_steger(**common, vectorized=True)
        np.testing.assert_array_equal(batch["steger_refined"], reference["steger_refined"])
        np.testing.assert_allclose(batch["estimated_x"], reference["estimated_x"], atol=1e-4, equal_nan=True)
        np.testing.assert_allclose(batch["estimated_y"], reference["estimated_y"], atol=1e-4, equal_nan=True)

    def test_steger_enforces_roi_and_guided_validity(self):
        yy, xx = np.mgrid[:80, :48].astype(np.float32)
        image = 200 * np.exp(-.5*((yy - 25.3)/2.3)**2)
        guided = extract_normal_guided_gray_centroid(image, config=NormalGuidedConfig(y_min=8,y_max=65))
        guided['valid'][12] = False
        mask = np.ones_like(image, dtype=bool)
        mask[24:27, 20:30] = False
        for vectorized in (False, True):
            result = extract_normal_guided_steger(image, guided=guided, roi_mask=mask, vectorized=vectorized)
            self.assertFalse(result['valid'][12])
            self.assertFalse(result['valid'][20:30].any())



if __name__ == "__main__":
    unittest.main()
