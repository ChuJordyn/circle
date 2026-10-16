"""Guard against reusing a compound scene as a single-factor test."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts')]
import realtime_scenarios


class SingleFactorScenarioTests(unittest.TestCase):
    def build(self, seed=7, noise_sigma=5.0):
        self.assertTrue(hasattr(realtime_scenarios, 'build_single_factor_scenarios'),
                        'Independent single-factor suite is missing')
        return {s.name: s for s in realtime_scenarios.build_single_factor_scenarios(seed, noise_sigma=noise_sigma)}

    def test_curvature_has_no_extra_brightness_or_missing_stripe(self):
        scenes = self.build(noise_sigma=0)
        curve = scenes['curvature_only']
        self.assertGreater(float(np.ptp(curve.truth_y)), 150)
        self.assertLessEqual(int(curve.image.max()), 180)
        self.assertGreaterEqual(int(curve.image.max(axis=0).min()), 175)

    def test_reflection_is_local_and_preserves_straight_truth(self):
        scenes = self.build(noise_sigma=0)
        control, reflection = scenes['control'], scenes['reflection_only']
        np.testing.assert_array_equal(reflection.truth_y, np.full(512, 255.0))
        np.testing.assert_array_equal(control.image[:, :15], reflection.image[:, :15])
        np.testing.assert_array_equal(control.image[:, -15:], reflection.image[:, -15:])
        self.assertTrue((reflection.image[:, 70:130] > control.image[:, 70:130]).any())
        self.assertLess(int(reflection.image.max()), 255)

    def test_weak_secondary_stripe_is_uniform_and_preserves_main_truth(self):
        scenes = self.build(noise_sigma=0)
        control, double = scenes['control'], scenes['double_peak_only']
        np.testing.assert_array_equal(double.truth_y, np.full(512, 255.0))
        np.testing.assert_array_equal(double.image[:, 0], double.image[:, 250])
        np.testing.assert_array_equal(double.image[:, 250], double.image[:, 511])
        # The requested contaminant contributes 160*15%=24 above the paired
        # control; with overlap/quantization it need not make a separate maximum.
        self.assertEqual(int(double.image[263, 100]) - int(control.image[263, 100]), 24)
        self.assertGreater(int(double.image[255, 100]), int(double.image[254, 100]))
        self.assertGreater(int(double.image[255, 100]), int(double.image[256, 100]))
        self.assertLess(int(double.image.max()), 255)

    def test_all_factors_share_paired_noise_and_seeds_are_reproducible(self):
        first, replay, other = self.build(), self.build(), self.build(seed=37)
        for name, scene in first.items():
            np.testing.assert_array_equal(scene.image, replay[name].image)
            self.assertFalse(np.array_equal(scene.image, other[name].image))
            np.testing.assert_array_equal(scene.truth_y, other[name].truth_y)
            if name!='noise_only':
                np.testing.assert_array_equal(scene.image[:60], first['control'].image[:60])

    def test_mild_curve_has_no_added_glare_or_saturation(self):
        scenes=self.build(noise_sigma=0)
        self.assertIn('curved_only',scenes.keys())
        curve=scenes['curved_only']
        self.assertGreater(float(np.ptp(curve.truth_y)),60.0)
        self.assertLessEqual(int(curve.image.max()),180)

    def test_weak_light_changes_signal_without_scaling_background(self):
        scenes=self.build(noise_sigma=0)
        self.assertIn('weak_light_only',scenes.keys())
        weak=scenes['weak_light_only']
        np.testing.assert_array_equal(weak.truth_y,np.full(512,255.0))
        self.assertTrue(np.all(weak.image[255]==87))  # 20 + 160*0.42, quantized.
        np.testing.assert_array_equal(weak.image[:60],scenes['control'].image[:60])
        np.testing.assert_array_equal(weak.image[:,0],weak.image[:,511])

    def test_stronger_noise_keeps_geometry_and_has_no_white_saturation(self):
        for seed in [7,37,67,97,127]:
            scenes=self.build(seed=seed)
            self.assertIn('noise_only',scenes.keys())
            noise=scenes['noise_only']
            np.testing.assert_array_equal(noise.truth_y,np.full(512,255.0))
            self.assertFalse((noise.image==255).any())
            self.assertGreater(float(noise.image[:60].std()),2*float(scenes['control'].image[:60].std()))

    def test_broken_signal_leaves_background_intact_and_other_columns_unchanged(self):
        scenes=self.build(noise_sigma=0)
        self.assertIn('broken_only',scenes.keys())
        broken=scenes['broken_only']
        gaps=np.zeros(512,dtype=bool)
        gaps[125:150]=True
        gaps[315:345]=True
        self.assertTrue(np.all(broken.image[:,gaps]==20))
        np.testing.assert_array_equal(broken.image[:,~gaps],scenes['control'].image[:,~gaps])
        np.testing.assert_array_equal(broken.truth_y,scenes['control'].truth_y)

    def test_saturation_has_a_straight_symmetric_flat_top_without_glare(self):
        scenes = self.build(noise_sigma=0)
        self.assertIn('saturation_only', scenes)
        saturated = scenes['saturation_only']
        np.testing.assert_array_equal(saturated.truth_y, np.full(512,255.0))
        np.testing.assert_array_equal(saturated.image[:,0], saturated.image[:,511])
        self.assertTrue(np.all(saturated.image[254:257] == 255))
        self.assertTrue(np.all(saturated.image[253] < 255))
        np.testing.assert_array_equal(saturated.image[253], saturated.image[257])

    def test_default_benchmark_uses_the_isolated_inputs(self):
        isolated = self.build()
        self.assertIn('saturation_only', isolated)
        default = {s.name: s for s in realtime_scenarios.build_scenarios(seed=7)}
        self.assertEqual(len(default),9)
        for name, replacement in [('normal_straight','control'),('curved','curved_only'),
                                  ('weak_light','weak_light_only'),('noise','noise_only'),
                                  ('broken','broken_only'),('large_curvature', 'curvature_only'),
                                  ('reflection', 'reflection_only'),
                                  ('double_peak', 'double_peak_only'),
                                  ('saturation', 'saturation_only')]:
            self.assertIn(replacement,isolated.keys())
            np.testing.assert_array_equal(default[name].image, isolated[replacement].image)
            np.testing.assert_array_equal(default[name].truth_y, isolated[replacement].truth_y)


if __name__ == '__main__':
    unittest.main()
