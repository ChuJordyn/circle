from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ring_measurement.polar import PolarGrid, unwrap_ring, polar_to_cartesian, cartesian_to_polar
from ring_measurement.calibration import CameraLaserCalibration, fit_scale_from_gauges, calibration_from_dict
from ring_measurement.geometry import fit_circle, fit_ellipse, contour_metrics
from ring_measurement.pipeline import RingMeasurementConfig, run_ring_measurement
from ring_measurement.sweep import analyze_sweep

def ring_image(noise=0.0,gap=False):
    y,x=np.mgrid[:256,:256]
    r=np.hypot(x-127.2,y-126.6)
    theta=np.mod(np.arctan2(y-126.6,x-127.2),2*np.pi)
    signal=160*np.exp(-.5*((r-80.3)/2.6)**2)
    if gap: signal[(theta<.25)|(theta>2*np.pi-.25)]=0
    return (20+signal+np.random.default_rng(7).normal(0,noise,r.shape)).astype(np.float32)

class PolarTests(unittest.TestCase):
    def test_subpixel_coordinates_round_trip_including_seam(self):
        grid=PolarGrid((37.4,42.8),10,30,3600,.5)
        uv=np.array([[.12,3.7],[3599.9,17.31],[1800.1,9.2]])
        np.testing.assert_allclose(cartesian_to_polar(polar_to_cartesian(uv,grid),grid),uv,atol=1e-9)
    def test_bilinear_affine_image_and_unique_angles(self):
        y,x=np.mgrid[:100,:110]
        grid=PolarGrid((53.2,47.1),8,12,72,1)
        polar=unwrap_ring((2*x+3*y).astype(np.float32),grid)
        np.testing.assert_allclose(polar.image,2*polar.source_x+3*polar.source_y,atol=3e-5)
        self.assertTrue(polar.valid_mask.all())
        self.assertEqual(polar.image.shape,(5,72))
        self.assertLess(grid.angles[-1],2*np.pi)
    def test_outside_and_mask_not_observations(self):
        grid=PolarGrid((1,1),5,12,72,1)
        polar=unwrap_ring(np.ones((30,30)),grid)
        self.assertFalse(polar.valid_mask.all())
        self.assertTrue(polar.valid_mask.any())
        self.assertFalse(unwrap_ring(np.ones((30,30)),grid,source_mask=np.zeros((30,30),bool)).valid_mask.any())
    def test_invalid_grids(self):
        for args in [((0,0),-1,10,360,1),((0,0),10,5,360,1),((0,0),1,10,0,1),((0,0),1,10,360,0)]:
            with self.assertRaises(ValueError): PolarGrid(*args)

class CalibrationTests(unittest.TestCase):
    def camera(self,**kwargs):
        values=dict(camera_matrix=[[800,3,128],[0,820,127],[0,0,1]],plane_normal=[0,0,1],plane_offset_mm=-400,distortion=[.07,-.015,.001,-.002,.003],is_simulated=True)
        values.update(kwargs)
        return CameraLaserCalibration(**values)
    def test_distortion_round_trip(self):
        model=self.camera()
        points=np.array([[20.2,11.4],[128,127],[232.9,241.8]])
        np.testing.assert_allclose(model.undistort_points(model.distort_points(points)),points,atol=1e-7)
        restored=calibration_from_dict(model.to_dict())
        np.testing.assert_allclose(restored.camera_matrix,model.camera_matrix)
        self.assertTrue(restored.is_simulated)
    def test_tilted_plane_reconstruction_and_behind_camera(self):
        model=self.camera(plane_normal=[.15,-.2,1])
        norm=np.array([[-.12,.07],[.08,-.04],[0,0]])
        rays=np.column_stack([norm,np.ones(3)])
        expected=rays*(400/(rays@np.array([.15,-.2,1])))[:,None]
        pixels=np.column_stack([norm[:,0]*800+norm[:,1]*3+128,norm[:,1]*820+127])
        points,valid=model.reconstruct(model.distort_points(pixels))
        self.assertTrue(valid.all())
        np.testing.assert_allclose(points,expected,atol=1e-7)
        points,valid=self.camera(plane_offset_mm=400).reconstruct(np.array([[128,127]]))
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(points).all())
    def test_gauge_fit_and_bad_calibration(self):
        scale=fit_scale_from_gauges([100,300,600],[50,150,300],is_simulated=True)
        self.assertAlmostEqual(scale.mm_per_px,.5)
        xy,valid=scale.to_plane(np.array([[20.,30.]]))
        np.testing.assert_allclose(xy,[[10,15]])
        self.assertTrue(valid.all())
        with self.assertRaises(ValueError): self.camera(plane_normal=[0,0,0])
        with self.assertRaises(ValueError): fit_scale_from_gauges([0],[100])

class GeometryTests(unittest.TestCase):
    def test_lsc_metric_uses_all_real_wear_points(self):
        t=np.linspace(0,2*np.pi,720,endpoint=False)
        radius=80+3*np.exp(-.5*((t-.8)/.08)**2)
        points=np.column_stack([radius*np.cos(t),radius*np.sin(t)])
        reference=fit_circle(points,robust=False)
        metrics=contour_metrics(points)
        np.testing.assert_allclose(metrics['circle_center'],reference.center,atol=1e-9)
        self.assertAlmostEqual(metrics['circle_radius'],reference.radius,places=9)
    def test_circle_with_gross_outliers(self):
        t=np.linspace(0,2*np.pi,720,endpoint=False)
        points=np.column_stack([4+80*np.cos(t),-3+80*np.sin(t)])
        fit=fit_circle(np.vstack([points,[[800,900],[-700,800]]]))
        np.testing.assert_allclose(fit.center,[4,-3],atol=1e-7)
        self.assertAlmostEqual(fit.radius,80,places=7)
        self.assertFalse(fit.inliers[-2:].any())
    def test_rotated_ellipse(self):
        t=np.linspace(0,2*np.pi,720,endpoint=False); angle=.43
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        fit=fit_ellipse(np.column_stack([90*np.cos(t),65*np.sin(t)])@rot.T+[7,-9])
        np.testing.assert_allclose(fit.center,[7,-9],atol=1e-8)
        np.testing.assert_allclose(fit.semi_axes,[90,65],atol=1e-8)
        self.assertAlmostEqual(np.sin(fit.angle_rad-angle),0,places=7)
    def test_roundness_preserves_real_lobes(self):
        t=np.linspace(0,2*np.pi,720,endpoint=False); r=80+2*np.cos(3*t)
        metrics=contour_metrics(np.column_stack([r*np.cos(t),r*np.sin(t)]),nominal_diameter=160)
        self.assertAlmostEqual(metrics['roundness_lsc'],4,places=7)
        self.assertAlmostEqual(metrics['maximum_radial_wear'],2,places=7)
        self.assertEqual(metrics['roundness_method'],'least_squares_circle_peak_to_valley')
    def test_degenerate_geometry(self):
        points=np.column_stack([np.arange(20),np.arange(20)*2])
        with self.assertRaises(ValueError): fit_circle(points)
        with self.assertRaises(ValueError): fit_ellipse(points)

class RingPipelineTests(unittest.TestCase):
    def test_saved_points_preserve_fit_flags_and_missing_indices(self):
        import tempfile
        from ring_measurement.io import save_measurement
        result=run_ring_measurement(ring_image(gap=True),self.config(),None)
        with tempfile.TemporaryDirectory() as folder:
            save_measurement(result,folder)
            with np.load(Path(folder)/'points.npz') as saved:
                self.assertEqual(saved['robust_circle_inlier_px'].shape,result['valid'].shape)
                self.assertFalse(saved['robust_circle_inlier_px'][~result['valid']].any())
                np.testing.assert_array_equal(saved['robust_circle_inlier_px'][result['valid']],result['metrics_px']['fit_inliers'])
    def test_missing_neighbor_data_not_used_by_steger_derivatives(self):
        y,x=np.mgrid[:256,:256]
        radius=np.hypot(x-127.2,y-126.6)
        theta=np.mod(np.arctan2(y-126.6,x-127.2),2*np.pi)
        mask=~((theta>1)&(theta<1.025)&(radius>80.4)&(radius<87))
        before=run_ring_measurement(ring_image(),self.config(),None)
        after=run_ring_measurement(ring_image(),self.config(),None,source_mask=mask)
        kept=after['valid']&before['valid']
        self.assertLess(np.max(np.linalg.norm(after['points_px'][kept]-before['points_px'][kept],axis=1)),.05)
    def test_tilted_distorted_camera_end_to_end(self):
        model=CameraLaserCalibration(camera_matrix=[[500,0,127.2],[0,520,126.6],[0,0,1]],
                    plane_normal=[.1,-.2,1],plane_offset_mm=-400,distortion=[.03,-.005,.001,-.001,0],
                    image_size=(256,256),is_simulated=True)
        y,x=np.mgrid[:256,:256]
        pixels=np.column_stack([x.ravel(),y.ravel()])
        plane,valid=model.to_plane(pixels)
        center=(np.array([0,0,400])-model.origin)@model.basis
        radius=np.linalg.norm(plane-center,axis=1)
        image=(20+160*np.exp(-.5*((radius-64)/2.08)**2)).reshape(256,256)
        result=run_ring_measurement(image,RingMeasurementConfig(angular_samples=720),model)
        self.assertTrue(result['measurement_valid'])
        self.assertAlmostEqual(result['metrics_mm']['diameter'],128,delta=.03)
        self.assertLess(np.linalg.norm(result['center_world_mm']-[0,0,400]),.03)
    def test_masked_flank_cannot_bias_a_valid_measurement(self):
        y,x=np.mgrid[:256,:256]
        radius=np.hypot(x-127.2,y-126.6)
        mask=~((radius>81.5)&(radius<89))
        result=run_ring_measurement(ring_image(),self.config(),None,source_mask=mask)
        self.assertFalse(result['measurement_valid'])
        self.assertLess(result['coverage'],.05)
    def config(self,center=(127.2,126.6)):
        return RingMeasurementConfig(grid=PolarGrid(center,55,105,720,1))
    def test_complete_ring_pixel_error_and_millimeters(self):
        model=fit_scale_from_gauges([100,300],[50,150],is_simulated=True)
        result=run_ring_measurement(ring_image(),self.config(),model)
        self.assertGreaterEqual(result['coverage'],.99)
        xy=result['points_px'][result['valid']]
        self.assertLess(np.sqrt(np.mean((np.hypot(xy[:,0]-127.2,xy[:,1]-126.6)-80.3)**2)),.1)
        self.assertAlmostEqual(result['metrics_mm']['diameter'],80.3,delta=.02)
        self.assertTrue(result['measurement_valid'])
        self.assertEqual(result['valid'].size,720)
        self.assertTrue(result['calibration']['is_simulated'])
    def test_offset_center_and_periodic_seam(self):
        result=run_ring_measurement(ring_image(),self.config((123,129)),None)
        self.assertTrue(result['valid'][:12].all() and result['valid'][-12:].all())
        self.assertAlmostEqual(result['metrics_px']['diameter'],160.6,delta=.04)
        self.assertIsNone(result['metrics_mm'])
    def test_auto_center(self):
        result=run_ring_measurement(ring_image(noise=3),RingMeasurementConfig(angular_samples=720),None)
        np.testing.assert_allclose(result['grid'].center_xy,[127.2,126.6],atol=.2)
        self.assertGreater(result['coverage'],.98)
    def test_gap_across_seam_never_interpolated(self):
        result=run_ring_measurement(ring_image(gap=True),self.config(),None)
        self.assertFalse(result['valid'][0]); self.assertFalse(result['valid'][-1])
        self.assertLess(result['coverage'],.95); self.assertFalse(result['measurement_valid'])
        self.assertTrue(np.isnan(result['points_px'][~result['valid']]).all())
    def test_blank_and_mask_have_no_ring(self):
        result=run_ring_measurement(np.ones((256,256))*20,self.config(),None)
        self.assertEqual(result['coverage'],0); self.assertIsNone(result['metrics_px'])
        result=run_ring_measurement(ring_image(),self.config(),None,source_mask=np.zeros((256,256),bool))
        self.assertEqual(result['coverage'],0)

class SweepTests(unittest.TestCase):
    def test_encoder_already_in_world_coordinates_not_added_twice(self):
        sections=[dict(position_mm=z,center_world_mm=[0,0,z],metrics_mm=dict(diameter=100),measurement_valid=True,
                       calibration=dict(frame_id='fixture',pose_reference='encoder_position')) for z in [0,10,20]]
        result=analyze_sweep(sections)
        np.testing.assert_allclose(result['centers_world_mm'][:,2],[0,10,20])
    def test_collapsed_section_axis_rejected(self):
        sections=[dict(position_mm=z,center_world_mm=[0,0,-z],metrics_mm=dict(diameter=100),measurement_valid=True,
                       calibration=dict(frame_id='fixture')) for z in [0,10,20]]
        with self.assertRaises(ValueError): analyze_sweep(sections)
    def test_fixed_axis_taper_and_straightness(self):
        sections=[dict(position_mm=z,center_world_mm=[.1*z/6000,0,0],metrics_mm=dict(diameter=100+.01*z/6000),measurement_valid=True,calibration=dict(frame_id='fixture',is_simulated=True)) for z in [0,3000,6000]]
        result=analyze_sweep(sections,reference_origin_mm=[0,0,0],reference_direction=[0,0,1])
        self.assertAlmostEqual(result['taper_mm_over_span'],.01,places=8)
        self.assertAlmostEqual(result['coaxiality_diameter_mm'],.2,places=8)
        self.assertLess(result['straightness_ls_mm'],1e-8)
    def test_incompatible_or_invalid_sections_rejected(self):
        section=dict(position_mm=0,center_world_mm=[0,0,0],metrics_mm=dict(diameter=100),measurement_valid=True,calibration=dict(frame_id='a'))
        other=dict(section,position_mm=10,calibration=dict(frame_id='b'))
        with self.assertRaises(ValueError): analyze_sweep([section,other])
        with self.assertRaises(ValueError): analyze_sweep([section,dict(other,measurement_valid=False)])

class FastV5Tests(unittest.TestCase):
    def test_fast_bilinear_preserves_boundaries_and_nonfinite_samples(self):
        from ring_measurement.fast_v5 import bilinear_sample
        from centerline.normal_guided_centroid import bilinear_sample as reference
        image=np.random.default_rng(31).normal(size=(21,29)).astype(np.float32)
        xs=np.array([-1,.3,27.9,28,29,np.nan],np.float32)
        ys=np.array([3.3,9.2,19.6,20,4,np.inf],np.float32)
        for count in [5,6]:
            np.testing.assert_array_equal(bilinear_sample(image,xs[:count],ys[:count]),reference(image,xs[:count],ys[:count]))
    def test_precomputed_dp_matches_original_on_ambiguous_profiles(self):
        from ring_measurement.fast_v5 import precomputed_ridge_peaks
        from centerline.ridge_support import continuous_ridge_peaks
        rng=np.random.default_rng(32)
        for _ in range(12):
            profiles=rng.uniform(0,1,(80,41))
            np.testing.assert_array_equal(precomputed_ridge_peaks(profiles),continuous_ridge_peaks(profiles))
    def test_certified_ridge_path_matches_full_dynamic_program(self):
        from ring_measurement.fast_v5 import certified_ridge_peaks
        from centerline.ridge_support import continuous_ridge_peaks
        rng=np.random.default_rng(19); y=np.arange(41)[None,:]
        for amplitude,noise in [(160,1),(50,12)]:
            center=20+3*np.sin(np.arange(180)[:,None]/30)
            signal=amplitude*np.exp(-.5*((y-center)/2.6)**2)+rng.normal(0,noise,(180,41))
            np.testing.assert_array_equal(certified_ridge_peaks(signal),continuous_ridge_peaks(signal))
    def test_vectorized_quantile_matches_numpy_with_missing_samples(self):
        from ring_measurement.fast_v5 import row_nanpercentile
        data=np.random.default_rng(21).normal(size=(90,37)).astype(np.float32)
        data[3,2:18]=np.nan; data[8,1:]=np.nan
        for q in [15,30,50]:
            np.testing.assert_allclose(row_nanpercentile(data,q,axis=1),np.nanpercentile(data,q,axis=1),rtol=1e-6,atol=1e-6)
    def test_vectorized_convolution_matches_v5(self):
        from ring_measurement.fast_v5 import convolve_axis
        from centerline.normal_guided_steger import convolve_axis as reference
        data=np.random.default_rng(21).normal(size=(37,49)).astype(np.float32)
        for axis in [0,1]:
            for kernel in [[.1,.2,.4,.2,.1],[-.2,0,.2]]:
                np.testing.assert_allclose(convolve_axis(data,np.array(kernel,np.float32),axis),reference(data,np.array(kernel,np.float32),axis),atol=1e-6)
    def test_ring_backend_matches_original_points_and_validity(self):
        from ring_measurement.fast_v5 import run_ring_v5
        from centerline.pipeline import RealtimePipelineConfig,run_realtime_centerline_pipeline
        grid=PolarGrid((127.2,126.6),55,105,360,1)
        image=np.pad(unwrap_ring(ring_image(noise=5,gap=True),grid).image,((0,0),(32,32)),mode='wrap')
        config=RealtimePipelineConfig(y_min=1,y_max=50)
        before=run_realtime_centerline_pipeline(image,config)['selected']
        after=run_ring_v5(image,config,np.ones_like(image,bool))['selected']
        np.testing.assert_array_equal(before['valid'],after['valid'])
        np.testing.assert_allclose(before['estimated_x'],after['estimated_x'],atol=2e-5,equal_nan=True)
        np.testing.assert_allclose(before['estimated_y'],after['estimated_y'],atol=2e-5,equal_nan=True)

if __name__=='__main__': unittest.main()
