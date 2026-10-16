"""Full annular measurement wrapper; V5 source modules remain unchanged."""
from dataclasses import dataclass,field,replace
import time
import numpy as np
from centerline.pipeline import RealtimePipelineConfig,run_realtime_centerline_pipeline
from centerline.preprocess import to_grayscale_float32
from .polar import PolarGrid,unwrap_ring,polar_to_cartesian,sample_image
from .geometry import fit_circle,contour_metrics
from .fast_v5 import run_ring_v5

@dataclass(frozen=True)
class RingMeasurementConfig:
    grid: PolarGrid|None=None
    angular_samples: int=3600
    radial_step: float=1.0
    roi_margin_px: float=18.0
    periodic_padding: int=32
    min_coverage: float=.95
    max_gap_deg: float=5.0
    nominal_diameter_mm: float|None=None
    accelerated_v5: bool=True
    centerline: RealtimePipelineConfig=field(default_factory=RealtimePipelineConfig)
    def __post_init__(self):
        if not 0<self.min_coverage<=1 or not 0<=self.max_gap_deg<=360:
            raise ValueError('invalid coverage or gap limit')
        if self.periodic_padding<16 or int(self.periodic_padding)!=self.periodic_padding:
            raise ValueError('periodic_padding must be an integer >=16')
        if not np.isfinite(self.roi_margin_px) or self.roi_margin_px<8:
            raise ValueError('annular ROI margin must be at least 8 pixels')

def estimate_ring_grid(image,config,calibration=None,source_mask=None):
    gray=to_grayscale_float32(image)
    background=float(np.median(gray))
    upper=float(np.percentile(gray,99.8))
    sigma=1.4826*float(np.median(np.abs(gray-background)))
    if upper-background<max(5.,6*sigma): raise ValueError('no sufficiently strong annular signal')
    mask=gray>background+.45*(upper-background)
    if source_mask is not None:
        if np.asarray(source_mask).shape!=gray.shape: raise ValueError('source_mask must match image')
        mask &= np.asarray(source_mask,bool)
    y,x=np.nonzero(mask)
    if len(x)<24: raise ValueError('not enough ring signal')
    points=np.column_stack([x,y]).astype(float)
    if calibration is not None: points=calibration.undistort_points(points)
    points=points[np.isfinite(points).all(axis=1)]
    fit=fit_circle(points)
    radius=np.linalg.norm(points[fit.inliers]-fit.center,axis=1)
    if fit.radius<8 or np.std(radius)>fit.radius*.3: raise ValueError('signal is not a compact ring')
    angle=np.mod(np.arctan2(points[:,1]-fit.center[1],points[:,0]-fit.center[0]),2*np.pi)
    if np.unique(np.floor(angle*36/(2*np.pi)).astype(int)).size<27:
        raise ValueError('insufficient angular support for automatic ring localization')
    return PolarGrid(tuple(fit.center),max(0,float(np.percentile(radius,1))-config.roi_margin_px),
                     float(np.percentile(radius,99))+config.roi_margin_px,config.angular_samples,config.radial_step)

def maximum_circular_gap_deg(valid):
    valid=np.asarray(valid,bool)
    if valid.all(): return 0.
    if not valid.any(): return 360.
    indices=np.flatnonzero(valid)
    gaps=np.diff(np.r_[indices,indices[0]+len(valid)])-1
    return float(gaps.max()*360/len(valid))

def run_ring_measurement(image,config=None,calibration=None,source_mask=None):
    config=config or RingMeasurementConfig()
    timings={}; start=time.perf_counter()
    gray=to_grayscale_float32(image)
    if gray.size==0: raise ValueError('empty image')
    if calibration is not None and calibration.image_size is not None and tuple(calibration.image_size)!=(gray.shape[1],gray.shape[0]):
        raise ValueError('image dimensions do not match calibration')
    grid=config.grid or estimate_ring_grid(image,config,calibration,source_mask)
    timings['localization_ms']=(time.perf_counter()-start)*1000
    phase=time.perf_counter()
    polar=unwrap_ring(image,grid,source_mask=source_mask,distort_points=None if calibration is None else calibration.distort_points)
    if polar.image.shape[0]<17: raise ValueError('annular ROI requires at least 17 radial rows for V5')
    halo=min(config.periodic_padding,grid.angular_samples)
    padded=np.pad(polar.image,((0,0),(halo,halo)),mode='wrap')
    observed=np.pad(polar.valid_mask,((0,0),(halo,halo)),mode='wrap')
    timings['unwrap_ms']=(time.perf_counter()-phase)*1000
    phase=time.perf_counter()
    core_config=replace(config.centerline,roi=None,y_min=1,y_max=polar.image.shape[0]-1)
    line=run_ring_v5(padded,core_config,observed,accelerated=config.accelerated_v5)
    selected=line['selected']; keep=slice(halo,halo+grid.angular_samples)
    uv=np.column_stack([selected['estimated_x'][keep].astype(float)-halo,selected['estimated_y'][keep].astype(float)])
    valid=selected['valid'][keep].copy()&np.isfinite(uv).all(axis=1)
    support,supported=sample_image(observed.astype(np.float32),uv[:,0]+halo,uv[:,1])
    valid &= supported&(support>=1-1e-6)&(uv[:,1]>=1)&(uv[:,1]<polar.image.shape[0]-1)
    # Missing flank data must not be replaced by a background-filled image.
    # Require observed samples throughout the actual normal search support.
    offsets=np.arange(-core_config.normal.normal_half_width,core_config.normal.normal_half_width+1,1)
    normal_x=line['guided']['normal_x'][keep]
    normal_y=line['guided']['normal_y'][keep]
    support_x=uv[:,0,None]+halo+normal_x[:,None]*offsets
    support_y=uv[:,1,None]+normal_y[:,None]*offsets
    if observed.all():
        valid &= np.all((support_x>=0)&(support_x<=observed.shape[1]-1)&(support_y>=0)&(support_y<=observed.shape[0]-1),axis=1)
    else:
        flank_support,flank_inside=sample_image(observed.astype(np.float32),support_x,support_y)
        valid &= np.all(flank_inside&(flank_support>=1-1e-6),axis=1)
    if not observed.all():
        # Hessian kernels use a 2-D footprint, not just the normal line.
        radius=max(int(np.ceil(3*max(core_config.steger.sigma,core_config.steger.sigma_max)))+2,
                   core_config.normal.mean_window//2+2)
        missing=np.pad((~observed).astype(np.int32),((radius,radius),(radius,radius)),constant_values=1)
        prefix=np.pad(missing.cumsum(axis=0).cumsum(axis=1),((1,0),(1,0)))
        size=2*radius+1
        count=prefix[size:,size:]-prefix[:-size,size:]-prefix[size:,:-size]+prefix[:-size,:-size]
        complete=(count==0).astype(np.float32)
        footprint,footprint_inside=sample_image(complete,uv[:,0]+halo,uv[:,1])
        valid &= footprint_inside&(footprint>=1-1e-6)
    timings['centerline_ms']=(time.perf_counter()-phase)*1000
    phase=time.perf_counter()
    corrected=polar_to_cartesian(uv,grid)
    raw=corrected.copy() if calibration is None else calibration.distort_points(corrected)
    _,source_valid=sample_image(image,raw[:,0],raw[:,1],source_mask)
    valid &= source_valid
    points_mm=None; world=None; center_world=None; metrics_mm=None
    if calibration is not None:
        points_mm,physical_valid=calibration.to_plane(corrected,undistorted=True)
        valid &= physical_valid
        points_mm[~valid]=np.nan
        world=calibration.world_from_plane(points_mm)
    uv[~valid]=np.nan; raw[~valid]=np.nan; corrected[~valid]=np.nan
    timings['backtransform_calibration_ms']=(time.perf_counter()-phase)*1000
    phase=time.perf_counter(); metrics_px=None; reasons=[]
    if valid.sum()>=3:
        try:
            metrics_px=contour_metrics(corrected[valid])
            if calibration is not None:
                metrics_mm=contour_metrics(points_mm[valid],config.nominal_diameter_mm)
                center_world=calibration.world_from_plane(np.asarray(metrics_mm['circle_center'])[None])[0]
        except ValueError as error: reasons.append(str(error))
    else: reasons.append('fewer_than_three_valid_points')
    coverage=float(valid.mean()); gap=maximum_circular_gap_deg(valid)
    if coverage<config.min_coverage: reasons.append('insufficient_angular_coverage')
    if gap>config.max_gap_deg: reasons.append('angular_gap_exceeds_limit')
    if metrics_px is None: reasons.append('pixel_geometry_unavailable')
    if calibration is not None and metrics_mm is None: reasons.append('physical_geometry_unavailable')
    timings['geometry_ms']=(time.perf_counter()-phase)*1000
    timings['total_ms']=(time.perf_counter()-start)*1000
    return dict(grid=grid,unwrapped=polar.image,unwrapped_valid=polar.valid_mask,polar_points=uv,
                points_px=raw,undistorted_points_px=corrected,valid=valid,points_mm=points_mm,
                points_world_mm=world,center_world_mm=center_world,metrics_px=metrics_px,metrics_mm=metrics_mm,
                calibration=None if calibration is None else calibration.to_dict(),
                section_normal_world=None if calibration is None else calibration.section_normal_world,
                coverage=coverage,valid_points=int(valid.sum()),angular_samples=grid.angular_samples,
                max_gap_deg=gap,measurement_valid=not reasons,quality_reasons=reasons,
                steger_refined=int((selected['steger_refined'][keep]&valid).sum()) if 'steger_refined' in selected else 0,
                timings_ms=timings,frame_budget_ms=core_config.frame_budget_ms,
                deadline_met=timings['total_ms']<=core_config.frame_budget_ms)
