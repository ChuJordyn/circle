"""Analysis in one fixed datum; encoder displacement is not recentering."""
import numpy as np

def analyze_sweep(sections,reference_origin_mm=None,reference_direction=(0,0,1),nominal_diameter_mm=None):
    if len(sections)<2: raise ValueError('at least two calibrated sections required')
    if any(not s.get('measurement_valid') or s.get('metrics_mm') is None or s.get('center_world_mm') is None for s in sections):
        raise ValueError('all sections must be valid and physically calibrated')
    frames={s['calibration']['frame_id'] for s in sections}
    if len(frames)!=1 or not next(iter(frames)): raise ValueError('sections must share an explicit coordinate frame')
    positions=np.array([s['position_mm'] for s in sections],float)
    direction=np.asarray(reference_direction,dtype=float)
    if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-12:
        raise ValueError('reference direction must be a finite nonzero 3-vector')
    direction=direction/np.linalg.norm(direction)
    if not np.isfinite(positions).all() or len(np.unique(positions))!=len(positions):
        raise ValueError('encoder positions must be finite and distinct')
    if any('section_normal_world' in s and abs(np.dot(s['section_normal_world'],direction))<1-1e-6 for s in sections):
        raise ValueError('section planes must be perpendicular to the encoder axis')
    references={s['calibration'].get('pose_reference','zero_encoder') for s in sections}
    if len(references)!=1 or not references.issubset({'zero_encoder','encoder_position'}):
        raise ValueError('all poses must share a known encoder reference convention')
    centers=np.asarray([s['center_world_mm'] for s in sections],float)
    if 'zero_encoder' in references:
        centers=centers+positions[:,None]*direction
    if centers.shape!=(len(sections),3) or not np.isfinite(centers).all(): raise ValueError('invalid section centers')
    origin=centers[np.argmin(positions)] if reference_origin_mm is None else np.asarray(reference_origin_mm,float)
    if origin.shape!=(3,) or not np.isfinite(origin).all(): raise ValueError('invalid reference origin')
    relative=centers-origin
    offsets=relative-(relative@direction)[:,None]*direction
    diameters=np.array([s['metrics_mm']['diameter'] for s in sections],float)
    if not np.isfinite(diameters).all() or np.any(diameters<=0): raise ValueError('invalid section diameter')
    design=np.column_stack([positions-positions.mean(),np.ones(len(positions))])
    diameter_slope=np.linalg.lstsq(design,diameters,rcond=None)[0][0]
    line=design@np.linalg.lstsq(design,centers,rcond=None)[0]
    line_direction=np.linalg.lstsq(design,centers,rcond=None)[0][0]
    if not np.isfinite(line_direction).all() or np.linalg.norm(line_direction)<1e-12:
        raise ValueError('section locations collapse to a degenerate axis')
    line_direction/=np.linalg.norm(line_direction)
    residual=centers-line
    perpendicular=residual-(residual@line_direction)[:,None]*line_direction
    result=dict(frame_id=next(iter(frames)),pose_reference=next(iter(references)),position_mm=positions,diameter_mm=diameters,
                centers_world_mm=centers,reference_origin_mm=origin,reference_direction=direction,
                datum='provided_fixed_axis' if reference_origin_mm is not None else 'first_section_fixed_axis',
                center_axis_offsets_mm=np.linalg.norm(offsets,axis=1),
                coaxiality_diameter_mm=float(2*np.max(np.linalg.norm(offsets,axis=1))),
                straightness_ls_mm=float(2*np.max(np.linalg.norm(perpendicular,axis=1))),
                straightness_method='twice_max_center_distance_to_least_squares_axis',
                taper_mm_per_m=float(diameter_slope*1000),
                taper_mm_over_span=float(abs(diameter_slope)*np.ptp(positions)),
                maximum_diameter_mm=float(diameters.max()),minimum_diameter_mm=float(diameters.min()),
                is_simulated=any(s['calibration'].get('is_simulated',False) for s in sections))
    if nominal_diameter_mm is not None:
        if not np.isfinite(nominal_diameter_mm) or nominal_diameter_mm<=0: raise ValueError('positive nominal diameter required')
        result['diameter_deviation_mm']=diameters-nominal_diameter_mm
    return result
