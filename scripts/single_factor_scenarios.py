"""Isolated factors around one paired clean/noisy baseline.

Curved stripe intensity uses distance to the true polyline, keeping normal width
fixed. Rendering cost is separate from the extraction algorithm.
"""
from functools import lru_cache
import numpy as np
from realtime_scenarios import Scenario, IMAGE_SIZE
from line_grayscale_centroid_demo import normalize_to_uint8

PARAMETERS = dict(background=20.0, main_amplitude=160.0, normal_sigma_px=2.6,
                  noise_sigma=5.0, center_y=255.0,
                  reflection_x=[100.0,250.0,400.0], reflection_y_offset=4.0,
                  reflection_amplitude=60.0, reflection_sigma_x=14.0,
                  reflection_sigma_y=7.0, secondary_offset=8.0,
                  secondary_amplitude=24.0, secondary_sigma=2.0,
                  saturation_amplitude=280.0,
                  weak_light_ratio=0.42, enhanced_noise_sigma=18.0,
                  broken_segments=[[125,150],[315,345]],
                  mild_curve_equation='255+46*sin((x-30)/68)+22*sin(x/23)',
                  curvature_equation='255+62*sin((x-25)/55)+34*sin(x/18)+13*sin(x/7)')

def _curve_y(x):
    return 255.0+62.0*np.sin((x-25.0)/55.0)+34.0*np.sin(x/18.0)+13.0*np.sin(x/7.0)

def _mild_curve_y(x):
    return 255.0+46.0*np.sin((x-30.0)/68.0)+22.0*np.sin(x/23.0)

def _distance_to_curve(yy, xx, curve_function=_curve_y):
    x_nodes = np.arange(-16, IMAGE_SIZE+16, dtype=np.float32)
    y_nodes = curve_function(x_nodes)
    ax, ay = x_nodes[:-1], y_nodes[:-1]
    dx, dy = np.diff(x_nodes), np.diff(y_nodes)
    length_squared = dx*dx+dy*dy
    out = np.empty(xx.size, dtype=np.float32)
    flat_x, flat_y = xx.ravel(), yy.ravel()
    for start in range(0, xx.size, 4096):
        end = min(start+4096, xx.size)
        vx, vy = flat_x[start:end,None]-ax, flat_y[start:end,None]-ay
        t = np.clip((vx*dx+vy*dy)/length_squared, 0.0, 1.0)
        squared = (vx-t*dx)**2+(vy-t*dy)**2
        out[start:end] = np.sqrt(squared.min(axis=1))
    return out.reshape(xx.shape)

@lru_cache(maxsize=1)
def _clean_components():
    yy, xx = np.mgrid[:IMAGE_SIZE,:IMAGE_SIZE].astype(np.float32)
    straight = np.full(IMAGE_SIZE,255.0,dtype=np.float32)
    curve = _curve_y(np.arange(IMAGE_SIZE,dtype=np.float32))
    main = 160.0*np.exp(-0.5*((yy-255.0)/2.6)**2)
    curved_main = 160.0*np.exp(-0.5*(_distance_to_curve(yy,xx)/2.6)**2)
    mild_curve = _mild_curve_y(np.arange(IMAGE_SIZE,dtype=np.float32))
    mild_main = 160.0*np.exp(-0.5*(_distance_to_curve(yy,xx,_mild_curve_y)/2.6)**2)
    broken_main = main.copy()
    for start,end in PARAMETERS['broken_segments']:
        broken_main[:,start:end] = 0.0
    glare = np.zeros_like(main)
    for center in [100.0,250.0,400.0]:
        glare += 60.0*np.exp(-0.5*(((xx-center)/14.0)**2+((yy-259.0)/7.0)**2))
    secondary = PARAMETERS['secondary_amplitude']*np.exp(-0.5*((yy-263.0)/2.0)**2)
    saturated_main = PARAMETERS['saturation_amplitude']*np.exp(-0.5*((yy-255.0)/2.6)**2)
    return [('control',main,straight),('curved_only',mild_main,mild_curve),
            ('curvature_only',curved_main,curve),
            ('weak_light_only',main*PARAMETERS['weak_light_ratio'],straight),
            ('noise_only',main,straight),
            ('reflection_only',main+glare,straight),('double_peak_only',main+secondary,straight),
            ('saturation_only',saturated_main,straight),('broken_only',broken_main,straight)]

def build_single_factor_scenarios(seed=7, noise_sigma=5.0):
    if not np.isfinite(noise_sigma) or noise_sigma<0:
        raise ValueError('noise_sigma must be nonnegative and finite')
    noise = np.random.default_rng(seed).normal(0.0,noise_sigma,(IMAGE_SIZE,IMAGE_SIZE))
    stronger_noise = noise * (PARAMETERS['enhanced_noise_sigma']/PARAMETERS['noise_sigma'])
    return [Scenario(name,normalize_to_uint8(20.0+signal+(stronger_noise if name=='noise_only' else noise)),truth.copy())
            for name,signal,truth in _clean_components()]
