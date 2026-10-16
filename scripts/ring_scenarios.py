"""Single-factor ring images in the original Cartesian sensor plane."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ring_measurement.calibration import CameraLaserCalibration

CENTER=(255.3,254.7)
RADIUS=160.3
NAMES=['circle','ellipse','large_curvature','weak_light','noise','reflection','saturation','double_peak','broken']

def simulation_calibration():
    return CameraLaserCalibration(camera_matrix=[[800,0,CENTER[0]],[0,800,CENTER[1]],[0,0,1]],
             plane_normal=[0,0,1],plane_offset_mm=-400,image_size=(512,512),
             frame_id='simulated_fixture',is_simulated=True,source='known_synthetic_pinhole_and_plane')

def radius_truth(name,theta):
    theta=np.asarray(theta,float)
    if name=='ellipse':
        a,b=170.3,150.3; angle=theta-.4
        denom=b*b*np.cos(angle)**2+a*a*np.sin(angle)**2
        radius=a*b/np.sqrt(denom)
        derivative=-a*b*(a*a-b*b)*np.sin(angle)*np.cos(angle)/denom**1.5
        return radius,derivative
    if name=='large_curvature': return RADIUS+10*np.cos(12*theta),-120*np.sin(12*theta)
    return np.full(theta.shape,RADIUS),np.zeros(theta.shape)

def build_ring_scenarios(seed=7):
    y,x=np.mgrid[:512,:512]
    dx,dy=x-CENTER[0],y-CENTER[1]
    radius=np.hypot(dx,dy); theta=np.mod(np.arctan2(dy,dx),2*np.pi)
    standard_noise=np.random.default_rng(seed).normal(0,5,radius.shape)
    rows=[]
    for name in NAMES:
        target,derivative=radius_truth(name,theta)
        normal_distance=(radius-target)/np.sqrt(1+(derivative/np.maximum(target,1))**2)
        amplitude=280 if name=='saturation' else 67.2 if name=='weak_light' else 160
        signal=amplitude*np.exp(-.5*(normal_distance/2.6)**2)
        if name=='reflection':
            for angle in [.6,2.7,4.9]:
                cx=CENTER[0]+(RADIUS+4)*np.cos(angle); cy=CENTER[1]+(RADIUS+4)*np.sin(angle)
                outward=(x-cx)*np.cos(angle)+(y-cy)*np.sin(angle)
                tangent=-(x-cx)*np.sin(angle)+(y-cy)*np.cos(angle)
                signal+=60*np.exp(-.5*((outward/7)**2+(tangent/14)**2))
        if name=='double_peak': signal+=24*np.exp(-.5*((radius-RADIUS-8)/2)**2)
        if name=='broken': signal[(theta<.13)|(theta>2*np.pi-.13)|((theta>2.1)&(theta<2.22))]=0
        noise=standard_noise*(18/5) if name=='noise' else standard_noise
        image=np.clip(20+signal+noise,0,255).astype(np.uint8)
        rows.append(dict(name=name,seed=seed,image=image))
    return rows
