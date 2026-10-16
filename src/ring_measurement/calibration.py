"""Explicit physical calibration; no image-only guess of millimetres."""
from dataclasses import dataclass,field
import numpy as np

def _points(points):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2:
        raise ValueError('points must have shape (N,2)')
    return p

def _pose(rotation,translation):
    r,t=np.asarray(rotation,dtype=float),np.asarray(translation,dtype=float)
    if r.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError('pose requires a finite 3x3 rotation and 3-vector translation')
    if not np.allclose(r.T@r,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(r),1,atol=1e-8):
        raise ValueError('rotation must be a proper orthonormal matrix')
    return r,t

def _metadata(frame_id,is_simulated,image_size,pose_reference):
    if not isinstance(frame_id,str) or not frame_id.strip() or not isinstance(is_simulated,bool):
        raise ValueError('nonempty frame_id and explicit boolean simulation flag required')
    if pose_reference not in {'zero_encoder','encoder_position'}:
        raise ValueError('pose_reference must identify whether encoder motion is already included')
    if image_size is not None:
        values=np.asarray(image_size)
        if values.shape!=(2,) or not np.isfinite(values).all() or np.any(values<=0) or np.any(values!=np.floor(values)):
            raise ValueError('image_size requires positive integer width,height')

@dataclass(frozen=True)
class CameraLaserCalibration:
    camera_matrix: object
    plane_normal: object
    plane_offset_mm: float
    distortion: object=(0.,0.,0.,0.,0.)
    rotation_world_from_camera: object=field(default_factory=lambda:np.eye(3))
    translation_world_mm: object=(0.,0.,0.)
    image_size: tuple[int,int]|None=None
    frame_id: str='camera'
    is_simulated: bool=False
    source: str='provided_camera_and_laser_plane'
    pose_reference: str='zero_encoder'

    def __post_init__(self):
        k=np.asarray(self.camera_matrix,dtype=float)
        n=np.asarray(self.plane_normal,dtype=float)
        d=np.asarray(self.distortion,dtype=float)
        if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]) or not np.isclose(k[1,0],0):
            raise ValueError('invalid pinhole camera matrix')
        if n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)<=1e-12 or not np.isfinite(self.plane_offset_mm) or abs(self.plane_offset_mm)<=1e-12:
            raise ValueError('laser plane must have a finite nonzero normal and offset in mm')
        if d.shape!=(5,) or not np.isfinite(d).all():
            raise ValueError('distortion order is k1,k2,p1,p2,k3')
        r,t=_pose(self.rotation_world_from_camera,self.translation_world_mm)
        _metadata(self.frame_id,self.is_simulated,self.image_size,self.pose_reference)
        length=np.linalg.norm(n)
        for name,value in [('camera_matrix',k.copy()),('plane_normal',n/length),('plane_offset_mm',float(self.plane_offset_mm/length)),('distortion',d.copy()),('rotation_world_from_camera',r.copy()),('translation_world_mm',t.copy())]:
            object.__setattr__(self,name,value)

    def _normalize(self,pixels):
        p=_points(pixels)
        homogeneous=np.column_stack([p,np.ones(len(p))])@np.linalg.inv(self.camera_matrix).T
        return homogeneous[:,:2]
    def _pixels(self,normalized):
        p=np.column_stack([normalized,np.ones(len(normalized))])@self.camera_matrix.T
        return p[:,:2]
    def _distort_normal(self,p):
        x,y=p[:,0],p[:,1]; k1,k2,p1,p2,k3=self.distortion
        r2=x*x+y*y; radial=1+k1*r2+k2*r2**2+k3*r2**3
        return np.column_stack([x*radial+2*p1*x*y+p2*(r2+2*x*x),y*radial+p1*(r2+2*y*y)+2*p2*x*y])
    def distort_points(self,undistorted_pixels):
        if not np.any(self.distortion): return _points(undistorted_pixels).copy()
        return self._pixels(self._distort_normal(self._normalize(undistorted_pixels)))
    def undistort_points(self,distorted_pixels):
        if not np.any(self.distortion): return _points(distorted_pixels).copy()
        target=self._normalize(distorted_pixels); p=target.copy()
        k1,k2,p1,p2,k3=self.distortion
        for _ in range(20):
            x,y=p[:,0],p[:,1]; r2=x*x+y*y
            radial=1+k1*r2+k2*r2**2+k3*r2**3
            derivative=k1+2*k2*r2+3*k3*r2**2
            rx,ry=2*x*derivative,2*y*derivative
            a=radial+x*rx+2*p1*y+6*p2*x
            b=x*ry+2*p1*x+2*p2*y
            c=y*rx+2*p1*x+2*p2*y
            d=radial+y*ry+6*p1*y+2*p2*x
            residual=self._distort_normal(p)-target
            det=a*d-b*c
            safe=np.isfinite(det)&(np.abs(det)>1e-12)
            delta=np.full_like(p,np.nan)
            delta[safe,0]=(d[safe]*residual[safe,0]-b[safe]*residual[safe,1])/det[safe]
            delta[safe,1]=(-c[safe]*residual[safe,0]+a[safe]*residual[safe,1])/det[safe]
            p-=delta
            if np.any(safe) and np.max(np.abs(delta[safe]))<1e-12: break
        good=np.isfinite(p).all(axis=1)&(np.linalg.norm(self._distort_normal(p)-target,axis=1)<1e-8)
        p[~good]=np.nan
        return self._pixels(p)
    def reconstruct(self,pixels,undistorted=False):
        normalized=self._normalize(pixels if undistorted else self.undistort_points(pixels))
        rays=np.column_stack([normalized,np.ones(len(normalized))])
        denom=rays@self.plane_normal
        valid=np.isfinite(rays).all(axis=1)&(np.abs(denom)>1e-10)
        distance=np.full(len(rays),np.nan)
        distance[valid]=-self.plane_offset_mm/denom[valid]
        valid &= distance>0
        points=rays*distance[:,None]; points[~valid]=np.nan
        return points,valid
    @property
    def basis(self):
        n=self.plane_normal
        reference=np.array([1.,0.,0.]) if abs(n[0])<.9 else np.array([0.,1.,0.])
        u=reference-n*np.dot(reference,n); u/=np.linalg.norm(u)
        return np.column_stack([u,np.cross(n,u)])
    @property
    def origin(self): return -self.plane_offset_mm*self.plane_normal
    def to_plane(self,pixels,undistorted=False):
        points,valid=self.reconstruct(pixels,undistorted)
        return (points-self.origin)@self.basis,valid
    def world_from_plane(self,points):
        camera=_points(points)@self.basis.T+self.origin
        return camera@self.rotation_world_from_camera.T+self.translation_world_mm
    @property
    def section_normal_world(self): return self.rotation_world_from_camera@self.plane_normal
    def to_dict(self):
        return dict(model='camera_laser_plane',camera_matrix=self.camera_matrix.tolist(),plane_normal=self.plane_normal.tolist(),plane_offset_mm=self.plane_offset_mm,distortion=self.distortion.tolist(),rotation_world_from_camera=self.rotation_world_from_camera.tolist(),translation_world_mm=self.translation_world_mm.tolist(),image_size=self.image_size,frame_id=self.frame_id,is_simulated=self.is_simulated,source=self.source,pose_reference=self.pose_reference)

@dataclass(frozen=True)
class PlanarScaleCalibration:
    mm_per_px: float
    origin_px: object=(0.,0.)
    frame_id: str='measurement_plane'
    is_simulated: bool=False
    source: str='planar_scale'
    gauge_rmse_mm: float|None=None
    image_size: tuple[int,int]|None=None
    pose_reference: str='zero_encoder'
    def __post_init__(self):
        if not np.isfinite(self.mm_per_px) or self.mm_per_px<=0 or np.asarray(self.origin_px).shape!=(2,) or not np.isfinite(self.origin_px).all():
            raise ValueError('positive scale and finite pixel origin required')
        _metadata(self.frame_id,self.is_simulated,self.image_size,self.pose_reference)
    def distort_points(self,points): return _points(points).copy()
    def undistort_points(self,points): return _points(points).copy()
    def to_plane(self,points,undistorted=False):
        p=(_points(points)-self.origin_px)*self.mm_per_px
        return p,np.isfinite(p).all(axis=1)
    def world_from_plane(self,points): return np.column_stack([_points(points),np.zeros(len(points))])
    @property
    def section_normal_world(self): return np.array([0.,0.,1.])
    def to_dict(self):
        return dict(model='planar_scale',mm_per_px=self.mm_per_px,origin_px=list(self.origin_px),frame_id=self.frame_id,is_simulated=self.is_simulated,source=self.source,gauge_rmse_mm=self.gauge_rmse_mm,image_size=self.image_size,pose_reference=self.pose_reference)

def fit_scale_from_gauges(pixel_diameters,known_diameters_mm,**kwargs):
    x,y=np.asarray(pixel_diameters,dtype=float),np.asarray(known_diameters_mm,dtype=float)
    if x.ndim!=1 or x.size==0 or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x<=0) or np.any(y<=0):
        raise ValueError('matched positive finite gauge diameters required')
    scale=float(np.dot(x,y)/np.dot(x,x))
    return PlanarScaleCalibration(scale,gauge_rmse_mm=float(np.sqrt(np.mean((scale*x-y)**2))),source='ring_gauge_zero_intercept_scale',**kwargs)

def calibration_from_dict(data):
    values=dict(data); model=values.pop('model',None)
    if model=='camera_laser_plane': return CameraLaserCalibration(**values)
    if model=='planar_scale': return PlanarScaleCalibration(**values)
    raise ValueError('unknown calibration model')
