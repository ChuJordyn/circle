"""Angle columns, radius rows; image coordinates use x right and y down."""
from dataclasses import dataclass
import numpy as np
from centerline.preprocess import to_grayscale_float32

@dataclass(frozen=True)
class PolarGrid:
    center_xy: tuple[float,float]
    radius_min: float
    radius_max: float
    angular_samples: int=3600
    radial_step: float=1.0
    angle_start_rad: float=0.0

    def __post_init__(self):
        if np.asarray(self.center_xy).shape!=(2,) or not np.isfinite(self.center_xy).all():
            raise ValueError('center_xy must be two finite coordinates')
        if not np.isfinite([self.radius_min,self.radius_max,self.radial_step,self.angle_start_rad]).all():
            raise ValueError('grid parameters must be finite')
        if self.radius_min<0 or self.radius_max<=self.radius_min or self.radial_step<=0:
            raise ValueError('require 0 <= radius_min < radius_max and positive radial_step')
        if isinstance(self.angular_samples,bool) or int(self.angular_samples)!=self.angular_samples or self.angular_samples<8:
            raise ValueError('angular_samples must be an integer >= 8')
        if self.radius_count<2:
            raise ValueError('grid requires at least two radial samples')

    @property
    def radius_count(self):
        return int(np.floor((self.radius_max-self.radius_min)/self.radial_step+1e-10))+1
    @property
    def radii(self):
        return self.radius_min+np.arange(self.radius_count)*self.radial_step
    @property
    def angles(self):
        return self.angle_start_rad+np.arange(self.angular_samples)*(2*np.pi/self.angular_samples)

@dataclass(frozen=True)
class PolarImage:
    image: np.ndarray
    valid_mask: np.ndarray
    source_x: np.ndarray
    source_y: np.ndarray
    grid: PolarGrid

def polar_to_cartesian(points,grid):
    uv=np.asarray(points,dtype=np.float64)
    if uv.ndim!=2 or uv.shape[1]!=2:
        raise ValueError('points must have shape (N,2)')
    angle=grid.angle_start_rad+uv[:,0]*(2*np.pi/grid.angular_samples)
    radius=grid.radius_min+uv[:,1]*grid.radial_step
    return np.column_stack([grid.center_xy[0]+radius*np.cos(angle),grid.center_xy[1]+radius*np.sin(angle)])

def cartesian_to_polar(points,grid):
    xy=np.asarray(points,dtype=np.float64)
    if xy.ndim!=2 or xy.shape[1]!=2:
        raise ValueError('points must have shape (N,2)')
    d=xy-np.asarray(grid.center_xy)
    angle=np.mod(np.arctan2(d[:,1],d[:,0])-grid.angle_start_rad,2*np.pi)
    return np.column_stack([angle*grid.angular_samples/(2*np.pi),(np.hypot(d[:,0],d[:,1])-grid.radius_min)/grid.radial_step])

def sample_image(image,xs,ys,source_mask=None):
    """Full-precision bilinear sampling plus observed-source support mask."""
    original=np.asarray(image)
    gray=original.astype(np.float32,copy=False) if original.ndim==2 else to_grayscale_float32(image)
    if gray.size==0:
        raise ValueError('image must not be empty')
    h,w=gray.shape
    xs,ys=np.asarray(xs,dtype=float),np.asarray(ys,dtype=float)
    valid=np.isfinite(xs)&np.isfinite(ys)&(xs>=0)&(xs<=w-1)&(ys>=0)&(ys<=h-1)
    sx=np.clip(np.nan_to_num(xs,nan=0,posinf=0,neginf=0),0,w-1)
    sy=np.clip(np.nan_to_num(ys,nan=0,posinf=0,neginf=0),0,h-1)
    x0,y0=np.floor(sx).astype(int),np.floor(sy).astype(int)
    x1,y1=np.minimum(x0+1,w-1),np.minimum(y0+1,h-1)
    dx,dy=sx-x0,sy-y0
    weights=[(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy]
    coords=[(y0,x0),(y0,x1),(y1,x0),(y1,x1)]
    values=sum(weight*gray[y,x] for weight,(y,x) in zip(weights,coords))
    finite_corners=[np.isfinite(original[y,x]).all(axis=-1) if original.ndim==3 else np.isfinite(original[y,x]) for y,x in coords]
    if source_mask is not None:
        mask=np.asarray(source_mask,dtype=bool)
        if mask.shape!=gray.shape:
            raise ValueError('source_mask must match image shape')
        finite_corners=[finite&mask[y,x] for finite,(y,x) in zip(finite_corners,coords)]
    support=sum(weight*finite for weight,finite in zip(weights,finite_corners))
    valid &= support>=1-1e-10
    if not valid.all():
        values=np.where(valid,values,float(np.nanmedian(gray)))
    return values.astype(np.float32),valid

def unwrap_ring(image,grid,source_mask=None,distort_points=None):
    radius=grid.radii[:,None]
    angle=grid.angles[None,:]
    x=grid.center_xy[0]+radius*np.cos(angle)
    y=grid.center_xy[1]+radius*np.sin(angle)
    if distort_points is not None:
        xy=distort_points(np.column_stack([x.ravel(),y.ravel()]))
        x,y=xy[:,0].reshape(x.shape),xy[:,1].reshape(y.shape)
    values,valid=sample_image(image,x,y,source_mask)
    return PolarImage(values,valid,x,y,grid)
