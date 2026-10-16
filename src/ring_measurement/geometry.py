"""Geometric circle fit and ellipse-constrained algebraic fit."""
from dataclasses import dataclass
import numpy as np

def _checked(points,minimum):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<minimum or not np.isfinite(p).all():
        raise ValueError(f'require at least {minimum} finite 2-D points')
    if np.linalg.matrix_rank(p-p.mean(axis=0))<2:
        raise ValueError('collinear points cannot determine a closed contour')
    return p

@dataclass(frozen=True)
class CircleFit:
    center: np.ndarray
    radius: float
    residuals: np.ndarray
    inliers: np.ndarray
    rmse: float

def fit_circle(points,robust=True):
    p=_checked(points,3)
    origin=np.median(p,axis=0)
    scale=float(np.median(np.linalg.norm(p-origin,axis=1)))
    if scale<=1e-12: raise ValueError('degenerate circle')
    q=(p-origin)/scale
    radial=np.linalg.norm(q,axis=1)
    mad=1.4826*np.median(np.abs(radial-np.median(radial)))
    keep=np.abs(radial-np.median(radial))<=max(6*mad,.05) if robust else np.ones(len(p),bool)
    if keep.sum()<3: raise ValueError('not enough circle support')
    design=np.column_stack([2*q[keep],np.ones(keep.sum())])
    solution=np.linalg.lstsq(design,np.sum(q[keep]**2,axis=1),rcond=None)[0]
    center=solution[:2]
    radius=np.sqrt(max(solution[2]+np.dot(center,center),1e-12))
    for _ in range(12):
        distance=np.linalg.norm(q-center,axis=1)
        residual=distance-radius
        if robust:
            median=np.median(residual[keep])
            sigma=1.4826*np.median(np.abs(residual[keep]-median))
            keep=np.abs(residual-median)<=max(6*sigma,.002)
        if keep.sum()<3: raise ValueError('not enough inliers for circle')
        jac=np.column_stack([(center-q[keep])/np.maximum(distance[keep,None],1e-12),-np.ones(keep.sum())])
        delta=np.linalg.lstsq(jac,-residual[keep],rcond=None)[0]
        center+=delta[:2]; radius+=delta[2]
        if np.linalg.norm(delta)<1e-12: break
    if radius<=0 or not np.isfinite(radius): raise ValueError('invalid circle radius')
    center=center*scale+origin; radius=float(radius*scale)
    residual=np.linalg.norm(p-center,axis=1)-radius
    return CircleFit(center,radius,residual,keep,float(np.sqrt(np.mean(residual[keep]**2))))

@dataclass(frozen=True)
class EllipseFit:
    center: np.ndarray
    semi_axes: np.ndarray
    angle_rad: float

def fit_ellipse(points):
    p=_checked(points,6)
    origin=p.mean(axis=0); scale=float(np.sqrt(np.mean((p-origin)**2)))
    if scale<=1e-12: raise ValueError('degenerate ellipse')
    x,y=((p-origin)/scale).T
    d1=np.column_stack([x*x,x*y,y*y]); d2=np.column_stack([x,y,np.ones(len(x))])
    s1=d1.T@d1; s2=d1.T@d2; s3=d2.T@d2
    try:
        transform=-np.linalg.solve(s3,s2.T)
        constraint=np.array([[0.,0.,2.],[0.,-1.,0.],[2.,0.,0.]])
        _,vectors=np.linalg.eig(np.linalg.solve(constraint,s1+s2@transform))
    except np.linalg.LinAlgError as error:
        raise ValueError('ellipse fit is singular') from error
    candidates=[]
    for vector in vectors.T:
        if np.max(np.abs(vector.imag))>1e-8: continue
        quadratic=vector.real
        a,b,c=quadratic
        if 4*a*c-b*b<=0: continue
        linear=transform@quadratic
        candidates.append((np.linalg.norm(d1@quadratic+d2@linear),quadratic,linear))
    if not candidates: raise ValueError('fit did not produce an ellipse')
    _,quadratic,linear=min(candidates,key=lambda item:item[0])
    a,b,c=quadratic
    matrix=np.array([[a,b/2],[b/2,c]])
    try: center=-.5*np.linalg.solve(matrix,linear[:2])
    except np.linalg.LinAlgError as error: raise ValueError('invalid ellipse center') from error
    level=float(center@matrix@center-linear[2])
    eigen,axes=np.linalg.eigh(matrix)
    ratios=level/eigen
    if not np.isfinite(ratios).all() or np.any(ratios<=0): raise ValueError('imaginary ellipse')
    radii=np.sqrt(ratios); order=np.argsort(radii)[::-1]
    major=axes[:,order[0]]
    return EllipseFit(center*scale+origin,radii[order]*scale,float(np.mod(np.arctan2(major[1],major[0]),np.pi)))

def contour_metrics(points,nominal_diameter=None):
    """Keep all observed radii for form and wear; fit masks are diagnostic."""
    p=_checked(points,3); circle=fit_circle(p,robust=False)
    robust_circle=fit_circle(p,robust=True)
    radius=np.linalg.norm(p-circle.center,axis=1)
    metrics=dict(circle_center=circle.center,circle_radius=circle.radius,diameter=2*circle.radius,
                 minimum_radius=float(radius.min()),maximum_radius=float(radius.max()),
                 minimum_centered_diameter=float(2*radius.min()),maximum_centered_diameter=float(2*radius.max()),
                 roundness_lsc=float(np.ptp(radius)),roundness_method='least_squares_circle_peak_to_valley',
                 circle_fit_rmse=circle.rmse,fit_inliers=robust_circle.inliers,
                 fit_inlier_count=int(robust_circle.inliers.sum()),
                 robust_circle_center=robust_circle.center,robust_circle_radius=robust_circle.radius,
                 robust_fit_is_diagnostic=True,ellipse=None,ovality=None,maximum_radial_wear=None)
    if len(p)>=6:
        try:
            ellipse=fit_ellipse(p)
            metrics['ellipse']=dict(center=ellipse.center,semi_axes=ellipse.semi_axes,angle_rad=ellipse.angle_rad,
                                    major_diameter=float(2*ellipse.semi_axes[0]),minor_diameter=float(2*ellipse.semi_axes[1]))
            metrics['ovality']=float(2*np.ptp(ellipse.semi_axes))
        except ValueError:
            pass
    if nominal_diameter is not None:
        if not np.isfinite(nominal_diameter) or nominal_diameter<=0: raise ValueError('nominal diameter must be positive')
        metrics['maximum_radial_wear']=float(max(0,radius.max()-nominal_diameter/2))
        metrics['diameter_deviation']=float(metrics['diameter']-nominal_diameter)
    return metrics
