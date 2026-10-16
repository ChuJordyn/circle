"""Isolated numeric acceleration of V5, without modifying its module globals.

Only row quantiles and separable convolution are replaced. Function code comes
from the original V5; private per-function namespaces prevent global patches
and keep the ordinary line benchmark independent of the annular backend.
"""
from dataclasses import replace
from types import FunctionType
import numpy as np
from centerline import ridge_support,normal_guided_centroid,normal_guided_steger,steger_batch
from centerline.preprocess import preprocess_laser_image

def row_nanpercentile(values,q,axis=None,**kwargs):
    a=np.asarray(values)
    if a.ndim!=2 or axis!=1 or np.ndim(q)!=0 or kwargs or np.isinf(a).any():
        return np.nanpercentile(a,q,axis=axis,**kwargs)
    if not 0<=q<=100: raise ValueError('quantile must be between 0 and 100')
    missing=np.isnan(a)
    if not missing.any(): return np.percentile(a,q,axis=1)
    count=np.sum(~missing,axis=1)
    ordered=np.sort(np.where(missing,np.inf,a),axis=1)
    rank=np.maximum(count-1,0)*(float(q)/100)
    lo=np.floor(rank).astype(int); hi=np.ceil(rank).astype(int)
    rows=np.arange(len(a)); fraction=rank-lo
    low=ordered[rows,lo].astype(float); high=ordered[rows,hi].astype(float)
    with np.errstate(invalid='ignore'):
        result=low+(high-low)*fraction
    result[count==0]=np.nan
    return result

def convolve_axis(image,kernel,axis):
    if axis not in [0,1]: raise ValueError('axis must be zero or one')
    a=np.asarray(image,dtype=np.float32); k=np.asarray(kernel,dtype=np.float32)
    padding=[(0,0),(0,0)]; padding[axis]=(len(k)//2,len(k)//2)
    windows=np.lib.stride_tricks.sliding_window_view(np.pad(a,padding,mode='edge'),len(k),axis=axis)
    return np.einsum('...k,k->...',windows,k[::-1],optimize=False).astype(np.float32)

def bilinear_sample(image,xs,ys):
    a=np.asarray(image); x=np.asarray(xs,dtype=np.float32); y=np.asarray(ys,dtype=np.float32)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return normal_guided_centroid.bilinear_sample(a,x,y)
    height,width=a.shape
    valid=(x>=0)&(x<=width-1)&(y>=0)&(y<=height-1)
    safe_x,safe_y=np.clip(x,0,width-1),np.clip(y,0,height-1)
    x0,y0=np.floor(safe_x).astype(np.int32),np.floor(safe_y).astype(np.int32)
    x1,y1=np.minimum(x0+1,width-1),np.minimum(y0+1,height-1)
    dx,dy=safe_x-x0,safe_y-y0
    top=a[y0,x0]*(1-dx)+a[y0,x1]*dx
    bottom=a[y1,x0]*(1-dx)+a[y1,x1]*dx
    return np.where(valid,top*(1-dy)+bottom*dy,0).astype(np.float32)

class _ArrayOperations:
    nanpercentile=staticmethod(row_nanpercentile)
    def __getattr__(self,name): return getattr(np,name)

def certified_ridge_peaks(smooth,max_step=8,penalty=.5):
    """Skip V5's DP only when its dominant path is provably optimal.

    Changing column i can save at most the dominant path costs of the three
    curvature terms and two motion terms that contain i. If EVERY competing
    peak's unary loss exceeds that bound, no changed subset can improve the
    path. Ambiguous profiles use the original five-candidate DP unchanged.
    """
    a=np.asarray(smooth); width,height=a.shape
    if width<3 or height<3 or penalty<0:
        return ridge_support.continuous_ridge_peaks(a,max_step,penalty)
    peaks=np.zeros_like(a,dtype=bool)
    peaks[:,1:-1]=(a[:,1:-1]>=a[:,:-2])&(a[:,1:-1]>=a[:,2:])&((a[:,1:-1]>a[:,:-2])|(a[:,1:-1]>a[:,2:]))
    ranked=np.where(peaks,a,-np.inf); count=min(5,height)
    candidates=np.argsort(ranked,axis=1)[:,-count:]
    scores=np.take_along_axis(ranked,candidates,axis=1)
    fallback=np.argmax(a,axis=1)
    candidates=np.where(np.isfinite(scores),candidates,fallback[:,None])
    scores=np.take_along_axis(a,candidates,axis=1)
    scale=np.maximum(np.max(np.where(np.isfinite(a),a,0),axis=1),1e-6)
    likelihood=np.maximum(np.where(np.isfinite(scores),scores,0),0)/scale[:,None]
    unary=-np.log(np.maximum(likelihood,.01)).astype(np.float32)
    best=np.argmin(unary,axis=1); row=np.arange(width)
    dominant=candidates[row,best]
    excess=unary-unary[row,best,None]
    competing=np.where(candidates!=dominant[:,None],excess,np.inf)
    loss=competing.min(axis=1)
    motion=np.zeros(width); curvature=np.zeros(width)
    delta=np.abs(np.diff(dominant))
    motion[1:]=.02*delta+.1*np.maximum(delta-max_step,0)**2
    curvature[2:]=penalty*np.abs(dominant[2:]-2*dominant[1:-1]+dominant[:-2])
    budget=motion.copy()+curvature
    budget[:-1]+=motion[1:]+curvature[1:]
    budget[:-2]+=curvature[2:]
    if np.all(loss>budget+1e-6): return dominant
    return precomputed_ridge_peaks(a,max_step,penalty,_data=(candidates,unary))

def precomputed_ridge_peaks(smooth,max_step=8,penalty=.5,_data=None):
    """Same DP and addition order as V5; precompute static transition terms."""
    a=np.asarray(smooth); width,height=a.shape
    if width<3 or height<3: return ridge_support.continuous_ridge_peaks(a,max_step,penalty)
    if _data is None:
        peaks=np.zeros_like(a,dtype=bool)
        peaks[:,1:-1]=(a[:,1:-1]>=a[:,:-2])&(a[:,1:-1]>=a[:,2:])&((a[:,1:-1]>a[:,:-2])|(a[:,1:-1]>a[:,2:]))
        ranked=np.where(peaks,a,-np.inf); count=min(5,height)
        candidates=np.argsort(ranked,axis=1)[:,-count:]
        scores=np.take_along_axis(ranked,candidates,axis=1)
        candidates=np.where(np.isfinite(scores),candidates,np.argmax(a,axis=1)[:,None])
        scores=np.take_along_axis(a,candidates,axis=1)
        scale=np.maximum(np.max(np.where(np.isfinite(a),a,0),axis=1),1e-6)
        likelihood=np.maximum(np.where(np.isfinite(scores),scores,0),0)/scale[:,None]
        unary=-np.log(np.maximum(likelihood,.01)).astype(np.float32)
    else: candidates,unary=_data
    count=candidates.shape[1]
    motion=np.abs(candidates[1:,None,:]-candidates[:-1,:,None])
    small_motion=.02*motion; jumps=.1*np.maximum(motion-max_step,0)**2
    curvature=penalty*np.abs(candidates[2:,None,None,:]-2*candidates[1:-1,None,:,None]+candidates[:-2,:,None,None])
    cost=unary[0][:,None]+unary[1][None,:]+small_motion[0]+jumps[0]
    next_cost=np.empty_like(cost); alternatives=np.empty((count,count,count))
    back=np.empty((width,count,count),dtype=np.int8)
    for column in range(2,width):
        np.add(cost[:,:,None],curvature[column-2],out=alternatives)
        alternatives.argmin(axis=0,out=back[column])
        alternatives.min(axis=0,out=next_cost)
        np.add(next_cost,unary[column][None,:],out=next_cost)
        np.add(next_cost,small_motion[column-1],out=next_cost)
        np.add(next_cost,jumps[column-1],out=next_cost)
        next_cost-=next_cost.min()
        cost,next_cost=next_cost,cost
    path=np.empty(width,dtype=np.int32)
    previous,current=np.unravel_index(np.argmin(cost),cost.shape)
    path[-1],path[-2]=candidates[-1,current],candidates[-2,previous]
    for column in range(width-1,1,-1):
        earlier=back[column,previous,current]
        path[column-2]=candidates[column-2,earlier]
        previous,current=earlier,previous
    return path

def _bind(function,**dependencies):
    namespace=dict(function.__globals__); namespace.update(dependencies)
    bound=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    bound.__kwdefaults__=function.__kwdefaults__
    return bound

_localize=_bind(ridge_support.localize_main_stripe,np=_ArrayOperations(),continuous_ridge_peaks=certified_ridge_peaks)
_guided=_bind(normal_guided_centroid.extract_normal_guided_gray_centroid_vectorized,
              np=_ArrayOperations(),localize_main_stripe=_localize,bilinear_sample=bilinear_sample)
_separable=_bind(normal_guided_steger.convolve_separable,convolve_axis=convolve_axis)
_fields=_bind(normal_guided_steger.build_derivative_fields,convolve_separable=_separable)
_batch=_bind(steger_batch.extract_batch,build_derivative_fields=_fields,bilinear_sample=bilinear_sample)

def run_ring_v5(image,config,observed_mask,accelerated=True):
    observed=np.asarray(observed_mask,dtype=bool)
    if observed.shape!=np.asarray(image).shape: raise ValueError('observed mask must match polar image')
    prep=preprocess_laser_image(image,roi=config.roi,background_percentile=config.background_percentile,
                                valid_threshold_ratio=config.valid_threshold_ratio,normalize=config.normalize)
    roi=prep.roi_mask&observed; valid=prep.valid_mask&observed
    normal=replace(config.normal,y_min=config.y_min,y_max=config.y_max)
    guided_function=_guided if accelerated else normal_guided_centroid.extract_normal_guided_gray_centroid_vectorized
    guided=guided_function(prep.image,config=normal,roi_mask=None if roi.all() else roi,valid_mask=valid)
    selected=guided
    if config.refine_with_steger:
        rows=np.arange(prep.image.shape[0])[:,None]
        search=roi&(rows>=config.y_min)&(rows<config.y_max)
        batch=_batch if accelerated else steger_batch.extract_batch
        selected=normal_guided_steger._enforce_output_region(batch(prep.image,config.steger,guided,search,valid),guided,search,valid)
    return dict(selected=selected,guided=guided,preprocess=prep)
