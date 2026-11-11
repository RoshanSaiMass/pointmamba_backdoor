"""Headless static figures with a renderer-independent data interface."""
import json
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def geometry_metrics(original,poisoned):
    d=cdist(np.asarray(original),np.asarray(poisoned))
    if d.size==0: raise ValueError('nonempty clouds required')
    a,b=d.min(1),d.min(0)
    return dict(chamfer=float(np.mean(a*a)+np.mean(b*b)),hausdorff=float(max(a.max(),b.max())))


def draw_pair(axes,original,poisoned,patches,trigger_indices,elev=22,azim=45):
    original,poisoned=np.asarray(original),np.asarray(poisoned)
    joint=np.concatenate((original,poisoned)); center=(joint.max(0)+joint.min(0))/2
    half=max(float(np.ptp(joint,axis=0).max()/2),1e-6)*1.05
    for ax,p,title in zip(axes,(original,poisoned),('Original','Poisoned')):
        ax.scatter(*p.T,c='#999999',s=3,alpha=.45,rasterized=True)
        ax.set_title(title,fontsize=9); ax.view_init(elev=elev,azim=azim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off()
        ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),zlim=(center[2]-half,center[2]+half))
    for j,patch in enumerate(patches):
        p=poisoned[np.asarray(patch,dtype=int)]
        axes[1].scatter(*p.T,c='#0072B2',s=9,alpha=.7,label='Selected patch' if j==0 else None)
    if trigger_indices:
        p=poisoned[np.asarray(trigger_indices,dtype=int)]
        axes[1].scatter(*p.T,c='#D55E00',s=23,marker='^',depthshade=False,label='Snapped trigger')


def save_comparison(original,poisoned,patches,trigger_indices,metadata,output_path,*,
                    prediction_before=None,prediction_after=None,target_class=None,elev=22,azim=45):
    metrics=geometry_metrics(original,poisoned)
    success=None
    if prediction_before is not None and prediction_after is not None:
        if target_class is None: raise ValueError('target class required with predictions')
        success=prediction_before!=target_class and prediction_after==target_class
    metrics.update(attack_succeeded=success,prediction_before=prediction_before,prediction_after=prediction_after)
    fig=plt.figure(figsize=(13,8)); axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
    draw_pair(axes,original,poisoned,patches,trigger_indices,elev,azim)
    axes[1].legend(loc='lower center',fontsize=8)
    label=json.dumps(metadata,sort_keys=True,default=str)
    fig.suptitle('\n'.join(textwrap.wrap(label,155)),fontsize=8,y=.99)
    status='unavailable (no checkpoint)' if success is None else ('yes' if success else 'no')
    text=f"Chamfer (sum of directed mean squared distances): {metrics['chamfer']:.6g}\nHausdorff (Euclidean): {metrics['hausdorff']:.6g}\nattack succeeded: {status}"
    if success is not None: text+=f' | predictions: {prediction_before} -> {prediction_after}, target={target_class}'
    fig.text(.5,.03,text,ha='center',fontsize=10,bbox=dict(facecolor='#eeeeee',edgecolor='#bbbbbb',boxstyle='round'))
    fig.subplots_adjust(top=.68,bottom=.16,wspace=.02)
    path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=180,metadata={'Description':label}); plt.close(fig)
    return metrics
