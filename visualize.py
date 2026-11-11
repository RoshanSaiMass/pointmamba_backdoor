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
                    prediction_before=None,prediction_after=None,target_class=None,
                    elev=22,azim=45,patch_records=None):
    """Original, unmarked poisoned, annotated poisoned, and local-plane diagnostics.

    Only scatter coordinates are classifier inputs; colors and reference curves
    are explanatory overlays. All three 3D views have identical camera/bounds.
    """
    original,poisoned=np.asarray(original),np.asarray(poisoned)
    metrics=geometry_metrics(original,poisoned)
    success=None
    if prediction_before is not None and prediction_after is not None:
        if target_class is None: raise ValueError('target class required with predictions')
        success=bool(prediction_before!=target_class and prediction_after==target_class)
    metrics.update(attack_succeeded=success,prediction_before=prediction_before,prediction_after=prediction_after)
    records=patch_records or []
    rows=max(2,int(np.ceil((3+len(records))/2)))
    fig=plt.figure(figsize=(14,5*rows))
    grid=fig.add_gridspec(rows,2)
    axes=[fig.add_subplot(grid[row,col],projection='3d')
          for row,col in ((0,0),(0,1),(1,0))]
    draw_pair(axes[:2],original,poisoned,[],[],elev,azim)
    draw_pair([axes[2]],poisoned,poisoned,[],[],elev,azim)
    axes[0].set_title('Original — all points unmarked')
    axes[1].set_title('Poisoned — all points unmarked')
    axes[2].set_title('Poisoned — explanatory annotations')
    # Reapply identical limits to every view, including the annotated one.
    for ax in axes[1:]:
        ax.set(xlim=axes[0].get_xlim(),ylim=axes[0].get_ylim(),zlim=axes[0].get_zlim())
    for j,patch in enumerate(patches):
        p=poisoned[np.asarray(patch,dtype=int)]
        axes[2].scatter(*p.T,c='#0072B2',s=10,alpha=.65,label='Selected patch' if j==0 else None)
    if len(trigger_indices):
        p=poisoned[np.asarray(trigger_indices,dtype=int)]
        axes[2].scatter(*p.T,c='#D55E00',s=22,marker='^',depthshade=False,label='Moved existing points')
    if patches or len(trigger_indices): axes[2].legend(loc='lower center',fontsize=8)
    for j,record in enumerate(records):
        cell=3+j
        ax=fig.add_subplot(grid[cell//2,cell%2])
        center,frame=np.asarray(record['center']),np.asarray(record['axes'])
        local=(poisoned[np.asarray(record['patch'],dtype=int)]-center)@frame
        curve,snapped=np.asarray(record['curve_uv']),np.asarray(record['snapped_uv'])
        ax.scatter(*local.T,c='#999999',s=12,label='Final patch points')
        ax.plot(*curve.T,c='#0072B2',lw=.8,alpha=.6,label='Reference curve (overlay only)')
        ax.scatter(*snapped.T,c='#D55E00',s=24,marker='^',zorder=3,label='Actual moved points')
        turns=record['curve']['effective']
        ax.set_title(f"Local top-down patch {j}: {len(snapped)} targets\n{turns}",fontsize=9)
        ax.set_aspect('equal',adjustable='box'); ax.set_xlabel('Local u'); ax.set_ylabel('Local v')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    if not records:
        ax=fig.add_subplot(grid[1,1]); ax.axis('off')
        ax.text(.5,.5,'Local top-down unavailable: patch frame records were not supplied',ha='center')
    label=json.dumps(metadata,sort_keys=True,default=str)
    # Full metadata remains in PNG and manifest; concise text keeps the plots readable.
    keys=('class_name','sample','curve_type','M','K_flat','planarity_thresh','placement','mode','reduced','poison_rate','target_class')
    caption=' | '.join(f'{key}={metadata[key]}' for key in keys if key in metadata)
    fig.suptitle('\n'.join(textwrap.wrap(caption or label,150)),fontsize=10,y=.99)
    status='unavailable (no checkpoint)' if success is None else ('yes' if success else 'no')
    text=f"Chamfer (sum of directed mean squared distances): {metrics['chamfer']:.6g} | Hausdorff (Euclidean): {metrics['hausdorff']:.6g}\nattack succeeded: {status}"
    if success is not None: text+=f' | predictions: {prediction_before} -> {prediction_after}, target={target_class}'
    fig.text(.5,.025,text,ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.93))
    path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=180,metadata={'Description':label}); plt.close(fig)
    return metrics
