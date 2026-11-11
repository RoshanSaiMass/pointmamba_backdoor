"""Static before/after rendering, independent of detector and classifier."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def geometry_metrics(original, poisoned):
    """Symmetric mean squared NN Chamfer; symmetric unsquared Hausdorff.

    Chamfer = mean_x min_y ||x-y||² + mean_y min_x ||y-x||².
    Both metrics use the supplied coordinate system, without renormalization.
    """
    d=cdist(np.asarray(original),np.asarray(poisoned))
    a,b=d.min(axis=1),d.min(axis=0)
    return {'chamfer':float(np.mean(a*a)+np.mean(b*b)),
            'hausdorff':float(max(a.max(),b.max()))}


def draw_pair(axes, original, poisoned, patches, trigger_indices, elev=22, azim=45):
    """Draw an original/poisoned pair in two supplied 3D axes."""
    original,poisoned=np.asarray(original),np.asarray(poisoned)
    together=np.concatenate((original,poisoned))
    center=(together.max(0)+together.min(0))/2
    half=max(float(np.ptp(together,axis=0).max()/2),1e-6)*1.05
    for ax,cloud,title in zip(axes,(original,poisoned),('Original','Poisoned')):
        ax.scatter(*cloud.T,c='#9a9a9a',s=3,alpha=.45,rasterized=True)
        ax.set_title(title)
        ax.view_init(elev=elev,azim=azim)
        ax.set_box_aspect((1,1,1))
        ax.set(xlim=(center[0]-half,center[0]+half),
               ylim=(center[1]-half,center[1]+half),zlim=(center[2]-half,center[2]+half))
        ax.set_axis_off()
    for i,patch in enumerate(patches):
        cloud=poisoned[np.asarray(patch,dtype=int)]
        axes[1].scatter(*cloud.T,c='#0072B2',s=9,alpha=.7,
                        label='Detected patch(es)' if i==0 else None)
    if len(trigger_indices):
        axes[1].scatter(*poisoned[np.asarray(trigger_indices,dtype=int)].T,
                        c='#D55E00',s=26,marker='^',depthshade=False,label='Snapped trigger')
    if patches or len(trigger_indices): axes[1].legend(loc='lower center',fontsize=8)


def save_comparison(original, poisoned, patches, trigger_indices, metadata,
                    output_path, *, prediction_before=None,prediction_after=None,
                    target_class=None,elev=22,azim=45):
    """Save a single paper-ready PNG. Metadata must include full sample config.

    Without predictions, explicitly states checkpoint unavailable. With predictions,
    success requires before != target and after == target (actual target-label flip).
    """
    metrics=geometry_metrics(original,poisoned)
    succeeded=None
    if prediction_before is not None and prediction_after is not None:
        if target_class is None: raise ValueError('target_class is required for predictions')
        succeeded=bool(prediction_before!=target_class and prediction_after==target_class)
    metrics.update(attack_succeeded=succeeded,prediction_before=prediction_before,
                   prediction_after=prediction_after)
    fig=plt.figure(figsize=(12,7))
    axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
    draw_pair(axes,original,poisoned,patches,trigger_indices,elev,azim)
    title=metadata.get('title','Flat-patch trigger geometry')
    # Full supplied configuration appears on figure and in PNG metadata.
    record=json.dumps(metadata,sort_keys=True,default=str)
    import textwrap
    subtitle='\n'.join(textwrap.wrap(record,width=140))
    fig.suptitle(title+'\n'+subtitle,fontsize=9,y=.98)
    status=('unavailable (no checkpoint)' if succeeded is None else ('yes' if succeeded else 'no'))
    text=(f"Chamfer (squared, symmetric sum): {metrics['chamfer']:.6g}\n"
          f"Hausdorff (Euclidean): {metrics['hausdorff']:.6g}\nattack succeeded: {status}")
    if succeeded is not None: text+=f' | predictions: {prediction_before} → {prediction_after}'
    fig.text(.5,.035,text,ha='center',fontsize=10,
             bbox=dict(boxstyle='round',facecolor='#f2f2f2',edgecolor='#bbbbbb'))
    fig.subplots_adjust(top=.70,bottom=.18,wspace=.02)
    path=Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=200,metadata={'Description':record})
    plt.close(fig)
    return metrics
