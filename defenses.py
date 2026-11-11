"""Input defenses and per-class feature filtering.

Activation clustering/spectral signatures are training-set sanitizers. Their
ASR requires refitting a fresh model on retained data, not transforming test
features or claiming removal alone is a defended accuracy measurement.
"""
import torch

@torch.no_grad()
def statistical_outlier_removal(x,k=20,std=1.):
    if k<1 or std<0: raise ValueError('invalid SOR configuration')
    k=min(k,len(x[0])-1)
    d=torch.cdist(x,x)
    eye=torch.eye(x.shape[1],device=x.device,dtype=torch.bool)[None]
    means=d.masked_fill(eye,float('inf')).topk(k,largest=False).values.mean(-1)
    out=[]
    for p,score in zip(x,means):
        keep=torch.where(score<=score.mean()+std*score.std(unbiased=False))[0]
        if len(keep)==0: keep=score.argmin().view(1)
        # Restore tensor size using retained points only, deterministically.
        idx=keep[torch.arange(len(p),device=x.device)%len(keep)]
        out.append(p[idx])
    return torch.stack(out)


def simple_random_sampling(x,keep=.75):
    if not 0<keep<=1: raise ValueError('SRS keep must be in (0,1]')
    count=max(2,int(x.shape[1]*keep))
    return torch.stack([p[torch.randperm(len(p),device=p.device)[:count]] for p in x])


def gaussian_jitter(x,std=.01,clip=.05):
    if std<0 or clip<0: raise ValueError('invalid jitter configuration')
    return x+(torch.randn_like(x)*std).clamp(-clip,clip)


def transform(x,name,c):
    if name=='none': return x
    if name=='sor': return statistical_outlier_removal(x,c['sor_k'],c['sor_std'])
    if name=='srs': return simple_random_sampling(x,c['srs_keep'])
    if name=='jitter': return gaussian_jitter(x,c['jitter_std'],c['jitter_clip'])
    raise ValueError(name)

@torch.no_grad()
def feature_filter(features,labels,method,fraction=.1,max_cluster_fraction=.35):
    if not 0<=fraction<.5 or not 0<max_cluster_fraction<.5:
        raise ValueError('invalid feature filter fractions')
    keep=torch.ones(len(labels),dtype=torch.bool)
    for cls in labels.unique():
        idx=torch.where(labels==cls)[0]
        if len(idx)<4: continue
        x=features[idx].double(); x=x-x.mean(0)
        if torch.linalg.norm(x)<1e-12: continue
        if method=='spectral':
            _,_,vh=torch.linalg.svd(x,full_matrices=False)
            score=(x@vh[0]).square()
            count=min(len(idx)-1,int(len(idx)*fraction))
            if count: keep[idx[score.topk(count).indices]]=False
        elif method=='activation_clustering':
            # k=2 in a small PCA subspace, fixed initialization and bounded iterations.
            u,s,_=torch.linalg.svd(x,full_matrices=False)
            z=u[:,:min(10,len(s))]*s[:min(10,len(s))]
            first=z.square().sum(1).argmax(); second=(z-z[first]).square().sum(1).argmax()
            centers=z[torch.stack((first,second))].clone(); assignment=torch.zeros(len(z),dtype=torch.long)
            for _ in range(50):
                new=torch.cdist(z,centers).argmin(1)
                if torch.equal(new,assignment) and _: break
                assignment=new
                for j in range(2):
                    if (assignment==j).any(): centers[j]=z[assignment==j].mean(0)
            sizes=torch.bincount(assignment,minlength=2); small=int(sizes.argmin())
            if 0<sizes[small]<=len(idx)*max_cluster_fraction:
                keep[idx[assignment==small]]=False
        else: raise ValueError(method)
    return keep
