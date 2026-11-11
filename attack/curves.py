"""Deterministic, radius-bounded local curves; no point-cloud dependencies."""
from dataclasses import dataclass, asdict
import math
import numpy as np

@dataclass(frozen=True)
class CurveConfig:
    curve_type: str = 'spiral'
    num_turns: int = 3
    k_petals: int = 5
    freq_a: int = 3
    freq_b: int = 2
    phase_delta: float = math.pi / 2
    curve_density: float = 256.0  # samples per unit arclength
    min_points_per_feature: int = 4
    rose_clip_negative: bool = True
    insufficient_budget: str = 'raise'  # raise | simplest

@dataclass
class CurveResult:
    points: np.ndarray
    metadata: dict


def _complexity(cfg, budget):
    if budget < 1 or cfg.min_points_per_feature < 1:
        raise ValueError('Budget and min_points_per_feature must be positive')
    allowed = budget // cfg.min_points_per_feature
    unmet = allowed < 1
    if unmet and cfg.insufficient_budget == 'raise':
        raise ValueError('Budget cannot represent even one feature')
    if cfg.insufficient_budget not in ('raise', 'simplest'):
        raise ValueError('Unknown insufficient_budget policy')
    allowed = max(1, allowed)
    if cfg.curve_type == 'spiral':
        if cfg.num_turns < 1: raise ValueError('num_turns must be positive')
        requested = {'num_turns': cfg.num_turns}
        effective = {'num_turns': min(cfg.num_turns, allowed)}
    elif cfg.curve_type == 'rose':
        if cfg.k_petals < 1: raise ValueError('k_petals must be positive')
        requested = {'k_petals': cfg.k_petals}
        # Clipped rose has k lobes; signed even-k rose has 2k petals.
        k = cfg.k_petals
        def features(k): return k if cfg.rose_clip_negative or k % 2 else 2*k
        while features(k) > allowed and k > 1: k -= 1
        effective = {'k_petals': k}
    elif cfg.curve_type == 'lissajous':
        a, b = cfg.freq_a, cfg.freq_b
        if min(a,b) < 1 or math.gcd(a,b) != 1:
            raise ValueError('Lissajous frequencies must be positive and coprime')
        requested = {'freq_a': a, 'freq_b': b}
        # Conservative feature proxy: a+b-1 crossing/loop regions.
        pairs = [(x,y) for x in range(1,a+1) for y in range(1,b+1)
                 if math.gcd(x,y)==1 and x+y-1 <= allowed]
        a,b = max(pairs, key=lambda p:(p[0]+p[1],p[0]))
        effective = {'freq_a': a, 'freq_b': b}
    else: raise ValueError(f'Unknown curve: {cfg.curve_type}')
    return requested, effective, unmet


def generate_curve(radius: float, budget: int, config: CurveConfig) -> CurveResult:
    """Return dense local (u,v) samples and per-patch effective parameters.

    Clipped roses use [0,2pi], with negative-r intervals OMITTED, preventing
    artificial radial connectors. Signed roses use pi for odd k, 2pi for even k.
    Lissajous amplitude radius/sqrt(2) bounds even its box corners by radius.
    Sampling resolution is estimated from arclength, independently of snap budget.
    """
    if not np.isfinite(radius) or radius < 0: raise ValueError('Invalid radius')
    if config.curve_density <= 0: raise ValueError('curve_density must be positive')
    requested, effective, unmet = _complexity(config, budget)
    def sample(n):
        if config.curve_type == 'spiral':
            end = 2*math.pi*effective['num_turns']
            theta = np.linspace(0,end,n)
            r = radius*theta/end
            return np.column_stack((r*np.cos(theta),r*np.sin(theta)))
        if config.curve_type == 'rose':
            k = effective['k_petals']
            end = 2*math.pi if config.rose_clip_negative or k%2==0 else math.pi
            theta = np.linspace(0,end,n)
            r = radius*np.cos(k*theta)
            if config.rose_clip_negative:
                keep = r >= 0
                theta,r = theta[keep],r[keep]
            return np.column_stack((r*np.cos(theta),r*np.sin(theta)))
        theta = np.linspace(0,2*math.pi,n)
        amp = radius/math.sqrt(2)
        return amp*np.column_stack((np.sin(effective['freq_a']*theta+config.phase_delta),
                                    np.sin(effective['freq_b']*theta)))
    pilot = sample(4096)
    length = np.linalg.norm(np.diff(pilot,axis=0),axis=1).sum()
    # Clipped rose retains approximately half of the angular samples.
    multiplier = 2 if config.curve_type=='rose' and config.rose_clip_negative else 1
    count = multiplier*max(128, 8*budget, int(math.ceil(length*config.curve_density))+1)
    points = sample(count)
    norms = np.linalg.norm(points,axis=1)
    points *= np.minimum(1.0, radius/np.maximum(norms,np.finfo(float).tiny))[:,None]
    return CurveResult(points, {'config':asdict(config), 'requested':requested,
        'effective':effective,'reduced':requested!=effective,
        'minimum_unmet':unmet,'radius':radius,'budget':budget,
        'sample_count':len(points),'estimated_length':float(length)})


def spiral(radius, budget, **kwargs):
    return generate_curve(radius,budget,CurveConfig(curve_type='spiral',**kwargs))


def rose(radius, budget, **kwargs):
    return generate_curve(radius,budget,CurveConfig(curve_type='rose',**kwargs))


def lissajous(radius, budget, **kwargs):
    return generate_curve(radius,budget,CurveConfig(curve_type='lissajous',**kwargs))
