"""Budget-aware curves in a circular local patch frame.

Clipped roses use [0, 2pi) and retain only nonnegative-radius intervals.
This gives k visible petals, including for odd k; it is deliberately not the
signed-radius standard parameterization (whose odd-k period is pi).
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

@dataclass
class CurveConfig:
    kind: str = 'spiral'
    turns: int = 3
    petals: int = 5
    freq_a: int = 3
    freq_b: int = 2
    delta: float = math.pi / 2
    density: float = 256.0  # samples per local coordinate unit of arc length
    min_points_per_feature: int = 4

@dataclass
class CurveResult:
    points: np.ndarray
    metadata: dict


def spiral(t, radius, turns):
    theta = t * (2 * np.pi * turns)
    r = radius * t
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def rose(t, radius, petals):
    theta = t * 2 * np.pi
    r = radius * np.cos(petals * theta)
    keep = r >= 0
    return np.column_stack((r[keep] * np.cos(theta[keep]), r[keep] * np.sin(theta[keep])))


def lissajous(t, radius, a, b, delta):
    theta = t * 2 * np.pi
    return radius / np.sqrt(2) * np.column_stack((np.sin(a * theta + delta), np.sin(b * theta)))


def generate_curve(radius, budget, config=None):
    c = config or CurveConfig()
    if radius < 0 or not np.isfinite(radius) or budget < 1:
        raise ValueError('radius must be finite/nonnegative and budget positive')
    if c.density <= 0 or c.min_points_per_feature < 1:
        raise ValueError('density and minimum feature budget must be positive')
    cap = max(1, budget // c.min_points_per_feature)
    turns, petals, a, b = c.turns, c.petals, c.freq_a, c.freq_b
    if min(turns, petals, a, b) < 1 or math.gcd(a, b) != 1:
        raise ValueError('positive complexity and coprime Lissajous frequencies required')
    if c.kind == 'spiral':
        turns = min(turns, cap)
        features = turns
    elif c.kind == 'rose':
        petals = min(petals, cap)
        features = petals
    elif c.kind == 'lissajous':
        # a+b-1 is a conservative feature-budget proxy, not an exact loop count.
        while a + b - 1 > cap:
            candidates = [(x,y) for x in range(1,a+1) for y in range(1,b+1)
                          if math.gcd(x,y)==1 and x+y < a+b]
            a,b = max(candidates, key=lambda p:(sum(p),min(p)))
        features = a+b-1
    else:
        raise ValueError(f'unknown curve: {c.kind}')
    def draw(t):
        if c.kind == 'spiral': return spiral(t, radius, turns)
        if c.kind == 'rose': return rose(t, radius, petals)
        return lissajous(t, radius, a, b, c.delta)
    pilot = draw(np.linspace(0,1,4096))
    length = np.linalg.norm(np.diff(pilot,axis=0),axis=1).sum()
    samples = max(128, budget*16, int(np.ceil(length*c.density)))
    if samples > 2_000_000:
        raise ValueError('curve density requests more than two million samples')
    points = draw(np.linspace(0,1,samples))
    norms = np.linalg.norm(points,axis=1)
    points *= np.minimum(1, radius/np.maximum(norms,1e-30))[:,None]
    effective = dict(turns=turns, petals=petals, freq_a=a, freq_b=b)
    requested = asdict(c)
    changed = ((c.kind=='spiral' and turns!=c.turns) or
               (c.kind=='rose' and petals!=c.petals) or
               (c.kind=='lissajous' and (a,b)!=(c.freq_a,c.freq_b)))
    return CurveResult(points, dict(requested=requested,effective=effective,
        reduced=changed,features=features,feature_budget_satisfied=budget>=features*c.min_points_per_feature,
        budget=budget,radius=radius,samples=len(points),pitch=radius/(2*np.pi*turns)))
