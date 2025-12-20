from scipy.stats import truncnorm
import numpy as np


def make_truncnorm(upper, lower, var, mean=None):
    '''
    If a mean is not passed the average of upper and lower will be used
    '''
    if not mean:
        mean = (upper+lower)/2

    return truncnorm((lower - mean)/var, (upper - mean)/var, loc=mean, scale=var)

class FastTruncNorm:
    def __init__(self, lo, hi, sd, mean=None):
        self.lo = float(lo)
        self.hi = float(hi)
        self.sd = float(sd)
        self.mean = float((self.lo + self.hi) / 2.0) if mean is None else float(mean)

    def rvs(self, rng: np.random.Generator, size=None):
        # rejection sampling; for your bounds this should accept quickly
        if size is None:
            while True:
                x = rng.normal(self.mean, self.sd)
                if self.lo <= x <= self.hi:
                    return x
        else:
            out = np.empty(size, dtype=float)
            k = 0
            while k < out.size:
                x = rng.normal(self.mean, self.sd, size=(out.size - k))
                x = x[(x >= self.lo) & (x <= self.hi)]
                m = x.size
                if m:
                    out[k:k+m] = x
                    k += m
            return out

def make_truncnorm(hi, lo, sd, mean=None):
    return FastTruncNorm(lo=lo, hi=hi, sd=sd, mean=mean)
