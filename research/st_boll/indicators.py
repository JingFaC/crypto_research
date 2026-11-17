"""Original causal resampling, Bollinger and ST formulas; Windows portable backend."""
import numpy as np
TFMS={"5m":300000,"15m":900000,"30m":1800000,"1h":3600000,"4h":14400000,"1d":86400000,"1w":604800000}
TFOFF={"1w":4*86400000}
def boll(c, n=20, k=2.0):
    cs = np.cumsum(np.insert(c, 0, 0)); cs2 = np.cumsum(np.insert(c * c, 0, 0))
    m = np.full(len(c), np.nan); sd = np.full(len(c), np.nan)
    if len(c) >= n:
        mm = (cs[n:] - cs[:-n]) / n; vv = np.maximum((cs2[n:] - cs2[:-n]) / n - mm * mm, 0)
        m[n - 1:] = mm; sd[n - 1:] = np.sqrt(vv)
    return m, m + k * sd, m - k * sd

def resample(t, o, h, l, c, tf):
    ms = TFMS[tf]; off = TFOFF.get(tf, 0); base = int(np.median(np.diff(t[:1000])))
    k = (t - off) // ms
    cut = np.flatnonzero(np.diff(k)) + 1
    s = np.concatenate(([0], cut)); e = np.concatenate((cut, [len(k)]))
    full = (e - s) == (ms // base)
    return ((k[s] * ms + off)[full], o[s][full], np.maximum.reduceat(h, s)[full],
            np.minimum.reduceat(l, s)[full], c[e - 1][full])

def st_both(h, l, c, length=6, mult=2.5):
    """与 stlib.st 逐行等价, 但同时返回上轨/下轨。"""
    n = len(c); tr = np.empty(n); tr[0] = h[0] - l[0]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = np.full(n, np.nan); atr[length - 1] = tr[:length].mean()
    for i in range(length, n): atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
    LO = np.full(n, np.nan); UP = np.full(n, np.nan); d = np.zeros(n); trend = -1; started = False; lo = up = 0.0
    for i in range(n):
        if np.isnan(atr[i]): continue
        m = (h[i] + l[i]) / 2; bl = m - mult * atr[i]; bu = m + mult * atr[i]
        if not started: lo, up = bl, bu; started = True
        else:
            lo = max(bl, lo) if c[i - 1] > lo else bl
            up = min(bu, up) if c[i - 1] < up else bu
            if c[i] > up: trend = 1
            elif c[i] < lo: trend = -1
        d[i] = trend; LO[i] = lo; UP[i] = up
    return d, LO, UP

def st(h,l,c,length=6,mult=2.5):
    d,lo,up=st_both(h,l,c,length,mult)
    tr=np.r_[h[0]-l[0],np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))]
    atr=np.full(len(c),np.nan)
    if len(c)>=length:
        atr[length-1]=tr[:length].mean()
        for i in range(length,len(c)):atr[i]=(atr[i-1]*(length-1)+tr[i])/length
    return d,np.where(d>0,lo,up),atr
