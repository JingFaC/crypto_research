"""ST(6,2.5) next-open replay, migrated without parameter search."""
import numpy as np
FEE=0.0005
FUND8H=0.0001
def run_fast(o, h, l, c, d, mode, tfms):
    n = len(c); d = d.astype(int); st = np.zeros(n, int); st[1:] = d[:-1]
    want = np.zeros(n, int)
    if mode == 'L': want[st == 1] = 1
    elif mode == 'S': want[st == -1] = -1
    else: want = st.copy()
    ch = np.flatnonzero(np.diff(want) != 0) + 1; bd = [0] + ch.tolist() + [n]
    eqc = np.ones(n); eql = np.ones(n); E = 1.0; tr = []; fb = FUND8H * tfms / (8 * 3600_000)
    for k in range(len(bd) - 1):
        a, b = bd[k], bd[k + 1]; p = want[a]
        if p == 0: eqc[a:b] = E; eql[a:b] = E; continue
        entry = o[a]; Ee = E * (1 - FEE); fx = fb * np.arange(1, b - a + 1) if p == 1 else 0.0
        eqc[a:b] = Ee * (1 + p * (c[a:b] / entry - 1) - fx)
        worst = l[a:b] if p == 1 else h[a:b]; eql[a:b] = Ee * (1 + p * (worst / entry - 1) - fx)
        fa = fb * (b - a) if p == 1 else 0.0
        if eql[a:b].min() <= 0:
            eqc[a:] = 0.0; eql[a:] = 0.0; tr.append(-1.0); return eqc, eql, np.array(tr)
        if b < n:
            g = p * (o[b] / entry - 1) - fa; E = Ee * (1 + g) * (1 - FEE); tr.append(g - 2 * FEE)
        else:
            g = p * (c[-1] / entry - 1) - fa; tr.append(g - 2 * FEE)
    return eqc, eql, np.array(tr)

def yearly(ts, eq):
    yy = (ts // 1000).astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970; out = {}; prev = 1.0
    for y in sorted(set(yy.tolist())):
        ix = np.flatnonzero(yy == y); out[y] = (eq[ix[-1]] / prev - 1) * 100; prev = eq[ix[-1]]
    return out

def dd_of(eqc, eql):
    pk = np.maximum.accumulate(eqc); return ((np.minimum(eql, eqc) - pk) / np.maximum(pk, 1e-12)).min() * 100
