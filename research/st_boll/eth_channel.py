import numpy as np
from . import indicators as S
FEE=0.0005
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

def run(t, o, h, l, c, side=1, length=6, mult=2.5, comp=False, fund8h=None, tpq=0.75, rfq=0.25):
    if fund8h is None: fund8h = 0.0001 if side == 1 else 0.0
    t4, o4, h4, l4, c4 = S.resample(t, o, h, l, c, '4h')
    d, LO, UP = st_both(h4, l4, c4, length, mult)
    n4 = len(t4); st_i = np.searchsorted(t, t4); en_i = np.searchsorted(t, t4 + 4 * 3600_000)
    q = lambda f: LO + f * (UP - LO)
    if side == 1: TP, RF = q(tpq), q(rfq)
    else: TP, RF = q(1 - tpq), q(1 - rfq)
    N = len(t); eqc = np.full(N, np.nan); eql = np.full(N, np.nan)
    cash = 1.0; pos = False; entry = 0.0; S_ = 0.0; pend = False; allow_rf = False; kind = ''; ent_i = 0; trades = []
    k0 = int(np.argmax(~np.isnan(LO))) + 1
    first = st_i[k0 + 1] if k0 + 1 < n4 else 0
    eqc[:first] = 1.0; eql[:first] = 1.0

    def close(i, px, why):
        nonlocal cash, pos
        r = side * (px / entry - 1)
        cash += S_ * r - S_ * FEE; trades.append((ent_i, i, entry, px, kind, why)); pos = False

    for k in range(k0, n4 - 1):
        j = k + 1; s, e = st_i[j], en_i[j]
        if s >= e or s >= N: continue
        if pend and pos: close(s, o[s], 'ST')
        pend = False
        entered_now = -1
        if d[k] == side and d[k - 1] != side and not pos:
            pos = True; entry = o[s]; S_ = cash if comp else 1.0; cash -= S_ * FEE; ent_i = s; kind = '首仓'; allow_rf = False; entered_now = s
        for i in range(s, e):
            if pos:
                cash -= S_ * fund8h / 96
                if i != entered_now:
                    tp = TP[k]
                    if (side == 1 and tp > entry and h[i] > tp) or (side == -1 and tp < entry and l[i] < tp):
                        close(i, tp, 'TP'); allow_rf = True
                        eqc[i] = cash; eql[i] = cash; continue
            elif allow_rf and d[k] == side:
                rf = RF[k]
                if (side == 1 and l[i] < rf) or (side == -1 and h[i] > rf):
                    pos = True; entry = rf; S_ = cash if comp else 1.0; cash -= S_ * FEE; ent_i = i; kind = '补仓'; entered_now = i
            if pos:
                worst = l[i] if side == 1 else h[i]
                eqc[i] = cash + S_ * side * (c[i] / entry - 1); eql[i] = cash + S_ * side * (worst / entry - 1)
            else:
                eqc[i] = cash; eql[i] = cash
        if d[j] == -side:
            allow_rf = False
            if pos: pend = True
    if pos: eqc[-1] = cash + S_ * side * (c[-1] / entry - 1)
    for arr in (eqc, eql):
        m = np.isnan(arr); idx = np.where(~m, np.arange(N), 0); np.maximum.accumulate(idx, out=idx); arr[:] = arr[idx]
    return dict(eqc=eqc, eql=eql, trades=trades, t4=t4, o4=o4, h4=h4, l4=l4, c4=c4, d=d, LO=LO, UP=UP, side=side)

def report(name, t, o, h, l, c, **kw):
    r = run(t, o, h, l, c, **kw); comp = kw.get('comp', False); side = r['side']
    tr = r['trades']; eqc, eql = r['eqc'], r['eql']
    rets = np.array([side * (ex / en - 1) - 2 * FEE for (_, _, en, ex, _, _) in tr])
    ev = np.minimum(eql, eqc); pk = np.maximum.accumulate(eqc)
    dd = ((ev - pk) / pk).min() * 100 if comp else (ev - pk).min() * 100
    tot = (eqc[-1] - 1) * 100
    w = rets[rets > 0]; ls = rets[rets <= 0]
    hold = np.array([(t[x[1]] - t[x[0]]) / 3.6e6 for x in tr])
    inpos = sum(t[x[1]] - t[x[0]] for x in tr) / (t[-1] - t[0]) * 100
    print(f'== {name} 笔数{len(tr)} 总收益{tot:+.1f}% 盯市最大回撤{dd:.1f}% 胜率{len(w)/max(len(rets),1)*100:.1f}% 盈亏比{(w.mean()/-ls.mean() if len(w) and len(ls) else np.nan):.2f} '
          f'均赢{w.mean()*100 if len(w) else 0:+.2f}% 均亏{ls.mean()*100 if len(ls) else 0:+.2f}% 持仓中位{np.median(hold):.0f}h 在仓时间{inpos:.0f}%')
    for kd in ('首仓', '补仓'):
        sel = [i for i, x in enumerate(tr) if x[4] == kd]
        if not sel: continue
        rr = rets[sel]; why = [tr[i][5] for i in sel]
        print(f'   {kd}: {len(sel)}笔 胜率{(rr>0).mean()*100:.0f}% 均{rr.mean()*100:+.2f}% 合计{rr.sum()*100:+.1f}% | TP出{why.count("TP")} ST出{why.count("ST")}')
    yy = (t // 1000).astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
    tyear = {}
    for x, rt in zip(tr, rets): tyear.setdefault(int(yy[x[1]]), []).append(rt)
    for y in sorted(set(yy.tolist())):
        idx = np.flatnonzero(yy == y); start = eqc[idx[0] - 1] if idx[0] > 0 else 1.0; end = eqc[idx[-1]]
        evy = np.minimum(eql[idx], eqc[idx]); pky = np.maximum.accumulate(np.concatenate([[start], eqc[idx]]))[1:]
        ddy = ((evy - pky) / pky).min() * 100 if comp else (evy - pky).min() * 100
        ret = (end / start - 1) * 100 if comp else (end - start) * 100
        bh = (c[idx[-1]] / o[idx[0]] - 1) * 100
        rr = np.array(tyear.get(y, [])); ww = rr[rr > 0]; ll = rr[rr <= 0]
        print(f'   {y}: 收益{ret:+7.1f}% 回撤{ddy:6.1f}% 笔{len(rr):3d} 胜率{(len(ww)/len(rr)*100 if len(rr) else np.nan):4.0f}% 盈亏比{(ww.mean()/-ll.mean() if len(ww) and len(ll) else np.nan):.2f} | 拿着{bh:+.0f}%')
    return r
