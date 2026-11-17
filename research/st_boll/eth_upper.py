import numpy as np
from . import indicators as S
FEE=0.0005
def run(t, o, h, l, c, length=6, mult=2.5, once=False, comp=True, fund8h=0.0, bn=20, bk=2.0, use_lift=True, use_boll=True):
    t4, o4, h4, l4, c4 = S.resample(t, o, h, l, c, '4h')
    d, b, a = S.st(h4, l4, c4, length, mult)
    _, up, _ = S.boll(c4, bn, bk)
    n4 = len(t4)
    # 4h 开始位置在 5m 数组里的下标
    st_i = np.searchsorted(t, t4); en_i = np.searchsorted(t, t4 + 4 * 3600_000)
    # 抬升标记与段号
    lift = np.zeros(n4, bool); seg = np.zeros(n4, int); sid = 0; short_up = np.nan
    for k in range(n4):
        if np.isnan(b[k]): continue
        if k > 0 and d[k] != d[k - 1]: sid += 1
        seg[k] = sid
        if d[k] < 0: short_up = b[k]
        elif not np.isnan(short_up): lift[k] = b[k] > short_up
    N = len(t); eqc = np.full(N, np.nan); eql = np.full(N, np.nan)
    cash = 1.0; pos = False; entry = 0.0; S_ = 0.0; seg_used = -1; pend = False; trades = []; ent_i = 0
    k0 = int(np.argmax(~np.isnan(b))) + 1
    first = st_i[k0 + 1] if k0 + 1 < n4 else 0
    eqc[:first] = 1.0; eql[:first] = 1.0
    for k in range(k0, n4 - 1):
        j = k + 1; s, e = st_i[j], en_i[j]
        if s >= e or s >= N: continue
        # 上一根4h收盘确认的ST反转 -> 本根第一根5m开盘平仓
        if pend and pos:
            ex = o[s]; r = (ex / entry - 1)
            cash += S_ * r - S_ * FEE; trades.append((ent_i, s, entry, ex, 'ST', None)); pos = False
        pend = False
        L = b[k] if (d[k] > 0 and (lift[k] or not use_lift) and (not once or seg_used != seg[k])) else np.nan
        U = up[k] if use_boll else np.nan
        for i in range(s, e):
            filled = False
            if pos:
                cash -= S_ * fund8h / 96
                if (not np.isnan(U)) and U > entry and h[i] > U:
                    r = U / entry - 1; cash += S_ * r - S_ * FEE; trades.append((ent_i, i, entry, U, 'BOLL', None)); pos = False
                    eqc[i] = cash; eql[i] = cash; continue
            elif (not np.isnan(L)) and l[i] < L:
                pos = True; entry = L; S_ = cash if comp else 1.0; cash -= S_ * FEE; ent_i = i; seg_used = seg[k]; filled = True
            if pos:
                eqc[i] = cash + S_ * (c[i] / entry - 1); eql[i] = cash + S_ * (min(l[i], entry) / entry - 1) if filled else cash + S_ * (l[i] / entry - 1)
            else:
                eqc[i] = cash; eql[i] = cash
        if pos and d[j] < 0: pend = True
    # 尾部
    if pos:
        eqc[-1] = cash + S_ * (c[-1] / entry - 1)
    # 前向填充
    for arr in (eqc, eql):
        m = np.isnan(arr)
        idx = np.where(~m, np.arange(N), 0); np.maximum.accumulate(idx, out=idx); arr[:] = arr[idx]
    return dict(eqc=eqc, eql=eql, trades=trades, t4=t4, d=d, b=b, lift=lift)

def trade_stats(trades, comp_eq_at=None):
    rets = []
    for (i0, i1, en, ex, why, _) in trades:
        rets.append(ex / en - 1 - 2 * FEE)
    return np.array(rets)

def yearly(t, eqc, eql, c, o, trades, comp):
    yy = (t // 1000).astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
    out = {}
    tr_year = {}
    for (i0, i1, en, ex, why, _) in trades:
        tr_year.setdefault(int(yy[i1]), []).append((ex / en - 1 - 2 * FEE, why, (t[i1] - t[i0]) / 3.6e6))
    for y in sorted(set(yy.tolist())):
        idx = np.flatnonzero(yy == y); start = eqc[idx[0] - 1] if idx[0] > 0 else 1.0; end = eqc[idx[-1]]
        ev = np.minimum(eql[idx], eqc[idx]); pk = np.maximum.accumulate(np.concatenate([[start], eqc[idx]]))[1:]
        dd = ((ev - pk) / pk).min() * 100 if comp else (ev - pk).min() * 100
        ret = (end / start - 1) * 100 if comp else (end - start) * 100
        bh = (c[idx[-1]] / o[idx[0]] - 1) * 100
        tr = tr_year.get(y, [])
        w = [r for r, _, _ in tr if r > 0]; ls = [r for r, _, _ in tr if r <= 0]
        out[y] = dict(ret=ret, dd=dd, bh=bh, n=len(tr), win=(len(w) / len(tr) * 100 if tr else np.nan),
                      pay=(np.mean(w) / -np.mean(ls) if w and ls else np.nan))
    return out

def report(name, t, o, h, l, c, **kw):
    r = run(t, o, h, l, c, **kw); comp = kw.get('comp', True)
    tr = r['trades']; rets = trade_stats(tr); eqc, eql = r['eqc'], r['eql']
    ev = np.minimum(eql, eqc); pk = np.maximum.accumulate(eqc); dd = ((ev - pk) / pk).min() * 100 if comp else (ev - pk).min() * 100
    tot = (eqc[-1] / 1 - 1) * 100 if comp else (eqc[-1] - 1) * 100
    w = rets[rets > 0]; ls = rets[rets <= 0]
    why = [x[4] for x in tr]; hold = np.array([(t[x[1]] - t[x[0]]) / 3.6e6 for x in tr])
    print(f'== {name} 笔数{len(tr)} 总收益{tot:+.1f}% 最大回撤(盯市){dd:.1f}% 胜率{len(w)/max(len(rets),1)*100:.1f}% 盈亏比{(w.mean()/-ls.mean() if len(w) and len(ls) else np.nan):.2f} '
          f'均赢{w.mean()*100 if len(w) else 0:+.2f}% 均亏{ls.mean()*100 if len(ls) else 0:+.2f}% 持仓中位{np.median(hold):.0f}h BOLL出{why.count("BOLL")} ST出{why.count("ST")}')
    ys = yearly(t, eqc, eql, c, o, tr, comp)
    for y, v in ys.items():
        print(f'   {y}: 收益{v["ret"]:+7.1f}% 回撤{v["dd"]:6.1f}% 笔{v["n"]:3d} 胜率{v["win"]:4.0f}% 盈亏比{v["pay"]:.2f} | 拿着{v["bh"]:+.0f}%')
    return r
