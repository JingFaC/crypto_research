"""日篮子累加器与统计（与 hold_sim.basket/basket_stats 同口径，但不需要把所有币的矩阵放进内存）。"""
import numpy as np
E0 = 10_000.0; DAY = 86_400_000
import numpy as np


def sharpe(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    return float(r.mean() / r.std(ddof=1) * np.sqrt(365)) if len(r) > 2 and r.std(ddof=1) > 0 else float('nan')


def daily(ts, eq):
    day = ts // DAY; last = np.flatnonzero(np.r_[day[1:] != day[:-1], True])
    return day[last], eq[last]

class Acc:
    def __init__(self, D0, D1):
        self.D0 = D0; self.D = D1 - D0 + 1
        self.tot = np.zeros(self.D); self.num = np.zeros(self.D - 1); self.den = np.zeros(self.D - 1); self.nact = np.zeros(self.D - 1)
    def add(self, days, e):
        a = int(days[0] - self.D0); b = int(days[-1] - self.D0); D = self.D
        M = np.empty(D); M[:a] = E0; M[a:b + 1] = e; M[b + 1:] = e[-1]
        self.tot += M
        act = np.zeros(D - 1, bool); act[a:min(b + 1, D - 1)] = True        # 前一日已上市且未结束
        self.num += np.where(act, M[1:] - M[:-1], 0.0); self.den += np.where(act, M[:-1], 0.0); self.nact += act
    def stats(self):
        tot = self.tot; r_res = tot[1:] / tot[:-1] - 1
        r_act = np.where(self.den > 0, self.num / np.where(self.den > 0, self.den, 1), np.nan)
        peak = np.maximum.accumulate(tot); mdd = float(((tot - peak) / peak).min())
        days = (self.D0 + 1 + np.arange(self.D - 1)).astype('int64')
        yr = (days.astype('datetime64[D]').astype('datetime64[Y]').astype(int) + 1970)
        return dict(ret_reserved=float(tot[-1] / tot[0] - 1), sharpe_reserved=sharpe(r_res), mdd_reserved=mdd, sharpe_active=sharpe(r_act),
                    by_year={int(y): sharpe(r_act[yr == y]) for y in sorted(set(yr))}, n_days=int(len(r_act)), r_act=r_act, tot=tot, yr=yr)
