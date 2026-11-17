from dataclasses import dataclass, field
import numpy as np
HOUR=3_600_000
TF_MS={"1h":HOUR,"4h":4*HOUR,"1d":24*HOUR}
@dataclass
class Gate:
    name: str
    entry: np.ndarray
    add: np.ndarray
    exit_now: np.ndarray
    state: np.ndarray                 # 5m 上的 0/1
    raw: np.ndarray                   # HTF 上的 0/1
    htf_open_ms: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.entry)

    def slice(self, s: int) -> "Gate":
        """从 5m bar s 起截断的门（三个决策点、状态、原始状态同步截断）。

        注意：`raw` / `htf_open_ms` 是**高周期域**的数组，索引与 5m 无关，
        故不随 s 截断；要查高周期的原始状态请直接读 `raw` + `htf_open_ms`。
        """
        return Gate(self.name, self.entry[s:], self.add[s:], self.exit_now[s:],
                    self.state[s:], self.raw, self.htf_open_ms, self.meta)

    def stats(self):
        return dict(
            name=self.name, bars=len(self.state),
            open_ratio=float(self.state.mean()),
            flips=int(np.count_nonzero(np.diff(self.state))),
            **self.meta,
        )
def ema_go(x, n):
    """与 Go pkg/indicator/ema.go::EMA 同：EMA[n-1]=前 n 根 SMA，之后 α=2/(n+1) 递推；前 n-1 根无值（NaN）。"""
    x = np.asarray(x, float); out = np.full(len(x), np.nan)
    if len(x) < n: return out
    a = 2.0 / (n + 1.0); out[n - 1] = x[:n].mean()
    for i in range(n, len(x)): out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def apply_start(state, valid=None):
    """只从第一次"真实 OFF→ON 翻转"开始交易：翻转前一根 bar 必须 EMA 都有值且门为 OFF（预热期不算 OFF）；
    这之前的 ON 视为 OFF。返回 (新状态, 起点索引或 None)。"""
    s = state.astype(np.int8).copy(); d = np.diff(s)
    m = (d == 1) if valid is None else ((d == 1) & valid[:-1].astype(bool))
    cand = np.flatnonzero(m)
    if not len(cand): return np.zeros_like(s), None
    f = int(cand[0]) + 1; s[:f] = 0; return s, f

def make_gate(state, on_off, name='ema', ho=None, raw=None):
    on = state.astype(bool); n = len(on)
    ex = (~on) if on_off == 'exit' else np.zeros(n, bool)
    return Gate(name, on.copy(), on.copy(), ex, state.astype(np.int8), raw if raw is not None else np.zeros(1, np.int8),
                ho if ho is not None else np.zeros(1, np.int64), dict(on_off=on_off))

def fund_array(ts, ft, fr, side):
    """引擎按多头视角处理 fund（正扣负返）。保守口径：只扣不返。做多：max(r,0)；做空：max(-r,0)。事件落到开盘时间相同的 bar。"""
    cost = np.maximum(fr, 0.0) if side == 'long' else np.maximum(-fr, 0.0)
    out = np.zeros(len(ts)); idx = np.searchsorted(ts, ft, side='right') - 1
    ok = (idx >= 0) & (ft < ts[-1] + HOUR); np.add.at(out, idx[ok], cost[ok]); return out
def agg_tf(ts, c, tf_ms):
    """1h 收盘 → tf 收盘。起点为 UTC 整倍数；根数不齐的丢弃。返回 (htf_open_ms, close)。"""
    if tf_ms == HOUR: return ts.copy(), np.asarray(c, float).copy()
    k = tf_ms // HOUR; key = ts // tf_ms
    cuts = np.r_[0, np.flatnonzero(np.diff(key)) + 1, len(ts)]
    opens = []; closes = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a == k and ts[a] == key[a] * tf_ms: opens.append(int(key[a] * tf_ms)); closes.append(c[b - 1])
    return np.array(opens, 'int64'), np.array(closes, float)

def align_state(ts, ho, ema_f, ema_m, ema_s, tf_ms, lag_ms=HOUR):
    """多头排列(快>中>慢，严格) → 基础 bar 门状态与"EMA 是否都有值"。基础 bar i 只用 close_ms <= ts[i]-lag 的 HTF bar。"""
    ok = np.isfinite(ema_f) & np.isfinite(ema_m) & np.isfinite(ema_s)
    raw = (ok & (ema_f > ema_m) & (ema_m > ema_s)).astype(np.int8)
    idx = np.searchsorted(ho + tf_ms, ts - lag_ms, side='right') - 1
    st = np.zeros(len(ts), np.int8); va = np.zeros(len(ts), np.int8); m = idx >= 0
    st[m] = raw[idx[m]]; va[m] = ok[idx[m]].astype(np.int8)
    return st, va

class CoinGate:
    """一个币：缓存各 tf 的聚合和各周期的 EMA，按 (tf, 快, 中, 慢) 出门状态。"""
    def __init__(self, ts, c):
        self.ts = ts; self.c = c; self.agg = {}; self.ema = {}
    def state(self, tf, f, m, s):
        tfm = TF_MS[tf]
        if tf not in self.agg: self.agg[tf] = agg_tf(self.ts, self.c, tfm)
        ho, cc = self.agg[tf]
        for p in (f, m, s):
            if (tf, p) not in self.ema: self.ema[(tf, p)] = ema_go(cc, p)
        return align_state(self.ts, ho, self.ema[(tf, f)], self.ema[(tf, m)], self.ema[(tf, s)], tfm)
