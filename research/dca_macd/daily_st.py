import numpy as np
HOUR=3_600_000
DAY=24*HOUR
def daily_bars(ts, o, h, l, c):
    """完整 UTC 日（从 00:00 起，24 根 1h 都在）→ (open_ms, o, h, l, c)。同 nautilus gates.py::daily。"""
    key = ts // DAY; cuts = np.r_[0, np.flatnonzero(np.diff(key)) + 1, len(ts)]
    rows = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        start = int(key[a] * DAY)
        if ts[a] == start and ts[b - 1] + HOUR == start + DAY:
            rows.append((start, o[a], h[a:b].max(), l[a:b].min(), c[b - 1]))
    if not rows: return np.zeros(0, 'int64'), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    a = np.array(rows, float)
    return a[:, 0].astype('int64'), a[:, 1], a[:, 2], a[:, 3], a[:, 4]

def st_direction(h, l, c, length=10, mult=3):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))
    atr = np.full(len(c), np.nan); out = np.zeros(len(c), dtype=bool)
    if len(c) < length: return out
    atr[length - 1] = tr[:length].mean()
    for i in range(length, len(c)): atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
    upper = lower = 0.; up = False
    for i in range(length - 1, len(c)):
        bu = (h[i] + l[i]) / 2 + mult * atr[i]; bl = (h[i] + l[i]) / 2 - mult * atr[i]
        if i == length - 1: upper, lower = bu, bl
        else:
            upper = bu if bu < upper or c[i - 1] > upper else upper
            lower = bl if bl > lower or c[i - 1] < lower else lower
            up = (c[i] >= lower) if up else (c[i] > upper)
        out[i] = up
    return out

def map_daily(ts, dopen, up, valid, lag_ms=HOUR):
    """基础 bar i 只用收盘时刻 dopen+DAY <= ts[i]-lag 的日线（lag=1h：00:00 收盘的日线，01:00 的 bar 才用）。"""
    idx = np.searchsorted(dopen + DAY, ts - lag_ms, side='right') - 1
    st = np.zeros(len(ts), np.int8); va = np.zeros(len(ts), np.int8); m = idx >= 0
    st[m] = up[idx[m]].astype(np.int8); va[m] = valid[idx[m]].astype(np.int8)
    return st, va

def st_state(ts, o, h, l, c, length, mult):
    dopen, do, dh, dl, dc = daily_bars(ts, o, h, l, c)
    up = st_direction(dh, dl, dc, length, mult); valid = np.arange(len(dc)) >= length - 1
    return map_daily(ts, dopen, up, valid)
