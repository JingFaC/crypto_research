from __future__ import annotations
import numpy as np
from datetime import datetime,timezone
MINUTE=60_000
HOUR=60*MINUTE
DAY=24*HOUR
FEE=.0007
ANCHOR15=int(datetime(2026,9,13,tzinfo=timezone.utc).timestamp()*1000)
PERIODS=[("15m",15*MINUTE,0),("30m",30*MINUTE,0),("1h",HOUR,0),("4h",4*HOUR,0),("1d",DAY,0),("1w",7*DAY,4*DAY),("15d",15*DAY,ANCHOR15%(15*DAY))]
def st_line(h, l, c, length, mult):
    """SuperTrend：返回 (方向 +1/-1, 轨道线)。轨道线多头时为下轨(支撑)，空头时为上轨(压力)。

    只用到截至第 i 根收盘的数据（因果）。空头阶段 upper 单调不升，这是 ST 的固有性质，
    「轨道横住不动」等价于 upper 不再创新低 —— 空头动能衰竭的精确表达。
    """
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = rma(tr, length)
    n = len(c)
    d = np.full(n, np.nan)
    band = np.full(n, np.nan)
    lower = upper = 0.0
    trend = -1
    started = False
    for i in range(n):
        if not np.isfinite(atr[i]):
            continue
        mid = (h[i] + l[i]) / 2
        bl, bu = mid - mult * atr[i], mid + mult * atr[i]
        if not started:
            lower, upper, started = bl, bu, True
            trend = 1 if c[i] > bu else -1
        else:
            lower = max(bl, lower) if c[i - 1] > lower else bl
            upper = min(bu, upper) if c[i - 1] < upper else bu
            if c[i] > upper:
                trend = 1
            elif c[i] < lower:
                trend = -1
        d[i] = trend
        band[i] = lower if trend > 0 else upper
    return d, band
def bb(c, w=20, k=2.0):
    n = len(c); ma = np.full(n, np.nan); sd = np.full(n, np.nan)
    for i in range(w - 1, n):
        x = c[i - w + 1:i + 1]; ma[i] = x.mean(); sd[i] = x.std()
    return ma, ma + k * sd, ma - k * sd
def utc(t):
    return datetime.fromtimestamp(int(t) / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M')

def validate(bars, step):
    if len(bars) < 2 or bars.shape[1] != 6 or not np.isfinite(bars).all():
        raise ValueError('Expected finite [timestamp,open,high,low,close,volume] bars')
    t, o, h, l, c, v = bars.T
    if np.any(t % step) or np.any(np.diff(t) != step):
        raise ValueError('Unaligned, duplicate or discontinuous timestamps')
    if np.any(l > np.minimum(o, c)) or np.any(h < np.maximum(o, c)) or np.any(l <= 0) or np.any(v < 0):
        raise ValueError('Invalid OHLCV')

def resample(bars, base_ms, period_ms, offset=0):
    """Only complete, aligned groups; input continuity is validated by the caller."""
    keys = (bars[:, 0].astype(np.int64) - offset) // period_ms
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(bars)]
    rows = []
    for s, e in zip(starts, ends):
        if (bars[s, 0] - offset) % period_ms or (e - s) * base_ms != period_ms:
            continue
        z = bars[s:e]
        rows.append([z[0, 0], z[0, 1], z[:, 2].max(), z[:, 3].min(), z[-1, 4], z[:, 5].sum()])
    return np.asarray(rows, dtype=float).reshape(-1, 6)

def closed_candles(payload, step):
    fetched = int(datetime.fromisoformat(payload['fetched_at'].replace('Z', '+00:00')).timestamp() * 1000)
    rows, excluded = [], 0
    for x in payload['candles']:
        t = x['OpenTime']
        if x['CloseTime'] != t + step - 1:
            raise ValueError('Unexpected candle close time')
        if t + step > fetched:
            excluded += 1
            continue
        rows.append([x[k] for k in ('OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume')])
    return rows, excluded

def indicators(bars):
    direction, _ = st_line(bars[:, 2], bars[:, 3], bars[:, 4], 6, 2.5)
    mid, up, lo = bb(bars[:, 4], 20, 2.0)
    return dict(t=bars[:, 0].astype(np.int64), d=direction, mid=mid, up=up, lo=lo)

def simulate(base, base_ms, signal, period_ms, fee=FEE, settle_end=True, take_profit=True):
    """Place/refresh orders at base-bar open using only already closed signal bars."""
    index = np.searchsorted(signal['t'] + period_ms, base[:, 0], side='right') - 1
    trades, entries, equity = [], [], np.ones(len(base))
    side_seen, used, episode = None, True, None
    position, cash, ambiguous = None, 0.0, 0

    def close_position(i, price, reason, target=None, at_close=False):
        nonlocal position, cash
        gross = position['side'] * (price / position['entry_price'] - 1)
        cash += gross - fee
        trade = dict(position, exit_bar_open_ms=int(base[i, 0]), exit_price=float(price),
                     exit_reason=reason, exit_target=target, gross_return=float(gross),
                     net_return=float(gross - 2 * fee),
                     holding_hours=float((base[i, 0] + (base_ms if at_close else 0)
                                          - position['entry_bar_open_ms']) / HOUR))
        trades.append(trade)
        position = None

    for i, (t, o, h, l, c, _) in enumerate(base):
        k = int(index[i])
        if k >= 0 and np.isfinite(signal['d'][k]):
            side = int(signal['d'][k])
            if side_seen is None:
                side_seen = side  # Initial left-censored trend has no entry allowance.
            elif side != side_seen:
                side_seen, used, episode = side, False, int(signal['t'][k] + period_ms)
                if position is not None:
                    close_position(i, o, 'FLIP')
            ready = all(np.isfinite(signal[key][k]) for key in ('mid', 'up', 'lo'))
            if ready:
                target = float((signal['mid'][k] + signal['lo' if side > 0 else 'up'][k]) / 2) if take_profit else None
                if position is not None and take_profit:
                    at_open = o >= target if side > 0 else o <= target
                    touched = h >= target if side > 0 else l <= target
                    if at_open or touched:
                        close_position(i, o if at_open else target, 'TP', target)
                if position is None and not used:
                    level = float(signal['lo' if side > 0 else 'up'][k])
                    touched = l <= level if side > 0 else h >= level
                    if touched:
                        at_open = o <= level if side > 0 else o >= level
                        price = o if at_open else level
                        position = dict(entry_bar_open_ms=int(t), side=side, episode_start_ms=episode,
                                        entry_signal_close_ms=int(signal['t'][k] + period_ms),
                                        entry_level=level, entry_price=float(price), entry_target=target,
                                        entry_at_open=bool(at_open), entry_tp_ambiguous=False)
                        used = True
                        cash -= fee
                        tp_touched = take_profit and (h >= target if side > 0 else l <= target)
                        close_confirms = take_profit and (c >= target if side > 0 else c <= target)
                        if tp_touched and (at_open or close_confirms):
                            entries.append(dict(position))
                            close_position(i, target, 'TP', target)
                        else:
                            if tp_touched:
                                ambiguous += 1
                                position['entry_tp_ambiguous'] = True
                            entries.append(dict(position))
        if i == len(base) - 1 and position is not None and settle_end:
            close_position(i, c, 'END', at_close=True)
        equity[i] = 1 + cash
        if position is not None:
            equity[i] += position['side'] * (c / position['entry_price'] - 1)
    return dict(trades=trades, entries=entries, equity=equity, ambiguous_entry_tp=ambiguous)

def summarize(result, base):
    trades = result['trades']
    equity = np.r_[1.0, result['equity']]
    m = trade_metrics(trades)
    m.update(max_drawdown_notional=float(np.max(np.maximum.accumulate(equity) - equity)),
             median_holding_hours=float(np.median([x['holding_hours'] for x in trades])) if trades else None,
             exits={key: sum(x['exit_reason'] == key for x in trades) for key in ('TP', 'FLIP', 'END')},
             ambiguous_entry_tp=result['ambiguous_entry_tp'])
    if not np.isclose(equity[-1] - 1, m['total_net'], atol=1e-9):
        raise AssertionError('Marked equity and sum of closed trade PnL disagree')
    for x in trades:
        if x['entry_signal_close_ms'] > x['entry_bar_open_ms']:
            raise AssertionError('Future signal used')
    episodes = [(x['side'], x['episode_start_ms']) for x in result['entries']]
    if len(episodes) != len(set(episodes)):
        raise AssertionError('More than one entry in a trend episode')
    m['by_side'] = {name: trade_metrics([x for x in trades if x['side'] == side])
                    for name, side in [('long', 1), ('short', -1)]}
    years = sorted({utc(x['entry_bar_open_ms'])[:4] for x in trades})
    m['by_entry_year'] = {y: trade_metrics([x for x in trades if utc(x['entry_bar_open_ms']).startswith(y)])
                          for y in years}
    return m

def rma(x, n):
    out = np.full(len(x), np.nan)
    seed = None
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        if np.all(np.isfinite(w)):
            out[i] = w.mean()
            seed = i
            break
    if seed is not None:
        for i in range(seed + 1, len(x)):
            out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out

def trade_metrics(trades):
    r = np.asarray([x['net_return'] for x in trades])
    win = r[r > 0]
    loss = r[r < 0]
    return dict(n=len(r), win_rate=float(len(win) / len(r)) if len(r) else None,
                mean_net=float(r.mean()) if len(r) else None, total_net=float(r.sum()),
                profit_factor=float(win.sum() / -loss.sum()) if len(loss) else None,
                avg_win=float(win.mean()) if len(win) else None,
                avg_loss=float(loss.mean()) if len(loss) else None)
