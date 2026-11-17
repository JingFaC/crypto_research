"""Local numpy runner for gated DCA: ZigZag and BTC-weekly-ST gates. Uses engine.simulate (no nautilus_trader).

Per coin: a fixed-size 10000 account, 1x, one gate-open state mask over the 1h bars, no funding
(project decision: only the per-side trade fee is charged).
"""
import csv
from pathlib import Path
import numpy as np

if __package__:
    from . import engine as E
    from . import native_signals as S
    from .planning import planning
    from .signals import make_gate
else:
    import engine as E
    import native_signals as S
    from planning import planning
    from signals import make_gate

HOUR = 3_600_000
ACCOUNT = 10000.0
SKIPPED = []  # coins whose 1h data has gaps or is missing; reported, never silently dropped


def mask_from_windows(n, windows):
    state = np.zeros(n, bool)
    for w in windows:
        state[w['open_i']:w['close_i']] = True
    return state


def params(step, tp, amt, n, compound):
    return E.Params(step=step, gx=1., amt=amt, tp=tp, n=n, lev=1., base=E.base_for(ACCOUNT, .95, amt, n, 1.),
                    intrabar='opt', slip=0., compound=compound, cap=.95, fb=1., side='long', sweep_tp=False)


def run_coin(bars, state, pr):
    o, h, l, c = (bars[:, k].astype(float).copy() for k in (1, 2, 3, 4))
    top = np.maximum(o, c)
    h[h > 1.4 * top] = top[h > 1.4 * top]
    r = E.simulate(o, h, l, c, np.zeros(len(c)), make_gate(state.astype(np.int8), 'exit'), pr, ACCOUNT, detail=False)
    return dict(ret=float(r['ret']), maxdd=float(r['maxdd']), rounds=int(r['rounds']),
                open_ratio=float(state.mean()), busts=int(r.get('busts', 0)) if not isinstance(r.get('busts'), list) else len(r['busts']))


def summarize(results):
    rets = np.array([x['ret'] for x in results], float)
    dds = np.array([x['maxdd'] for x in results], float)
    return dict(coins=len(results), positive_pct=float((rets > 0).mean() * 100), median_ret_pct=float(np.median(rets) * 100),
                mean_ret_pct=float(rets.mean() * 100), median_maxdd_pct=float(np.median(dds) * 100),
                rounds_total=int(sum(x['rounds'] for x in results)))


def load_bars(data_root, symbol):
    p = Path(data_root) / '加密货币' / '1h' / (symbol + '_bars.npy')
    b = np.load(p, allow_pickle=False)
    if len(b) < 2 or not np.all(np.diff(b[:, 0]) == HOUR):
        raise ValueError(f'NONCONTIGUOUS_1H {symbol}')
    return b


def zigzag(data_root, symbols, cost_multiplier=1.0):
    E.FEE = .0005 * cost_multiplier
    out = []
    SKIPPED.clear()
    for sym in symbols:
        try:
            bars = load_bars(data_root, sym)
        except ValueError:
            SKIPPED.append(sym); continue
        windows, info = planning(bars)
        state = mask_from_windows(len(bars), windows)
        for resize in (False, True):
            r = run_coin(bars, state, params(.008, .012, 1.2, 14, resize))
            r.update(symbol=sym, resize=resize, windows=len(windows))
            out.append(r)
    return out


def btc_weekly(data_root, cohort_file, liquidity_file, symbols, cost_multiplier=1.0):
    E.FEE = .0005 * cost_multiplier
    cohort = [x for x in csv.DictReader(open(cohort_file, encoding='utf-8-sig')) if not symbols or x['symbol'] in symbols]
    with np.load(liquidity_file, allow_pickle=False) as z:
        days = z['days']; liquid = z['high']; names = list(z['symbols'])
    btc = load_bars(data_root, 'BTCUSDT')
    profiles = [('btc_weekly_6_2p5', dict(tf='1w', length=6, mult=2.5), False),
                ('btc_weekly_10_3', dict(tf='1w', length=10, mult=3.), False),
                ('gate_compound_1x', dict(tf='1w', length=6, mult=2.5), True)]
    out = []
    SKIPPED.clear()
    for co in cohort:
        sym = co['symbol']
        try:
            bars = load_bars(data_root, sym)
        except (ValueError, FileNotFoundError):
            SKIPPED.append(sym); continue
        ts = bars[:, 0].astype('int64')
        qualified = S.qualify_hours(ts, days, liquid[names.index(sym)])
        for name, profile, compound in profiles:
            up = S.raw_state(btc, profile)
            ix = np.searchsorted(btc[:, 0].astype('int64'), ts, side='right') - 1
            if ts[-1] > btc[-1, 0]:
                raise ValueError('BTC_DOES_NOT_COVER_ASSET_INPUT')
            raw = np.where(ix >= 0, up[np.maximum(ix, 0)], False)
            state = S.admitted(raw, qualified, ts >= int(ts[0]))
            r = run_coin(bars, state, params(.005, .01, 1.1, 20, compound))
            r.update(symbol=sym, profile=name)
            out.append(r)
    return out
