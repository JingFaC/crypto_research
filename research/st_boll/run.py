"""Fixed ST/Boll experiments. All outputs are explicit JSON files."""
import argparse
from contextlib import redirect_stdout
from pathlib import Path
import hashlib
import io
import json
import subprocess
import shutil
import tempfile
import numpy as np
from ..datasets import DataStore
from . import daily, indicators, eth_upper, eth_middle, eth_channel

BIG20 = [s+'USDT' for s in 'BTC ETH ZEC SOL XRP NEAR DOGE BNB SUI 1000PEPE UNI WLD ADA LINK AVAX FIL AAVE ARB BCH LTC'.split()]
OLD10 = [s+'USDT' for s in 'BTC ETH BNB SOL XRP DOGE ADA AVAX LINK LTC'.split()]


def prices(store, symbol):
    bars = store.array(f'backtest/data/nautilus_1h/{symbol}.npz')
    bars = bars[np.argsort(bars[:, 0])]
    bars = bars[np.r_[True, np.diff(bars[:, 0]) > 0]]
    return indicators.resample(bars[:, 0].astype('int64'), *(bars[:, k].astype(float) for k in (1, 2, 3, 4)), '1d')


def stats(t, eq, worst, trades):
    years = (t[-1]-t[0])/3.15576e10
    day_end = np.r_[np.flatnonzero(np.diff(t//86400000)), len(t)-1]
    day_eq = eq[day_end]
    previous = np.r_[1., day_eq[:-1]]
    returns = np.divide(day_eq, previous, out=np.ones_like(day_eq), where=previous > 0)-1
    sr = returns.mean()/returns.std()*np.sqrt(365) if returns.std() else 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        annual = {str(k): float(v) if np.isfinite(v) else None for k, v in daily.yearly(t, eq).items()}
    dd = float(daily.dd_of(eq, worst))
    return dict(trades=len(trades), total_pct=float((eq[-1]-1)*100),
                cagr_pct=float((max(eq[-1], 1e-12)**(1/years)-1)*100),
                dd_pct=dd, sharpe=float(sr), annual_pct=annual,
                start_ms=int(t[0]), end_ms=int(t[-1]), days=len(t),
                net_trade_returns=[float(v) for v in trades])


def portfolio(items, mode):
    times = np.arange(min(x[0][0] for x in items), max(x[0][-1] for x in items)+1, 86400000)
    rr = np.full((len(items), len(times)), np.nan)
    ww = rr.copy()
    for i, (ts, curves) in enumerate(items):
        eq, worst = curves[mode]
        ix = np.searchsorted(times, ts)
        prev = np.r_[eq[0], eq[:-1]]
        prev = np.where(prev <= 0, np.nan, prev)
        rr[i, ix] = np.nan_to_num(eq/prev-1)
        ww[i, ix] = np.nan_to_num(worst/prev-1)
        rr[i, ix[0]] = ww[i, ix[0]] = 0
    ret, worst_ret = np.nan_to_num(np.nanmean(rr, axis=0)), np.nan_to_num(np.nanmean(ww, axis=0))
    # Daily equal-weight rebalance cost: members drift to (1+r_i)/(1+mean) of 1/n; trading back costs fee on |dw| (both sides).
    n_t = np.sum(~np.isnan(rr), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        drift = (1+np.nan_to_num(rr))/(1+ret)
        turnover = np.nansum(np.where(np.isnan(rr), 0., np.abs(1./np.maximum(n_t, 1)*(1-drift))), axis=0)
    cost = daily.FEE*turnover
    ret, worst_ret = ret-cost, worst_ret-cost
    eq = np.cumprod(1+ret)
    return times, eq, np.r_[1., eq[:-1]]*(1+worst_ret)


def run(args):
    store = DataStore(args.data_root, args.legacy_data_root)
    fee = args.fee * args.cost_multiplier
    result = dict(strategy=args.strategy, fee_per_side=fee, funding_per_8h=0.0001*args.cost_multiplier,
                  validation='historical reproduction, not fresh out-of-sample')
    daily.FEE = eth_upper.FEE = eth_middle.FEE = eth_channel.FEE = fee
    daily.FUND8H = 0.0001 * args.cost_multiplier
    if args.strategy in ('daily_st', 'daily_pool', 'daily_big'):
        syms = [args.symbol] if args.strategy == 'daily_st' else Path(__file__).with_name('universe.txt').read_text().split()
        if args.strategy == 'daily_big':
            syms = sorted(set(BIG20 + OLD10))
        records, curves = {}, {}
        for symbol in syms:
            t, o, h, l, c = prices(store, symbol)
            if len(c) < 300:
                continue
            d, _, _ = indicators.st_both(h, l, c, 6, 2.5)
            modes, cc = {}, {}
            for mode in ('L', 'LS'):
                eq, worst, tr = daily.run_fast(o, h, l, c, d, mode, 86400000)
                modes[mode] = stats(t, eq, worst, tr)
                cc[mode] = (eq, worst)
            modes['hold_total_pct'] = float((c[-1]/c[0]-1)*100)
            cc['H'] = (c/c[0], c/c[0])
            records[symbol], curves[symbol] = modes, (t, cc)
        result['coins'] = records
        if args.strategy == 'daily_pool':
            result['pools'] = {}
            for years in (1, 3, 5):
                chosen = [v for v in records.values() if years == 1 or v['L']['days'] >= years*365]
                result['pools'][str(years)] = dict(count=len(chosen),
                    positive_pct=100*float(np.mean([v['L']['total_pct'] > 0 for v in chosen])),
                    cagr_median_pct=float(np.median([v['L']['cagr_pct'] for v in chosen])),
                    dd_median_pct=float(np.median([v['L']['dd_pct'] for v in chosen])),
                    beat_hold_pct=100*float(np.mean([v['L']['total_pct'] > v['hold_total_pct'] for v in chosen])))
        if args.strategy == 'daily_big':
            result['portfolios'] = {}
            for name, members in [('BIG20', BIG20), ('OLD10', OLD10), ('TOP4', OLD10[:4])]:
                if any(s not in curves for s in members):
                    raise ValueError('Missing portfolio member; refusing silent exclusion')
                for mode in ('L', 'LS', 'H'):
                    t, eq, worst = portfolio([curves[s] for s in members], mode)
                    result['portfolios'][name+'_'+mode] = stats(t, eq, worst, [])
                    result['portfolios'][name+'_'+mode]['trades'] = None
                    result['portfolios'][name+'_'+mode]['account_note'] = 'Daily equal weight across available members; rebalance turnover cost = fee x sum|dw| charged each day'
    elif args.strategy in ('eth_boll', 'eth_channel'):
        path = store.resolve(f'backtest/data/dca_indicator_study/5m/{args.symbol}_5m.npy')
        bars = np.load(path, allow_pickle=False)
        t, o, h, l, c = bars[:, 0].astype('int64'), *(bars[:, k] for k in (1, 2, 3, 4))
        result['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result['start_ms'], result['end_ms'] = int(t[0]), int(t[-1])
        result['versions'] = {}
        for label, module in [('upper', eth_upper), ('middle', eth_middle)] if args.strategy == 'eth_boll' else [('channel', eth_channel)]:
            for once in (False, True) if args.strategy == 'eth_boll' else (False,):
                for side in (1,) if args.strategy == 'eth_boll' else (1, -1):
                    kw = dict(length=6, mult=2.5, comp=True)
                    if args.strategy == 'eth_boll':
                        kw.update(once=once, fund8h=0.)
                    else:
                        kw['side'] = side
                        kw['fund8h'] = 0.0001*args.cost_multiplier if side == 1 else 0.
                    replay = module.run(t, o, h, l, c, **kw)
                    eq, worst, tr = replay['eqc'], replay['eql'], replay['trades']
                    rets = np.array([side*(row[3]/row[2]-1)-2*fee for row in tr])
                    wins, losses = rets[rets > 0], rets[rets <= 0]
                    item = stats(t, eq, worst, rets)
                    item.update(win_pct=float(np.mean(rets > 0)*100), payoff=float(wins.mean()/-losses.mean()) if len(wins) and len(losses) else None)
                    result['versions'][f'{label}_once{once}_side{side}'] = item
    else:
        raise ValueError('Unknown strategy')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', choices=['daily_st', 'daily_pool', 'daily_big', 'eth_boll', 'eth_channel', 'matrix', 't2'], required=True)
    parser.add_argument('--data-root', default=str(Path(__file__).resolve().parents[2]/'数据'))
    parser.add_argument('--legacy-data-root', help='Old backtest/data directory; data only, no imported source')
    parser.add_argument('--output', required=True)
    parser.add_argument('--symbol', default='ETHUSDT')
    parser.add_argument('--fee', type=float, default=0.0005)
    parser.add_argument('--split', choices=['tr', 'va', 'te'], default='te')
    parser.add_argument('--source-data', help='Frozen k63_<split> directory for Go strategies')
    parser.add_argument('--cost-multiplier', type=float, default=1.)
    args = parser.parse_args(argv)
    if args.fee < 0 or args.cost_multiplier <= 0:
        parser.error('Costs must be nonnegative, multiplier must be positive')
    if args.strategy in ('matrix', 't2'):
        with tempfile.TemporaryDirectory(prefix='crypto_st_boll_') as stage:
            source_data = args.source_data
            if not source_data:
                store = DataStore(args.data_root, args.legacy_data_root)
                members = [relative for relative, kind in store.index if kind == '原文件' and relative.startswith(f'data/k63_{args.split}/') and relative.endswith('_4h.json')]
                if not members:
                    parser.error('Frozen split absent from data manifest; provide --source-data')
                for relative in members:
                    shutil.copyfile(store.resolve(relative), Path(stage)/Path(relative).name)
                source_data = stage
            command = ['go', 'run', '-mod=readonly', './engine', '-data', str(Path(source_data).resolve()), '-entry', 't_matrix' if args.strategy == 'matrix' else 'phase_t2', '-include_btc', '-interval', '4h', '-min_bars', '250' if args.strategy == 'matrix' else '500', '-fee_bps', str(10*args.cost_multiplier), '-slip_bps', str(5*args.cost_multiplier)]
            if args.strategy == 't2':
                command += ['-t2_target_atr', '1.5', '-t2_stop_atr', '1.5', '-t2_hold', '48', '-t2_risk_pct', '0.0025', '-t2_max_alloc', '0.10', '-t2_gap', '0']
            proc = subprocess.run(command, cwd=Path(__file__).parent, capture_output=True, text=True, encoding='utf-8')
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        result = dict(strategy=args.strategy, command=command, stdout=proc.stdout,
                      warning='Current-source reproduction. The old T2 report specifies 99% partial and 0.5ATR trail, which this source does not implement.')
    else:
        result = run(args)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(str(target))


if __name__ == '__main__':
    main()
