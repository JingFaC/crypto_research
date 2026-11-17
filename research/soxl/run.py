"""Frozen SOXL research replays. No trading, no network, no parameter search."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def legacy_repo(p):
    p=Path(p).resolve()
    for candidate in [p,p.parent,p.parent.parent]:
        if (candidate/'backtest/reports').is_dir():return candidate
    raise ValueError('--legacy-data-root must be the old repo, backtest, or backtest/data')

def load_candles(p):
    if p.suffix=='.npy':a=np.load(p,allow_pickle=False)
    else:
        o=read_json(p);o=o.get('candles',o.get('k',[])) if isinstance(o,dict) else o
        if not o:raise ValueError('No candles in '+str(p))
        a=np.array([[x['OpenTime'],x['Open'],x['High'],x['Low'],x['Close']] for x in o],float) if isinstance(o[0],dict) else np.asarray(o,dtype=float)
    if a.ndim!=2 or a.shape[1]<5 or len(a)<150:raise ValueError('Expected >=150 OHLC rows')
    a=a[:,:5]
    if not np.isfinite(a).all() or not np.all(np.diff(a[:,0])==300000):raise ValueError('Expected finite continuous 5m candles, timestamp in milliseconds')
    return a

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--strategy',required=True,choices=['kdj_ema','grid_v11','grid_v12','st3x_compound','ema_st3x'])
    ap.add_argument('--data-root',type=Path,help='New data root containing 加密货币/5m/SOXLUSDT_5m.json')
    ap.add_argument('--legacy-data-root',type=Path,help='Old repo, backtest, or backtest/data; read-only data fallback')
    ap.add_argument('--candles',type=Path);ap.add_argument('--funding',type=Path);ap.add_argument('--risk-snapshot',type=Path)
    ap.add_argument('--cost-multiplier',type=float,default=1.);ap.add_argument('--maker-fee',type=float,default=0.)
    ap.add_argument('--j-cross-only',action='store_true');ap.add_argument('--half',action='store_true',help='Only for rejected tick-based strategies')
    ap.add_argument('--output',type=Path,required=True,help='JSON result path; no CSV is generated')
    args=ap.parse_args(argv)
    if args.cost_multiplier<=0 or args.maker_fee<0:ap.error('Costs must be nonnegative and cost multiplier positive')
    old=legacy_repo(args.legacy_data_root) if args.legacy_data_root else None
    if args.strategy in ('st3x_compound','ema_st3x'):
        if old is None:ap.error('This exact tick/mark replay needs --legacy-data-root; OHLC alone is insufficient')
        if __package__:
            from . import market_core
        else:
            import market_core
        result,inputs=market_core.evaluate(old,args.strategy,args.cost_multiplier,args.half)
    else:
        candles=args.candles
        if candles is None and args.data_root:
            root=args.data_root/'数据' if (args.data_root/'数据').is_dir() else args.data_root
            candles=root/'加密货币/5m/SOXLUSDT_5m.json'
        if candles is None and old:candles=old/'backtest/reports/soxl_ema5m_st3x_2026-11-11/SOXLUSDT.klines_5m.json'
        if candles is None:ap.error('Supply --data-root, --candles, or --legacy-data-root')
        funding=args.funding or (old/'backtest/sox_grid/data/SOXLUSDT.funding.json' if old else None)
        if funding is None:ap.error('--funding is required without --legacy-data-root; zero funding is not silently assumed')
        rows=load_candles(candles);f=read_json(funding)
        times=[int(x['fundingTime']) for x in f]
        if times!=sorted(times) or len(times)!=len(set(times)):raise ValueError('Funding timestamps must be sorted and unique')
        inputs={str(candles):sha(candles),str(funding):sha(funding)}
        if args.strategy=='kdj_ema':
            if __package__:
                from . import kdj_core
            else:
                import kdj_core
            result=kdj_core.evaluate(rows,f,.0004*args.cost_multiplier,args.j_cross_only)
        else:
            risk=args.risk_snapshot or (old/'backtest/reports/soxl_sar_grid_1h_2026-11-12/inputs.json' if old else None)
            if risk is None:ap.error('--risk-snapshot is required without --legacy-data-root')
            raw=read_json(risk)
            if 'tiers' in raw:tiers=[(x['cap'],x['rate'],x['cum']) for x in raw['tiers']]
            else:
                items=raw['response']['data']['brackets'][0]['riskBrackets'];tiers=[(x['bracketNotionalCap'],x['bracketMaintenanceMarginRate'],x['cumFastMaintenanceAmount']) for x in items]
            inputs[str(risk)]=sha(risk)
            if __package__:
                from . import grid_core
            else:
                import grid_core
            result=grid_core.evaluate(rows,f,tiers,None if args.strategy=='grid_v11' else .03,args.cost_multiplier,args.maker_fee)
    envelope=dict(strategy=args.strategy,parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},input_sha256=inputs,result=result,validation_status='Previously seen historical replay; not out-of-sample or proof of a live edge')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(envelope,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
