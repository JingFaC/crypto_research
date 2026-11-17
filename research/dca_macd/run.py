import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
HOUR=3_600_000
EMA_SYMBOLS=['1000BONKUSDT', '1000FLOKIUSDT', '1000PEPEUSDT', '1000SHIBUSDT', 'AAVEUSDT', 'ADAUSDT', 'AKEUSDT', 'ALGOUSDT', 'APEUSDT', 'APTUSDT', 'ARBUSDT', 'ATOMUSDT', 'AVAXUSDT', 'AXSUSDT', 'BCHUSDT', 'BLZUSDT', 'BNBUSDT', 'BTCUSDT', 'CHZUSDT', 'COMPUSDT', 'CRVUSDT', 'DASHUSDT', 'DOGEUSDT', 'DOTUSDT', 'DYDXUSDT', 'ENAUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHUSDT', 'FARTCOINUSDT', 'FETUSDT', 'FILUSDT', 'FTMUSDT', 'GALAUSDT', 'GMTUSDT', 'HBARUSDT', 'HYPEUSDT', 'INJUSDT', 'IOSTUSDT', 'IOTAUSDT', 'KAVAUSDT', 'KNCUSDT', 'LINAUSDT', 'LINKUSDT', 'LRCUSDT', 'LTCUSDT', 'LUNAUSDT', 'MANAUSDT', 'MASKUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT', 'NEOUSDT', 'ONDOUSDT', 'ONEUSDT', 'ONTUSDT', 'OPUSDT', 'ORDIUSDT', 'PAXGUSDT', 'PEOPLEUSDT', 'PIPPINUSDT', 'PUMPUSDT', 'QTUMUSDT', 'REEFUSDT', 'RLCUSDT', 'RUNEUSDT', 'SANDUSDT', 'SEIUSDT', 'SIRENUSDT', 'SOLUSDT', 'STORJUSDT', 'STXUSDT', 'SUIUSDT', 'SUSHIUSDT', 'SXPUSDT', 'TAOUSDT', 'THETAUSDT', 'TIAUSDT', 'TOMOUSDT', 'TRBUSDT', 'TRXUSDT', 'UNFIUSDT', 'UNIUSDT', 'VETUSDT', 'WAVESUSDT', 'WIFUSDT', 'WLDUSDT', 'XLMUSDT', 'XMRUSDT', 'XRPUSDT', 'XTZUSDT', 'ZECUSDT', 'ZILUSDT']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
DATA_ROOT=Path(__file__).resolve().parents[2]/'数据'
def load_raw(folder,symbol,mark=False):
 """folder=None: read new-repo 1h bars/mark; funding is not stored, so an empty event array is returned
 (cost is carried by the per-side trade fee, 0.05%x2 = 0.1% per buy+sell). folder given: legacy npz with real funding."""
 if folder is None:
  d=DATA_ROOT/'加密货币'/'1h';pb=d/(symbol+'_bars.npy')
  if not pb.is_file():raise FileNotFoundError(pb)
  b=np.load(pb,allow_pickle=False);arrays=[b,np.zeros((0,2))]
  if mark:arrays.append(np.load(d/(symbol+'_mark.npy'),allow_pickle=False))
  p=pb
 else:
  p=Path(folder)/'nautilus_1h'/(symbol+'.npz')
  if not p.is_file():raise FileNotFoundError(f'{p}: original bars/funding input required')
  with np.load(p,allow_pickle=False) as z:
   required=['bars','funding']+(['mark'] if mark else [])
   if any(k not in z.files for k in required):raise ValueError(f'{p}: requires {required}, got {z.files}')
   arrays=[z[k].copy() for k in required]
 b=arrays[0]
 if len(b)<2 or not np.all(np.diff(b[:,0])==HOUR):raise ValueError(f'NONCONTIGUOUS_1H {symbol}')
 return arrays,dict(symbol=symbol,sha256=digest(p),first_ms=int(b[0,0]),end_ms=int(b[-1,0]+HOUR),hours=len(b))
def ema_run(args):
 if __package__:
  from . import engine as E
  from .signals import CoinGate,fund_array,make_gate
  from .account import acct
 else:
  import engine as E
  from signals import CoinGate,fund_array,make_gate
  from account import acct
 if not args.ranking_file:raise ValueError('--ranking-file frozen univ.pkl is required; only load your trusted research input')
 u=pickle.load(open(args.ranking_file,'rb'));ok=u['ok'];n=len(ok);days=u['D']
 (btc,_,),_=load_raw(args.legacy_data_root,'BTCUSDT');T0=int(btc[0,0]);T=len(btc);E.FEE=.0005*args.cost_multiplier
 def top(age):
  el=np.zeros((n,days),bool);rk=np.full((n,days),9999)
  for d in range(days):
   good=np.isfinite(u['med'][:,d])&np.isfinite(u['turn'][:,d])&((d-u['first'])>=int(age*365))&(d<=u['last']);ii=np.flatnonzero(good)
   if len(ii):o=ii[np.argsort(-u['med'][ii,d])];el[o[:20],d]=True;rk[o,d]=np.arange(len(o))
  months=(np.datetime64(T0,'ms')+np.arange(days)*np.timedelta64(1,'D')).astype('datetime64[M]');cur=0
  for d in range(days):
   if d==0 or months[d]!=months[d-1]:cur=d
   el[:,d]=el[:,cur];rk[:,d]=rk[:,cur]
  return el,rk
 el,rk=top(1);el3,rk3=top(3);d23=int((np.datetime64('2023-01-01')-np.datetime64(T0,'ms'))/np.timedelta64(1,'D'));el[:,d23:]=el3[:,d23:];rk[:,d23:]=rk3[:,d23:];idx={s:i for i,s in enumerate(ok)}
 symbols=EMA_SYMBOLS;cyc={1:{},3:{}};inputs=[]
 for sym in symbols:
  (b,f),fingerprint=load_raw(args.legacy_data_root,sym);inputs.append(fingerprint);ts=b[:,0].astype('int64');o,h,l,c=[b[:,k].copy() for k in (1,2,3,4)];top_price=np.maximum(o,c);h[h>1.4*top_price]=top_price[h>1.4*top_price]
  st,va=CoinGate(ts,c).state('4h',10,30,80);va[:720]=0;fund=fund_array(ts,f[:,0].astype('int64'),f[:,1],'long')*args.cost_multiplier;flips=np.flatnonzero((np.diff(st)==1)&va[:-1].astype(bool))+1;off=int((ts[0]-T0)//HOUR)
  for lev in (1,3):
   pr=E.Params(step=.01,gx=1.,amt=1.1,tp=.015,n=20,lev=lev,base=E.base_for(10000,.95,1.1,20,1.),intrabar='opt',slip=0.,compound=False,cap=.95,fb=1.,side='long',sweep_tp=False);out=[]
   for f0 in flips:
    z=np.flatnonzero(st[f0:]==0);g=f0+int(z[0]) if len(z) else len(st)-1;sl=slice(f0,g+1);r=E.simulate(o[sl],h[sl],l[sl],c[sl],fund[sl],make_gate(st[sl],'exit'),pr,10000.,detail=False);out.append((off+int(f0),off+int(g),(np.asarray(r['eq'])/10000.).astype('float32'),int(r['busts'])>0))
   cyc[lev][sym]=out
 st0=int((np.datetime64('2021-01-01')-np.datetime64(T0,'ms'))/np.timedelta64(1,'h'));results=[]
 for lev in (1,3):
  cs=[]
  for sym,cycles in cyc[lev].items():
   ci=idx[sym]
   for a,b,path,bu in cycles:
    d=min(a//24,days-1)
    if el[ci,d]:cs.append((a,b,path,bu,rk[ci,d]))
  cs.sort(key=lambda x:(x[0],x[4]));cs=[x[:4] for x in cs];equity,info=acct(cs,st0,T,cap=10);seg=equity[st0:];daily=seg[::24];dr=np.diff(daily)/daily[:-1];yrs=len(seg)/8760;yearly={}
  for y in range(2021,2027):
   di=lambda d:int((np.datetime64(d)-np.datetime64(T0,'ms'))/np.timedelta64(1,'h'));a=equity[di(f'{y}-01-01')] if y>2021 else 1.;b=equity[di(f'{y+1}-01-01')] if y<2026 else equity[-1];yearly[str(y)]=float(b/a-1)
  cr=np.sort([float(path[-1])-1 for a,b,path,bu in cs if a>=st0]);k=max(1,int(np.ceil(len(cr)*.05)))
  results.append(dict(lev=lev,cost_multiplier=args.cost_multiplier,final=float(seg[-1]),annual=float(seg[-1]**(1/yrs)-1),dd=float((seg/np.maximum.accumulate(seg)-1).min()),sharpe=float(dr.mean()/dr.std()*np.sqrt(365)),years=yearly,cycles=len(cr),remove_top5_mean_cycle=float(cr[:-k].mean()),info=info))
 return dict(strategy='ema_dca',results=results,inputs=inputs,ranking_sha256=digest(args.ranking_file),note='Account DD uses hour-close path; removed-top5 metric is unweighted candidate-cycle mean, including cycles skipped by account capacity; not account-recomputed tail proof.')
def native_run(args):
 from decimal import Decimal
 if __package__:
  from . import native_core as N
  from . import native_signals as S
 else:
  import native_core as N
  import native_signals as S
 if not args.cohort_file or not args.liquidity_file:raise ValueError('--cohort-file and --liquidity-file frozen inputs are required')
 N.FEE=Decimal(str(.0005*args.cost_multiplier));cohort=list(csv.DictReader(open(args.cohort_file,encoding='utf-8-sig')));cohort=[x for x in cohort if not args.symbols or x['symbol'] in args.symbols.split(',')]
 with np.load(args.liquidity_file,allow_pickle=False) as z:days=z['days'];liquid=z['high'];symbols=list(z['symbols'])
 btc,_=load_raw(args.legacy_data_root,'BTCUSDT');bt=btc[0];profiles=[('btc_weekly_6_2p5',dict(tf='1w',length=6,mult=2.5)),('btc_weekly_10_3',dict(tf='1w',length=10,mult=3.))] if args.strategy=='btc_weekly_dca' else [('gate_1d_default',dict(tf='1d',length=10,mult=3.))]
 if args.strategy=='btc_weekly_dca':profiles.append(('gate_compound_1x',dict(tf='1w',length=6,mult=2.5)))
 cases=[];inputs=[]
 for co in cohort:
  sym=co['symbol'];(bars,funds,marks),fingerprint=load_raw(args.legacy_data_root,sym,True);inputs.append(fingerprint);ts=bars[:,0].astype('int64');qualified=S.qualify_hours(ts,days,liquid[symbols.index(sym)]);funds[:,1]*=args.cost_multiplier
  for name,profile in profiles:
   if name.startswith('gate_1d'):raw=S.raw_state(bars,profile)
   else:
    up=S.raw_state(bt,profile);ix=np.searchsorted(bt[:,0],ts,side='right')-1
    if ts[-1]>bt[-1,0]:raise ValueError('BTC_DOES_NOT_COVER_ASSET_INPUT')
    raw=np.where(ix>=0,up[np.maximum(ix,0)],False)
   mask=S.admitted(raw,qualified,ts>=fingerprint['first_ms']);wins=S.windows(ts,mask)
   for path in ('OHL','OLH'):
    result,trace=N.run_native(sym,bars,funds,marks,resize=False,windows=wins,path=path,leverage=1,compound=name=='gate_compound_1x',actual_funding=False,step=.005,tp=.01,amount=1.1,levels=20)
    result.update(name=name,deposit_ms=int(co['deposit_ms']));cases.append(result)
 return dict(strategy=args.strategy,cases=cases,inputs=inputs,liquidity_sha256=digest(args.liquidity_file),cohort_sha256=digest(args.cohort_file),note='Fresh native fills; OHL/OLH remain separate scenarios. This entry reports per-account results, not a fabricated portfolio Sharpe.')
def macd_run(args):
 if __package__:
  from . import macd_core as M
 else:
  import macd_core as M
 M.initialize(args.legacy_data_root);results=[]
 for start,end,tag in [('2023-10-22','2026-10-22','recent3y'),('2022-01-01','2023-10-22','earlier_bear'),('2022-01-01','2026-10-22','continuous')]:
  bars={sym:M.load_bars(sym,start,end) for sym in ('BNB','SOL')}
  for key in ('baseline','extreme_SELF_p6'):
   ens={sym:M.joint_gate(bars[sym],sym,key) for sym in bars};paths=[]
   for path in (0,1):
    rs=[M.sim_bar_open(bars[sym],*ens[sym],path,fee=.00018*args.cost_multiplier) for sym in bars];a=M.aggregate(bars['BNB'],rs);a.update(path=path,sell_fills=sum(len(r['trades']) for r in rs));paths.append(a)
   results.append(dict(period=tag,id=key,ret_min=min(a['ret'] for a in paths),dd_max=max(a['dd'] for a in paths),sharpe_min=min(a['sharpe'] for a in paths),paths=paths))
 return dict(strategy='st_macd',results=results,note='Frozen baseline and own-weekly-ST-down/daily-MACD-V extreme candidate only; fee .018% per side times multiplier, no funding/slippage in the original spot experiment.')
def daily_numpy_run(args):
 if __package__:
  from . import engine as E
  from .signals import make_gate,apply_start,fund_array
  from .daily_st import st_state
  from .aggregate import Acc,daily,sharpe
 else:
  import engine as E
  from signals import make_gate,apply_start,fund_array
  from daily_st import st_state
  from aggregate import Acc,daily,sharpe
 syms=['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','TRXUSDT','LINKUSDT','LTCUSDT'];data={};inputs=[];E.FEE=.0005*args.cost_multiplier
 for sym in syms:
  (b,f),fingerprint=load_raw(args.legacy_data_root,sym);inputs.append(fingerprint);data[sym]=(b,f)
 btc=data['BTCUSDT'][0];T0=int(btc[0,0]);T=len(btc);results=[];days0={s:daily(data[s][0][:,0].astype('int64'),np.zeros(len(data[s][0])))[0] for s in syms};D0=min(int(x[0]) for x in days0.values());D1=max(int(x[-1]) for x in days0.values());split=int(np.datetime64('2025-07-01','D').astype(int))
 pr=E.Params(step=.2/30,gx=1.,amt=1.1,tp=.015,n=30,lev=1.,base=E.base_for(10000.,1.,1.1,30,1.),intrabar='opt',slip=0.,compound=False,cap=1.,fb=1.,side='long',sweep_tp=False)
 for kind,length,mult in [('own',10,3.),('own',6,2.5),('btc',10,3.),('btc',6,2.5)]:
  acc=Acc(D0,D1);eqs=[];pnl=[]
  for sym in syms:
   b,f=data[sym];ts=b[:,0].astype('int64');source=b if kind=='own' else btc;st,va=st_state(source[:,0].astype('int64'),*[source[:,k] for k in (1,2,3,4)],length,mult)
   if kind=='btc':off=int((ts[0]-T0)//HOUR);st,va=st[off:off+len(ts)].copy(),va[off:off+len(ts)].copy()
   va[:720]=0;sr,f0=apply_start(st,va);fund=fund_array(ts,f[:,0].astype('int64'),f[:,1],'long')*args.cost_multiplier
   r=E.simulate(*[b[:,k] for k in (1,2,3,4)],fund,make_gate(sr,'exit'),pr,10000.,detail=True);eq=r['eq'];pnl.extend(float(x['pnl']) for x in r['rounds_detail']);acc.add(days0[sym],daily(ts,eq)[1]);o=int((ts[0]-T0)//HOUR);a=np.full(T,10000.);a[o:o+len(ts)]=eq;a[o+len(ts):]=eq[-1];eqs.append(a)
  stat=acc.stats();r=stat['r_act'];days=D0+1+np.arange(len(r));mask=days<split;tot=stat['tot'];hours=np.sum(eqs,axis=0);year=(D0+np.arange(len(tot))).astype('datetime64[D]').astype('datetime64[Y]').astype(int)+1970;years={}
  for y in sorted(set(year)):
   ix=np.flatnonzero(year==y);base=tot[ix[0]-1] if ix[0]>0 else 100000.;years[str(y)]=float((tot[ix[-1]]-base)/100000.)
  results.append(dict(kind=kind,length=length,mult=mult,sh=[sharpe(r),sharpe(r[mask]),sharpe(r[~mask])],ret=float(tot[-1]/tot[0]-1),hour_dd=float((hours/np.maximum.accumulate(hours)-1).min()),worst_round_pct=min(pnl)/10000*100,rounds=len(pnl),years=years))
 return dict(strategy='daily_st_numpy',results=results,inputs=inputs,note='Frozen 30-add DCA on reserved 100k, own/BTC ST10/3 or6/2.5; not the 20-add native protocol. Placebo controls not rerun by this entry.')
def first_touch_run(args):
 if __package__:
  from . import touch as T
 else:
  import touch as T
 if not args.hour_file:raise ValueError('--hour-file ETHUSDT_perp_1h.csv frozen input required')
 merged={};inputs=[]
 for rel in ('kline/ETHUSDT_15m.json','top50/ETHUSDT_15m.json'):
  p=Path(args.legacy_data_root)/rel;payload=json.loads(p.read_text(encoding='utf-8'));rows,excluded=T.closed_candles(payload,15*T.MINUTE);inputs.append(dict(path=rel,sha256=digest(p),excluded_unclosed=excluded))
  for row in rows:
   t=int(row[0])
   if t in merged and not np.array_equal(merged[t],row):raise ValueError('Conflicting 15m candles')
   merged[t]=row
 minute=np.asarray([merged[t] for t in sorted(merged)],float);hour=np.loadtxt(args.hour_file,delimiter=',',skiprows=1);T.validate(minute,15*T.MINUTE);T.validate(hour,T.HOUR);inputs.append(dict(path='external_hour_file',sha256=digest(args.hour_file)));known={int(x[0]):x[1:5] for x in hour};overlap=[np.max(np.abs(x[1:5]-known[int(x[0])])) for x in T.resample(minute,15*T.MINUTE,T.HOUR) if int(x[0]) in known]
 if not overlap or max(overlap)>1e-8:raise ValueError('15m/hour OHLC disagree')
 results={}
 for name,duration,offset in T.PERIODS:
  base,step=(minute,15*T.MINUTE) if duration<T.HOUR else (hour,T.HOUR);bars=base if duration==step else T.resample(base,step,duration,offset);res=T.simulate(base,step,T.indicators(bars),duration,fee=.0007*args.cost_multiplier);results[name]=T.summarize(res,base)
 return dict(strategy='st_first_touch',results=results,inputs=inputs,note='ETH fixed-notional ST6/2.5 first-touch/Boll20/2, seven frozen periods; no funding or liquidation.')

def zigzag_run(args):
 from decimal import Decimal
 if __package__:
  from . import native_core as N
  from .planning import planning
 else:
  import native_core as N
  from planning import planning
 N.FEE=Decimal(str(.0005*args.cost_multiplier));symbols=(args.symbols or 'BTCUSDT,1000PEPEUSDT').split(',')
 if symbols==['all']:symbols=sorted(p.stem for p in (Path(args.legacy_data_root)/'nautilus_1h').glob('*.npz'))
 results=[];inputs=[]
 for sym in symbols:
  (bars,funds,marks),fingerprint=load_raw(args.legacy_data_root,sym,True);inputs.append(fingerprint);funds[:,1]*=args.cost_multiplier;windows,info=planning(bars)
  for resize in (False,True):
   result,_=N.run_native(sym,bars,funds,marks,resize=resize,windows=windows,path='OHL',step=.008,tp=.012,amount=1.2,levels=14);result['planning']=info;results.append(result)
 return dict(strategy='zigzag_dca',results=results,inputs=inputs,note='Default is a two-coin execution diagnostic, not all-661 validation. Pass --symbols all explicitly for every local NPZ.')

def local_run(args):
 if __package__:
  from . import local_gate as G
 else:
  import local_gate as G
 if args.strategy=='zigzag_local':
  syms=args.symbols.split(',') if args.symbols else sorted(p.name[:-9] for p in (DATA_ROOT/'加密货币'/'1h').glob('*_bars.npy'))
  res=G.zigzag(DATA_ROOT,syms,args.cost_multiplier)
  by={flag:G.summarize([r for r in res if r['resize']==flag]) for flag in (False,True)}
  return dict(strategy=args.strategy,cost_multiplier=args.cost_multiplier,summary={('resize' if k else 'fixed'):v for k,v in by.items()},coins=res,note='Local numpy engine; per-coin 10000 account, 1x, no funding, 1h bars.',skipped_symbols=list(G.SKIPPED))
 if not args.cohort_file or not args.liquidity_file:raise ValueError('--cohort-file and --liquidity-file are required')
 syms=args.symbols.split(',') if args.symbols else None
 res=G.btc_weekly(DATA_ROOT,args.cohort_file,args.liquidity_file,syms,args.cost_multiplier)
 prof={n:G.summarize([r for r in res if r['profile']==n]) for n in sorted({r['profile'] for r in res})}
 return dict(strategy=args.strategy,cost_multiplier=args.cost_multiplier,summary=prof,coins=res,note='Local numpy engine; per-coin 10000 account, 1x, no funding, 1h bars.',skipped_symbols=list(G.SKIPPED))

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--strategy',required=True,choices=['ema_dca','btc_weekly_dca','daily_st_dca','st_macd','daily_st_numpy','st_first_touch','zigzag_dca','zigzag_local','btc_weekly_local']);p.add_argument('--data-root',type=Path);p.add_argument('--legacy-data-root',type=Path,default=None,help='Directory containing nautilus_1h/ and/or dca_indicator_study/5m/; never imported as code');p.add_argument('--symbols',help='Explicit comma-separated diagnostic symbols; native weekly/daily otherwise use cohort');p.add_argument('--hour-file',type=Path);p.add_argument('--ranking-file',type=Path);p.add_argument('--cohort-file',type=Path);p.add_argument('--liquidity-file',type=Path);p.add_argument('--cost-multiplier',type=float,default=1.);p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
 if args.cost_multiplier<=0:p.error('cost multiplier must be positive')
 result=({'ema_dca':ema_run,'st_macd':macd_run,'daily_st_numpy':daily_numpy_run,'st_first_touch':first_touch_run,'zigzag_dca':zigzag_run,'zigzag_local':local_run,'btc_weekly_local':local_run}.get(args.strategy,native_run))(args);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(args.output)
if __name__=='__main__':main()
