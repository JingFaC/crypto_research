"""Exact tick-event replays of two frozen rejected SOXL strategies. Requires cached raw inputs."""
from pathlib import Path
from types import SimpleNamespace
import heapq,math,json,hashlib,zipfile
import numpy as np
import pandas as pd
if __package__:
    from .grid_core import supertrend_bands
else:
    from grid_core import supertrend_bands
L,F,S=3.,.0004,.0002
START,HALF,END=1784192700000,1789272000000,1794326400000
def maintenance(notional, tiers):
    for t in tiers:
        if notional<=t["bracketNotionalCap"]:
            return max(0.,notional*t["bracketMaintenanceMarginRate"]-t["cumFastMaintenanceAmount"])
    raise ValueError("Beyond public risk tiers")

market=SimpleNamespace(maintenance=maintenance)
source=SimpleNamespace(supertrend_bands=supertrend_bands)
run=SimpleNamespace(stamp=lambda t:str(pd.Timestamp(t,unit="ms",tz="UTC").tz_convert("Asia/Shanghai")))
class Account:
    def __init__(self,tiers,cap,clearance,record):
        self.cash=self.oracle=40000.
        self.q=self.cost=self.d=self.fees=self.funding=self.credit=0.
        self.tiers,self.cap,self.clearance,self.record=tiers,cap,clearance,record
        self.fills=[];self.funds=[];self.liq=[];self.curve=[]
        self.entries=self.cap_hits=self.too_small=0
        self.error=0.

    def equity(self,p):return self.cash+self.d*(self.q*p-self.cost)

    def enter(self,t,p,mark,d):
        if self.cash<=0:return
        execution=p*(1+d*S)
        # Use all available collateral, with the public cost formula's adverse
        # mark-price reserve and opening fee; no discretionary 80% buffer.
        assumed=execution if d==1 else max(execution,mark)
        denominator=assumed/L+max(0.,-d*(mark-execution))+F*execution
        proposed=math.floor(self.cash/denominator*100+1e-9)/100
        maximum=math.floor(self.cap/max(mark,execution)*100+1e-9)/100
        if proposed>maximum:self.cap_hits+=1
        q=min(proposed,maximum)
        if q*execution<5:self.too_small+=1;return
        self.d=d;fee=q*execution*F
        self.cash-=fee;self.fees+=fee
        self.q=q;self.cost=q*execution;self.oracle-=d*q*execution+fee
        self.entries+=1
        assert self.equity(mark)+1e-6>=q*assumed/L
        if self.record:self.fills.append(dict(timestamp=t,direction=d,action='OPEN',price=execution,
            quantity=q,fee=fee,realized=0.,position=d*q,wallet=self.cash))

    def close(self,t,p,reason='ST_FLIP'):
        if not self.q:return
        execution=p*(1-self.d*S)
        pnl=self.d*(self.q*execution-self.cost);fee=self.q*execution*F
        self.cash+=pnl-fee;self.fees+=fee
        self.oracle+=self.d*self.q*execution-fee
        if self.cash<0:
            added=-self.cash;self.credit+=added;self.cash+=added
        if self.record:self.fills.append(dict(timestamp=t,direction=-self.d,action=reason,price=execution,
            quantity=self.q,fee=fee,realized=pnl,position=0.,wallet=self.cash))
        self.q=self.cost=self.d=0.

    def fund(self,t,rate,mark):
        payment=-self.d*self.q*mark*rate
        self.cash+=payment;self.oracle+=payment;self.funding+=payment
        if self.record:self.funds.append(dict(timestamp=t,position=self.d*self.q,mark=mark,rate=rate,payment=payment))

    def mark(self,t,p):
        eq=self.equity(p);mm=market.maintenance(self.q*p,self.tiers)
        if self.q and eq<=mm:
            fee=min(max(eq,0.),self.q*p*self.clearance)
            pnl=self.d*(self.q*p-self.cost)
            self.cash+=pnl-fee;self.fees+=fee;self.oracle+=self.d*self.q*p-fee
            self.liq.append(dict(timestamp=t,equity_before=eq,maintenance=mm,fee=fee,position=self.d*self.q,mark=p))
            if self.record:self.fills.append(dict(timestamp=t,direction=-self.d,action='LIQUIDATE',price=p,
                quantity=self.q,fee=fee,realized=pnl,position=0.,wallet=self.cash))
            added=max(0.,-self.cash);self.credit+=added;self.cash+=added
            self.q=self.cost=self.d=0.
            eq=self.cash;mm=0.
        error=abs(eq-(self.oracle+self.d*self.q*p+self.credit))
        self.error=max(self.error,error);assert error<1e-4
        if self.record:self.curve.append((t,eq,self.cash,self.d*self.q,p,mm,self.fees,self.funding,self.q*p,error))
        return eq,mm

def simulate_st(start,data,record=False):
    times,prices,marks,known,up,signals,funding,tiers,cap,clearance=data
    a=Account(tiers,cap,clearance,record)
    queue=[];serial=0
    def trade_event(signal,side):
        nonlocal serial
        ix=int(np.searchsorted(times,signal,side='left'))
        if ix<len(times) and times[ix]<END:
            heapq.heappush(queue,(int(times[ix]),1,serial,'trade',side,float(prices[ix])/100.));serial+=1
    i=int(np.searchsorted(known,start,side='right')-1)
    assert i>=5
    trade_event(start,1 if up[i] else -1)
    for ts,d in signals:
        if ts>start:trade_event(ts,d)
    for f in funding:
        ts=int(f['fundingTime'])
        if start<=ts<END:
            heapq.heappush(queue,(ts,0,serial,'fund',float(f['fundingRate']),float(f['markPrice'])));serial+=1
    first=int(np.searchsorted(marks[:,0],start))
    mark=float(marks[first,1]);peak=40000.;minimum=40000.;maxdd=0.;maxlev=0.;flags=0;active=0
    for k in range(first,len(marks)):
        r=marks[k];ending=int(r[5]);states=[(a.cash,a.q,a.d,a.cost)]
        while queue and queue[0][0]<=ending:
            ts,_,_,kind,value,extra=heapq.heappop(queue)
            if kind=='fund':a.fund(ts,value,extra)
            else:
                d=int(value)
                if a.q and a.d!=d:a.close(ts,extra)
                if not a.q:a.enter(ts,extra,mark,d)
            states.append((a.cash,a.q,a.d,a.cost))
        low_buffer=float('inf')
        for cash,q,d,cost in states:
            if q:
                worst=float(r[3] if d==1 else r[2])
                low_buffer=min(low_buffer,cash+d*(q*worst-cost)-market.maintenance(q*worst,tiers))
        flags+=int(low_buffer<=0)
        old_liq=len(a.liq)
        mark=float(r[4]);eq,mm=a.mark(ending,mark)
        peak=max(peak,eq);minimum=min(minimum,eq);maxdd=max(maxdd,1-eq/peak)
        if eq>0:maxlev=max(maxlev,a.q*mark/eq)
        active+=int(a.q>0)
        if len(a.liq)>old_liq and a.cash>0:
            # Continue after forced closure if there is spendable collateral.
            nexttime=int(np.searchsorted(times,ending,side='right'))
            if nexttime<len(times):
                ts=int(times[nexttime]);j=int(np.searchsorted(known,ts,side='right')-1)
                trade_event(ts,1 if up[j] else -1)
        if a.cash<=0 and a.q==0:
            if record:
                for later in marks[k+1:]:a.curve.append((int(later[5]),0.,0.,0.,float(later[4]),0.,a.fees,a.funding,0.,a.error))
            break
    final=a.equity(float(marks[-1,4]))
    return a,dict(start=start,start_bjt=run.stamp(start),end_bjt=run.stamp(END-1),
        observation_days=(END-start)/86400000,final_equity=final,net_profit=final-40000.,
        return_pct=(final/40000.-1)*100,max_drawdown_pct=maxdd*100,peak_equity=peak,min_equity=minimum,
        liquidations=len(a.liq),entries=a.entries,cap_bound_entries=a.cap_hits,too_small_entries=a.too_small,
        fees=a.fees,funding=a.funding,final_position=a.d*a.q,max_actual_leverage=maxlev,
        active_minutes_pct=100*active/((END-start)//60000),intraminute_risk_flags=flags,oracle_error=a.error,
        bankruptcy_writeoff=a.credit)

base=SimpleNamespace(Account=Account,market=market)
def indicators(rows):
    close=np.asarray([float(r[4]) for r in rows])
    emas=[pd.Series(close).ewm(span=n,adjust=False).mean().to_numpy() for n in (5,20,60)]
    _,st,up=source.supertrend_bands(rows,6,2.5)[:3]
    gate=(emas[0]>emas[1])&(emas[1]>emas[2])&up
    gate[:59]=False
    return emas,st,up,gate

def desired_events(known,up,gate,start):
    i=int(np.searchsorted(known,start,side='right')-1)
    assert i>=59
    holding=bool(gate[i]);events=[]
    if holding:events.append((start,1))
    for j in range(i+1,len(known)):
        if known[j]>=END:break
        if holding and not up[j]:holding=False;events.append((int(known[j]),0))
        elif not holding and gate[j]:holding=True;events.append((int(known[j]),1))
    return events

def simulate_ema(start,data,record=False):
    market,known,up,gate=data
    times,prices,marks,_,_,_,funding,tiers,cap,clearance=market
    a=base.Account(tiers,cap,clearance,record)
    queue=[];serial=0
    def enqueue(signal,wanted):
        nonlocal serial
        ix=int(np.searchsorted(times,signal,side='left'))
        if ix<len(times) and times[ix]<END:
            heapq.heappush(queue,(int(times[ix]),1,serial,'trade',wanted,float(prices[ix])/100.));serial+=1
    signals=desired_events(known,up,gate,start)
    for ts,d in signals:enqueue(ts,d)
    for f in funding:
        ts=int(f['fundingTime'])
        if start<=ts<END:
            heapq.heappush(queue,(ts,0,serial,'fund',float(f['fundingRate']),float(f['markPrice'])));serial+=1
    first=int(np.searchsorted(marks[:,0],start));mark=float(marks[first,1])
    peak=minimum=40000.;dd=lev=0.;active=flags=0
    for k in range(first,len(marks)):
        r=marks[k];end=int(r[5]);states=[(a.cash,a.q,a.d,a.cost)]
        while queue and queue[0][0]<=end:
            ts,_,_,kind,value,extra=heapq.heappop(queue)
            if kind=='fund':a.fund(ts,value,extra)
            elif value:
                if not a.q:a.enter(ts,extra,mark,1)
            elif a.q:a.close(ts,extra,'ST_BEAR_EXIT')
            states.append((a.cash,a.q,a.d,a.cost))
        buffer=float('inf')
        for cash,q,d,cost in states:
            if q:
                worst=float(r[3])
                buffer=min(buffer,cash+q*worst-cost-base.market.maintenance(q*worst,tiers))
        flags+=int(buffer<=0)
        previous=len(a.liq);mark=float(r[4]);eq,_=a.mark(end,mark)
        peak=max(peak,eq);minimum=min(minimum,eq);dd=max(dd,1-eq/peak)
        if eq>0:lev=max(lev,a.q*mark/eq)
        active+=int(a.q>0)
        if len(a.liq)>previous and a.cash>0:
            ix=int(np.searchsorted(times,end,side='right'))
            if ix<len(times):
                ts=int(times[ix]);j=int(np.searchsorted(known,ts,side='right')-1)
                if gate[j]:enqueue(ts,1)
        if a.cash<=0 and not a.q:
            if record:
                for later in marks[k+1:]:a.curve.append((int(later[5]),0.,0.,0.,float(later[4]),0.,a.fees,a.funding,0.,a.error))
            break
    final=a.equity(float(marks[-1,4]))
    return a,dict(start=start,start_bjt=run.stamp(start),end_bjt=run.stamp(END-1),
        observation_days=(END-start)/86400000,final_equity=final,net_profit=final-40000.,return_pct=(final/40000.-1)*100,
        max_drawdown_pct=dd*100,peak_equity=peak,min_equity=minimum,liquidations=len(a.liq),entries=a.entries,
        cap_bound_entries=a.cap_hits,too_small_entries=a.too_small,fees=a.fees,funding=a.funding,
        active_minutes_pct=100*active/((END-start)//60000),max_actual_leverage=lev,
        intraminute_risk_flags=flags,oracle_error=a.error,bankruptcy_writeoff=a.credit,final_position=a.q)

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load_market(legacy_root):
    old=Path(legacy_root);r=old/'backtest/reports/soxl_snowball_2026-11-11'
    audit=json.loads((r/'tape_audit.json').read_text(encoding='utf-8'))
    tape=old/'backtest/reports/soxl_snowball_2026-11-11_invalid_missing_funding'
    hashes={}
    for name,key in [('times.i64','times_sha256'),('prices.i32','prices_sha256')]:
        p=tape/name;h=digest(p)
        if h!=audit[key]:raise ValueError('Raw trade tape SHA mismatch: '+str(p))
        hashes[str(p)]=h
    times=np.memmap(tape/'times.i64',dtype=np.int64,mode='r');prices=np.memmap(tape/'prices.i32',dtype=np.int32,mode='r')
    manifest=json.loads((r/'data_manifest.json').read_text(encoding='utf-8'));marks=[]
    for item in manifest:
        if item['kind']!='markPriceKlines':continue
        # Resolve relative to the user-supplied root, regardless of original host path.
        raw=item['path'].replace('\\','/');rel=raw[raw.index('backtest/'):];p=old/rel;h=digest(p)
        if h!=item['sha256']:raise ValueError('Mark archive SHA mismatch: '+str(p))
        hashes[str(p)]=h
        with zipfile.ZipFile(p) as z:
            with z.open(z.namelist()[0]) as f:
                header=0 if f.peek(50)[:1] not in b'0123456789' else None
                a=pd.read_csv(f,usecols=[0,1,2,3,4,6],header=header).to_numpy();marks.extend(a[(a[:,0]>=START)&(a[:,0]<END)])
    marks=np.asarray(marks,float)
    if len(marks)!=(END-START)//60000 or not np.all(np.diff(marks[:,0])==60000):raise ValueError('Minute mark continuity failed')
    snap=json.loads((r/'risk_snapshot.json').read_text(encoding='utf-8'));tiers=snap['response']['data']['brackets'][0]['riskBrackets'];cap=max(t['bracketNotionalCap'] for t in tiers if t['maxOpenPosLeverage']>=3)
    fp=old/'backtest/sox_grid/data/SOXLUSDT.funding.json';funding=json.loads(fp.read_text(encoding='utf-8'));hashes[str(fp)]=digest(fp)
    cp=old/'backtest/reports/sox_grid_20x_cross_st4h_2026-11-11/SOXLUSDT.klines_4h.json';rows=json.loads(cp.read_text(encoding='utf-8'));hashes[str(cp)]=digest(cp)
    _,st,up=supertrend_bands(rows)[:3];known=np.array([int(x[0])+14400000 for x in rows],np.int64);flips=np.flatnonzero((up[1:]!=up[:-1])&np.isfinite(st[1:]))+1
    signals=[(int(known[i]),1 if up[i] else -1) for i in flips if START<known[i]<END]
    return (times,prices,marks,known,up,signals,funding,tiers,cap,float(snap['liquidationFee'])),hashes
def evaluate(legacy_root,strategy,cost_multiplier=1.,half=False):
    global F,S
    F=.0004*cost_multiplier;S=.0002*cost_multiplier;data,hashes=load_market(legacy_root)
    if strategy=='st3x_compound':fun=simulate_st
    else:
        p=Path(legacy_root)/'backtest/reports/soxl_ema5m_st3x_2026-11-11/SOXLUSDT.klines_5m.json';rows=json.loads(p.read_text(encoding='utf-8'));hashes[str(p)]=digest(p)
        _,_,up,gate=indicators(rows);data=(data,np.array([int(x[0])+300000 for x in rows]),up,gate);fun=simulate_ema
    account,result=fun(HALF if half else START,data,False)
    return result,hashes
