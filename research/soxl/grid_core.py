"""Frozen 0.5% ST/SAR grid, pessimistic intrabar model. Source attribution: SOURCES.md."""
import math,bisect
import numpy as np
import pandas as pd
MMV2=[[2,0,0],[-1,-1,-1],[1,1,1],[0,0,2]]
def maint(tiers,n):
    for cap,rate,cum in tiers:
        if n<=cap: return max(0,n*rate-cum)
    c,r,m=tiers[-1]; return max(0,n*r-m)

class Res: pass

def run(D,mapping,rng='full',start=None,end=None,order='pess',maker=0.0,fee=0.0004,slip=0.0002,cap=40000.0,fund=True,rec=False,init_cash=40000.0,stops=None,gate=None,gate_mode='entry',gl=None,gs=None,mmap=None,sp=1.0025,lev=3.0,reinv=None,rext=0.0,rmin=0.0,rmode='ext',rfa=0.0,rff=0.0,ia=False,dly=0,noinit=False,trace=None,idle=False):
    K,S,F,tiers=D
    Kt,Ko,Kh,Kl,Kc=K['t'],K['o'],K['h'],K['l'],K['c']
    sclose=S.close.values; sk=S.known_time.values; slo=S.lower.values; shi=S.upper.values; sst=S.st_direction.values; ssar=S.sar_direction.values
    i0=int(np.searchsorted(Kt,start)); i1=int(np.searchsorted(Kt,end)) if end else len(Kt)
    sidx=np.searchsorted(sk,Kt-dly,'right')-1
    fidx={}
    if fund:
        for ft,fr,fm in F:
            j=int(np.searchsorted(Kt,ft,'right'))-1
            if i0<=j<i1: fidx.setdefault(j,[]).append((fr,fm))
    cash=init_cash; pos=0.0; cost=0.0; mode=0; lv=[]; cnt=0; q=0.0; fr_=0; started=False; blocked=True; d=0; fmx=10**9; fmn=-1; lastf=0
    EQ=np.full(len(Kt),np.nan)
    fees=0.0; fundtot=0.0; liq=0; rej=0; peak=init_cash; mindd=0.0; mineq=init_cash
    pair=0.0; cycles=[]; cyc=None; prev=-1; stopped=False; fills=0; notion=0.0
    stops=stops or {}; lock=False; nstop=0; cur_t=0
    def eq(p): return cash+pos*p/100.0-cost
    def reserve(p,f,position):
        o_=0.0
        if not blocked:
            if d==1:
                for k in range(f): o_+=lv[k]/100.0
            else:
                for k in range(f+1,cnt+1): o_+=max(p/100.0,lv[k]/100.0)
        return (abs(position)*p/100.0+q*o_)/lev
    def force_close(px,reason):
        nonlocal cash,pos,cost,fees,notion,fills,cyc,lv,cnt,q,blocked,mode,d,lock,nstop
        if pos!=0:
            sign=-1 if pos>0 else 1; ex=px*(1+sign*slip); fe=abs(pos)*ex*fee
            cash+=pos*ex-cost-fe; fees+=fe; notion+=abs(pos)*ex; pos=0.0; cost=0.0; fills+=1
        if cyc is not None:
            cyc['t1']=cur_t; cyc['e1']=cash; cyc['pr1']=pair; cyc['why']=reason; cycles.append(cyc); cyc=None
        lv=[]; cnt=0; q=0.0; blocked=True; mode=0; d=0; lock=True; nstop+=1
    for i in range(i0,i1):
        o=int(Ko[i]); h=int(Kh[i]); l=int(Kl[i]); c=int(Kc[i])
        si=sidx[i]; cur_t=int(Kt[i])
        if stops.get('time_h') and mode!=0 and cyc is not None and Kt[i]-cyc['t0']>=stops['time_h']*3600000:
            force_close(o/100.0,'TIME')
        if si!=prev and si>=1 and stops.get('band') and mode!=0 and cyc is not None:
            if (d==1 and sclose[si]<slo[si-1]) or (d==-1 and sclose[si]>shi[si-1]): force_close(o/100.0,'BAND')
        if si!=prev and si>=0:
            prev=si
            st=sst[si]; sar=ssar[si]
            state=(0 if st==1 else 2)+(0 if sar==1 else 1)
            des=mapping[state] if mmap is None else mmap[state][mode+1]
            if des==2: des=mode
            if des==1 and gl is not None and (mode!=1) and not gl[si]: des=mode if mode!=-1 else 0
            if des==-1 and gs is not None and (mode!=-1) and not gs[si]: des=mode if mode!=1 else 0
            if gate is not None and des!=0 and (gate_mode=='both' or mode==0) and not gate[si]: des=0
            if lock:
                if des==0: lock=False
                else: des=mode
            if des!=mode:
                if pos!=0:
                    sign=-1 if pos>0 else 1; ex=o/100.0*(1+sign*slip); fe=abs(pos)*ex*fee
                    cash+=pos*ex-cost-fe; fees+=fe; notion+=abs(pos)*ex; pos=0.0; cost=0.0; fills+=1
                if cyc is not None:
                    cyc['t1']=int(Kt[i]); cyc['e1']=cash; cyc['pr1']=pair; cycles.append(cyc); cyc=None
                lv=[]; cnt=0; q=0.0; blocked=True; mode=des; d=des
                if mode!=0:
                    lo=slo[si]; hi=shi[si]
                    if rng=='lower': hi=(lo+hi)/2.0
                    elif rng=='upper': lo=(lo+hi)/2.0
                    skip_=False
                    if rmin>0.0 and hi/lo-1.0<rmin:
                        if rmode=='skip': skip_=True
                        elif d==1: lo=hi/(1.0+rmin)
                        else: hi=lo*(1.0+rmin)
                    if rext>0.0:
                        if d==1: lo=lo*(1.0-rext)
                        else: hi=hi*(1.0+rext)
                    if rfa!=0.0 or rff!=0.0:
                        wd=hi-lo
                        if d==1: lo,hi=lo+rfa*wd,hi-rff*wd
                        else: hi,lo=hi-rfa*wd,lo+rff*wd
                    cnt=int(round(math.log(hi/lo)/math.log(sp)))
                    if skip_ or cnt<1 or cnt>1000: mode=0; d=0; cnt=0
                    else:
                        ratio=(hi/lo)**(1.0/cnt); lv=[int(math.floor(lo*ratio**k*100+1e-7)) for k in range(cnt+1)]
                        mark=o/100.0; denom=0.0
                        for k in range(cnt):
                            e=lv[k+(0 if d==1 else 1)]/100.0
                            asm=e if d==1 else max(mark,e)
                            denom+=asm/lev+max(-d*(mark-e),0)+0.0006*max(mark,e)
                        budget=min(cap,max(0.0,cash)) if reinv is None else min(max(0.0,cash),init_cash+reinv*max(cash-init_cash,0.0)); q=math.floor(budget/denom*100+1e-9)/100.0
                        fr_=cnt if d==1 else 0; blocked=False; started=False; fmx=10**9; fmn=-1
                        cyc=dict(t0=int(Kt[i]),e0=cash,mode=mode,p0=o/100.0,liq=0,pr0=pair,q=q,lv=list(lv))
                        if q*lv[0]/100.0<5 or eq(o)+1e-7<reserve(o,fr_,pos): blocked=True; rej+=1; (cyc.__setitem__('blk',cyc.get('blk',0)+1) if cyc is not None else None)
        if fund and i in fidx:
            for fr,fm in fidx[i]:
                pay=-pos*fm*fr; cash+=pay; fundtot+=pay
        if mode!=0 and cnt>0:
            if order=='pess': pts=(o,l,h,c) if d==-1 else (o,h,l,c)
            else: pts=(o,h,l,c) if d==-1 else (o,l,h,c)
            for p in pts:
                # trade at point p
                if not started:
                    ial=(ia is True) or (ia=='long'); ias=(ia is True) or (ia=='short')
                    if (not ial) and d==1 and p<lv[0]:
                        if idle and noinit: started=True; blocked=True; fmx=-1
                        continue
                    if (not ias) and d==-1 and p>lv[-1]:
                        if idle and noinit: started=True; blocked=True; fmn=10**9
                        continue
                    started=True
                    if (((ial if d==1 else ias)) or lv[0]<=p<=lv[-1]) and not blocked:
                        tg=0
                        if d==1:
                            while tg<cnt and lv[tg]<=p: tg+=1
                        else:
                            while tg<cnt and lv[tg+1]<p: tg+=1
                        if noinit: fr_=tg; fmx=(tg if d==1 else 10**9); fmn=(tg if d==-1 else -1)
                        while (not noinit) and fr_!=tg:
                            nx=fr_-d; ex=p/100.0*(1+d*slip); fe=q*ex*fee; npos=pos+d*q
                            e2=eq(p)+d*q*(p/100.0-ex)-fe
                            if e2+1e-7<reserve(p,nx,npos): blocked=True; rej+=1; (cyc.__setitem__('blk',cyc.get('blk',0)+1) if cyc is not None else None); break
                            pos=npos; cost+=d*q*ex; cash-=fe; fees+=fe; fills+=1; notion+=q*ex; fr_=nx
                if d==1:
                    while not blocked and fr_>0 and p<lv[fr_-1]:
                        L=lv[fr_-1]/100.0; nx=fr_-1; fe=q*L*maker; npos=pos+q
                        if eq(p)+q*(p/100.0-L)-fe+1e-7<reserve(p,nx,npos): blocked=True; rej+=1; (cyc.__setitem__('blk',cyc.get('blk',0)+1) if cyc is not None else None); break
                        pos=npos; cost+=q*L; cash-=fe; fees+=fe; fills+=1; notion+=q*L; fr_=nx
                    while fr_<cnt and fr_<fmx and p>lv[fr_+1]:
                        L=lv[fr_+1]/100.0; pair+=q*(lv[fr_+1]-lv[fr_])/100.0; avg=cost/pos; cash+=(L-avg)*q-q*L*maker; fees+=q*L*maker; pos-=q; cost-=q*avg; fills+=1; notion+=q*L; fr_+=1
                        if abs(pos)<1e-7: pos=0.0; cost=0.0
                else:
                    while not blocked and fr_<cnt and p>lv[fr_+1]:
                        L=lv[fr_+1]/100.0; nx=fr_+1; fe=q*L*maker; npos=pos-q
                        if eq(p)-q*(p/100.0-L)-fe+1e-7<reserve(p,nx,npos): blocked=True; rej+=1; (cyc.__setitem__('blk',cyc.get('blk',0)+1) if cyc is not None else None); break
                        pos=npos; cost-=q*L; cash-=fe; fees+=fe; fills+=1; notion+=q*L; fr_=nx
                    while fr_>0 and fr_>fmn and p<lv[fr_-1]:
                        L=lv[fr_-1]/100.0; pair+=q*(lv[fr_]-lv[fr_-1])/100.0; avg=cost/pos; cash+=(avg-L)*q-q*L*maker; fees+=q*L*maker; pos+=q; cost+=q*avg; fills+=1; notion+=q*L; fr_-=1
                        if abs(pos)<1e-7: pos=0.0; cost=0.0
                if trace is not None and fills!=lastf: trace.append((i,p,fills,round(pos,4),fr_)); lastf=fills
                e=eq(p)
                if pos!=0 and e<=maint(tiers,abs(pos)*p/100.0):
                    fe=min(max(e,0),abs(pos)*p/100.0*0.015); cash=max(0.0,cash+pos*p/100.0-cost-fe); pos=0.0; cost=0.0; liq+=1
                    if cyc: cyc['liq']=1; cyc['pr1']=pair; cyc['t1']=int(Kt[i]); cyc['e1']=cash; cycles.append(cyc); cyc=None
                    stopped=True; mineq=min(mineq,cash); break
                if e<mineq: mineq=e
                EQ[i]=e
                if e>peak: peak=e
                dd=1-e/peak
                if dd>mindd: mindd=dd
                if stops and stops.get('tp') is not None and cyc is not None and lv:
                    _tp=stops['tp']
                    if (d==1 and p>=int(lv[-1]*(1+_tp))) or (d==-1 and p<=int(lv[0]*(1-_tp))):
                        force_close(p/100.0,'TP'); break
                if stops and pos!=0:
                    trig=None
                    if stops.get('depth') is not None:
                        dp=stops['depth']
                        if d==1:
                            th=int(lv[0]+dp*(lv[-1]-lv[0]))
                            if p<=th: trig=(min(th,o) if stops.get('fill','ext')=='thr' else p)
                        else:
                            th=int(lv[-1]-dp*(lv[-1]-lv[0]))+1
                            if p>=th: trig=(max(th,o) if stops.get('fill','ext')=='thr' else p)
                    if trig is None and stops.get('edge') is not None:
                        buf=stops['edge']
                        if d==1:
                            th=int(lv[0]*(1-buf))
                            if p<=th: trig=(min(th,o) if stops.get('fill','ext')=='thr' else p)
                        else:
                            th=int(lv[-1]*(1+buf))+1
                            if p>=th: trig=(max(th,o) if stops.get('fill','ext')=='thr' else p)
                    if trig is None and stops.get('eq_r'):
                        net=e-abs(pos)*p/100.0*(slip+fee)
                        if net<=cyc['e0']-stops['eq_r']*min(cap,cyc['e0']): trig=p
                    if trig is not None:
                        force_close(trig/100.0,'STOP'); break
            if stopped: break
        else:
            e=cash
            EQ[i]=e
            if e>peak: peak=e
            dd=1-e/peak
            if dd>mindd: mindd=dd
    fin=cash if stopped else eq(int(Kc[i1-1]))
    if cyc is not None and not stopped:
        cyc['t1']=int(Kt[i1-1]); cyc['e1']=fin; cyc['pr1']=pair; cyc['open']=1; cycles.append(cyc)
    r=Res(); r.final=fin; r.ret=(fin/init_cash-1)*100; r.dd=mindd*100; r.liq=liq; r.rej=rej; r.cycles=cycles; r.fees=fees; r.fund=fundtot; r.fills=fills; r.notion=notion; r.mineq=mineq; r.pair=pair; r.nstop=nstop; r.eq=EQ
    return r

def supertrend_bands(rows, length=6, multiplier=2.5):
    """The existing ST recurrence, retaining BOTH final (not basic) bands."""
    a = np.asarray(rows, dtype=float)
    h, l, c = a[:, 2], a[:, 3], a[:, 4]
    tr = np.maximum(h-l, np.maximum(abs(h-np.r_[c[0], c[:-1]]), abs(l-np.r_[c[0], c[:-1]])))
    atr, line, lower, upper = (np.full(len(c), np.nan) for _ in range(4))
    up = np.zeros(len(c), dtype=bool)
    if len(c) < length:
        return atr, line, up, lower, upper
    atr[length-1] = tr[:length].mean()
    bullish = False
    for i in range(length-1, len(c)):
        if i >= length:
            atr[i] = (atr[i-1]*(length-1)+tr[i])/length
        bu, bl = (h[i]+l[i])/2+multiplier*atr[i], (h[i]+l[i])/2-multiplier*atr[i]
        if i == length-1:
            upper[i], lower[i] = bu, bl
        else:
            upper[i] = bu if bu < upper[i-1] or c[i-1] > upper[i-1] else upper[i-1]
            lower[i] = bl if bl > lower[i-1] or c[i-1] < lower[i-1] else lower[i-1]
            bullish = c[i] >= lower[i] if bullish else c[i] > upper[i]
        line[i], up[i] = (lower[i] if bullish else upper[i]), bullish
    return atr, line, up, lower, upper

def sar(a):
    # 同协作者 run_soxl_sar_grid.py 的 sar()（TradingView 递推）：列 [t,o,h,l,c]
    a=np.asarray(a,dtype=float);h,l,c=a[:,2],a[:,3],a[:,4]
    values=np.full(len(a),np.nan);direction=np.zeros(len(a),dtype=np.int8)
    if len(a)<2:return values,direction
    up=c[1]>c[0];value=l[0] if up else h[0];ep=h[1] if up else l[1];af=.02
    for i in range(1,len(a)):
        value+=af*(ep-value)
        reverse=(value>l[i]) if up else (value<h[i])
        if reverse:
            value=max(ep,h[i]) if up else min(ep,l[i]); up=not up;ep=h[i] if up else l[i];af=.02
        elif i>1:
            extreme=h[i]>ep if up else l[i]<ep
            if extreme:ep=h[i] if up else l[i];af=min(.2,af+.02)
        previous=l if up else h
        value=min(value,previous[i-1]) if up else max(value,previous[i-1])
        if i>1:value=min(value,previous[i-2]) if up else max(value,previous[i-2])
        values[i]=value;direction[i]=1 if up else -1
    return values,direction

def st_of(S): return np.where(S.st_direction.values==1,0,2)+np.where(S.sar_direction.values==1,0,1)

def apply_stop(S,kind,x=0.0):
    S=S.copy(); cl=S.close.values; stt=st_of(S); lo=S.lower.values; hi=S.upper.values; n=len(S)
    sd=S.st_direction.values.copy(); sa=S.sar_direction.values.copy()
    mode=0; lock=0; ent=None; nstop=0
    for i in range(n):
        s=stt[i]
        if lock!=0:
            # 锁定区: 多单被止损后, 真实状态仍在{2,3}; 空单被止损后仍在{0,1}
            if (lock==1 and s in (2,3)) or (lock==-1 and s in (0,1)):
                sd[i],sa[i]=(1,1) if lock==1 else (-1,-1); continue
            lock=0
        new=MMV2[s][mode+1]
        if new==2: new=mode
        if mode==0 and new!=0 and not gx[i]: new=0
        if new!=mode and new!=0: ent=(lo[i],hi[i])      # 开仓/反手：记下开仓时的ST带
        mode=new
        if mode!=0 and ent is not None and i>0:
            hit=False
            if kind=='A': hit=(mode==1 and s==3) or (mode==-1 and s==0)
            if kind=='B':
                hit=(mode==1 and hi[i]<ent[1]*(1-x)) or (mode==-1 and lo[i]>ent[0]*(1+x))
            if kind=='C': hit=(mode==1 and cl[i]<ent[0]*(1-x)) or (mode==-1 and cl[i]>ent[1]*(1+x))
            # 开仓那一行不判断
            if hit and not (new!=0 and ent==(lo[i],hi[i]) and mode!=0 and False):
                nstop+=1; STOPS.append(mode); lock=mode; sd[i],sa[i]=(1,1) if mode==1 else (-1,-1); mode=0
    S['st_direction']=sd; S['sar_direction']=sa; return S,nstop

def derive_signals(rows):
    a=np.asarray(rows,dtype=float);t=a[:,0].astype(np.int64);block=t//3600000
    d=pd.DataFrame(dict(b=block,t=t,o=a[:,1],h=a[:,2],l=a[:,3],c=a[:,4]));gp=d.groupby('b')
    hourly=pd.DataFrame(dict(t=gp.t.min(),n=gp.size(),o=gp.o.first(),h=gp.h.max(),l=gp.l.min(),c=gp.c.last()))
    hourly=hourly[hourly.n==12].reset_index(drop=True);bars=hourly[['t','o','h','l','c']].to_numpy()
    atr,line,up,lower,upper=supertrend_bands(bars,6,2.5);sv,sd=sar(bars)
    signal=pd.DataFrame(dict(known_time=hourly.t.values+3600000,close=hourly.c.values,st_direction=np.where(up,1,-1),lower=lower,upper=upper,sar=sv,sar_direction=sd))
    return signal.dropna(subset=['lower','upper']).loc[lambda x:x.sar_direction!=0].reset_index(drop=True)
def evaluate(rows, funding, tiers, stop=0.03, cost_multiplier=1., maker=0., start=1779033600000, end=1794326400000):
    global gx,STOPS
    a=np.asarray(rows,dtype=float);K=dict(t=a[:,0].astype(np.int64),**{k:np.round(a[:,i]*100).astype(np.int64) for i,k in enumerate('ohlc',1)})
    S=derive_signals(a);w=pd.Series((S.upper.values-S.lower.values)/S.close.values);gx=(w<w.rolling(120,min_periods=100).median()).fillna(False).values;STOPS=[]
    Sx,nstop=apply_stop(S,'B',stop) if stop is not None else (S,0)
    F=[(int(x['fundingTime']),float(x['fundingRate']),float(x['markPrice'])) for x in funding]
    r=run((K,Sx,F,tiers),[0,0,0,0],'full',start,end,cap=20000.,init_cash=20000.,lev=3.,sp=1.005,gate=gx,gate_mode='entry',reinv=None,mmap=MMV2,noinit=True,idle=True,fee=.0004*cost_multiplier,slip=.0002*cost_multiplier,maker=maker)
    e=pd.Series(r.eq,index=K['t']).loc[start:end].ffill().fillna(20000.);daily=pd.Series(e.values,index=pd.to_datetime(e.index,unit='ms')).resample('1D').last().ffill().pct_change().dropna();pnl=np.array([x['e1']-x['e0'] for x in r.cycles]);k=max(1,int(np.ceil(len(pnl)*.05)));trim=np.delete(pnl,np.argsort(pnl)[-k:])
    return dict(return_pct=float(r.ret),dd_pct=float(100*(1-e.values/np.maximum.accumulate(e.values)).max()),sharpe=float(daily.mean()/daily.std()*np.sqrt(365)),cycles=len(pnl),liquidations=int(r.liq),min_equity_ratio=float(e.min()/20000),exclude_top5pct_cycle_pnl_pct_initial=float(trim.sum()/20000*100),stop_signals=nstop)
