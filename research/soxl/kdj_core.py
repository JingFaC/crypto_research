"""Frozen KDJ14/EMA5,20,60 study; one dataset per process. Source attribution: SOURCES.md."""
import numpy as np
import pandas as pd
def kdj(nn):
    HH=pd.Series(h).rolling(nn,min_periods=1).max().values;LL=pd.Series(l).rolling(nn,min_periods=1).min().values;rsv=np.where(HH>LL,(c-LL)/np.where(HH>LL,HH-LL,1)*100,50.)
    K=np.zeros(n);D=np.zeros(n);K[0]=D[0]=50.
    for i in range(1,n):K[i]=(2*K[i-1]+rsv[i])/3;D[i]=(2*D[i-1]+K[i])/3
    return K,D,3*K-2*D

def cx(a,thr,up):  # 首次穿越
    p=np.concatenate([[a[0]],a[:-1]]);return (a>thr)&(p<=thr) if up else (a<thr)&(p>=thr)

def signals(nn):
    K,D,J=kdj(nn);pk=np.concatenate([[K[0]],K[:-1]]);pd_=np.concatenate([[D[0]],D[:-1]])
    gold=(K>D)&(pk<=pd_);dead=(K<D)&(pk>=pd_)
    return K,D,J,{'J<0':cx(J,0,False),'J>100':cx(J,100,True),'K<20':cx(K,20,False),'K>80':cx(K,80,True),'超卖区金叉':gold&(D<20),'超买区死叉':dead&(D>80)}

def simsig(Ls,Ss,Xl,Xs,maxh=36,fee=0.0004,f=0.10,only_reg=False):
    pos=0;rb=np.zeros(n);nf=np.zeros(n);T=[];i=60
    def close(ei,x,q,xp,end=False):
        rb[ei]+=q*(c[ei]/o[ei]-1)
        if x-1>ei:rb[ei+1:x]+=q*(c[ei+1:x]/c[ei:x-1]-1)
        if not end:rb[x]+=q*(o[x]/c[x-1]-1);nf[x]+=1
        else:nf[n-1]+=1
        nf[ei]+=1;g=100*q*(xp/o[ei]-1);a=np.searchsorted(fts,ts[ei],'left');b=np.searchsorted(fts,ts[x] if not end else ts[-1]+1,'left');fd=100*q*fr[a:b].sum()
        T.append((ts[ei],q,g,g-2*100*fee-fd,(x if not end else n)-ei,reg[ei]))
    while i<n-1:
        if pos==0:
            q=1 if Ls[i] else (-1 if Ss[i] else 0)
            if q and (not only_reg or reg[i+1]):pos=q;ei=i+1
        else:
            if (pos==1 and Xl[i]) or (pos==-1 and Xs[i]) or (i+1-ei)>=maxh:close(ei,i+1,pos,o[i+1]);pos=0
        i+=1
    if pos!=0:close(ei,n-1,pos,c[-1],True)
    D=pd.DataFrame(T,columns=['t','p','gross','net','hold','reg']);e=np.cumprod(np.maximum(1+f*(rb-nf*fee),1e-9));days=(ts[-1]-ts[0])/86400000
    s=pd.Series(e,index=pd.to_datetime(ts,unit='ms'));dd=s.resample('D').last().dropna().pct_change().dropna();sh=dd.mean()/dd.std()*np.sqrt(365) if dd.std()>0 else 0
    return D,100*(e[-1]**(365/days)-1) if e[-1]>0 else -100,100*(1-e/np.maximum.accumulate(e)).max(),sh

def ema(x,p):return pd.Series(x).ewm(alpha=2/(p+1),adjust=False).mean().values

def state(P):
    a,b,d=ema(c,P[0]),ema(c,P[1]),ema(c,P[2]);return np.where((a>b)&(b>d),1,np.where((a<b)&(b<d),-1,0))

def newly(cond):return cond&~np.concatenate([[False],cond[:-1]])

def sim(Ls,Ss,Xl,Xs,stop=None,maxh=36):
    T=[];pos=0;i=60
    while i<n-1:
        if pos==0:
            q=1 if Ls[i] else (-1 if Ss[i] else 0)
            if q and reg[i+1]:pos=q;ei=i+1;sp=o[ei]*(1-q*stop) if stop else None;mae=0.0
            i+=1;continue
        j=i;done=False   # 持仓中，检查bar j
        # 盘中止损
        if stop:
            hit=(l[j]<=sp) if pos==1 else (h[j]>=sp)
            if hit:
                xp=min(o[j],sp) if pos==1 else max(o[j],sp);x=j;done=True
        if not done:
            mae=min(mae,100*pos*((l[j]/o[ei]-1) if pos==1 else (h[j]/o[ei]-1))) if False else mae
            if (pos==1 and Xl[j]) or (pos==-1 and Xs[j]) or (j+1-ei)>=maxh:xp=o[j+1];x=j+1;done=True
        if done:
            q=pos;g=100*q*(xp/o[ei]-1);a=np.searchsorted(fts,ts[ei],'left');b=np.searchsorted(fts,ts[x],'left');fd=100*q*fr[a:b].sum()
            hi=h[ei:x+1].max() if x<n else h[ei:].max();lo=l[ei:x+1].min() if x<n else l[ei:].min();adv=100*((lo/o[ei]-1) if q==1 else -(hi/o[ei]-1))
            T.append((ts[ei],ts[min(x,n-1)],q,g,g-200*fee-fd,min(adv,0.0)));pos=0;i=j+1 if stop and x==j else x
        else:i=j+1
    return pd.DataFrame(T,columns=['t','tx','p','gross','net','mae'])

def lev(Dd,f):
    e=1.0;pk=1.0;mdd=0.0
    for x,m in zip(Dd.net.values,Dd.mae.values):
        lo=e*max(1+f*m/100-0.0008*f,0);pk=max(pk,e);mdd=max(mdd,1-lo/pk);e*=max(1+f*x/100,1e-9);pk=max(pk,e)
    return 100*(e-1),100*mdd

def evaluate(rows, funding, fee_value=0.0004, j_cross_only=False):
    global ts,o,h,l,c,n,reg,mon,fts,fr,fee
    a=np.asarray(rows,dtype=float)
    ts=a[:,0].astype(np.int64);o,h,l,c=a[:,1],a[:,2],a[:,3],a[:,4];n=len(a)
    dates=pd.to_datetime(ts,unit='ms',utc=True).tz_convert('America/New_York');mins=dates.hour*60+dates.minute
    reg=np.asarray((dates.dayofweek<5)&(mins>=570)&(mins<960));mon=dates.strftime('%Y-%m').values
    fts=np.array([x['fundingTime'] for x in funding],np.int64);fr=np.array([float(x['fundingRate']) for x in funding]);fee=fee_value
    K,D,J,_=signals(14);st=state((5,20,60))
    if j_cross_only:
        prev=np.r_[J[0],J[:-1]];L=(J<10)&(prev>=10)&(st==1);S=(J>90)&(prev<=90)&(st==-1)
    else:L=newly((J<10)&(st==1));S=newly((J>90)&(st==-1))
    trades,ann,dd,sh=simsig(L,S,K>=50,K<=50,fee=fee,f=0.1,only_reg=True)
    closed=sim(L,S,K>=50,K<=50);ret,dd1=lev(closed,1)
    def stats(x):
        count=max(1,int(np.ceil(len(x)*.05)));net=x.net.to_numpy();trim=np.delete(net,np.argsort(net)[-count:])
        return dict(trades=len(x),mean_net_pct=float(x.net.mean()),mean_gross_pct=float(x.gross.mean()),win_pct=float(100*(x.net>0).mean()),worst_pct=float(x.net.min()),net_sum_pct=float(x.net.sum()),exclude_top5pct_net_sum_pct=float(trim.sum()),long_n=int((x.p==1).sum()),short_n=int((x.p==-1).sum()))
    return dict(simsig=stats(trades),closed_only=stats(closed),simsig_10pct_notional_annualized_pct=float(ann),simsig_10pct_notional_dd_pct=float(dd),simsig_10pct_notional_sharpe=float(sh),closed_only_full_notional_return_pct=float(ret),closed_only_full_notional_dd_pct=float(dd1),note='simsig closes final open position at end; closed_only drops it. Account metrics exclude funding in mark-to-market daily curve in original simsig; trade net includes funding.')
