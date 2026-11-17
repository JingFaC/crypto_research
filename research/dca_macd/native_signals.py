import numpy as np
HOUR=3_600_000
DAY=24*HOUR
TFS={"1d":(DAY,0),"1w":(7*DAY,4*DAY)}
def st_direction(h,l,c,length=10,mult=3):
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.r_[c[0],c[:-1]]),np.abs(l-np.r_[c[0],c[:-1]])))
    atr=np.full(len(c),np.nan);out=np.zeros(len(c),dtype=bool)
    if len(c)<length:return out
    atr[length-1]=tr[:length].mean()
    for i in range(length,len(c)):atr[i]=(atr[i-1]*(length-1)+tr[i])/length
    upper=lower=0.;up=False
    for i in range(length-1,len(c)):
        bu=(h[i]+l[i])/2+mult*atr[i];bl=(h[i]+l[i])/2-mult*atr[i]
        if i==length-1:upper,lower=bu,bl
        else:
            upper=bu if bu<upper or c[i-1]>upper else upper
            lower=bl if bl>lower or c[i-1]<lower else lower
            up=(c[i]>=lower) if up else (c[i]>upper)
        out[i]=up
    return out

def windows(ts,mask):
    cuts=np.r_[0,np.flatnonzero(np.diff(mask.astype(np.int8)))+1,len(ts)]
    out=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        if mask[a]:out.append(dict(open_ms=int(ts[a]),close_ms=int(ts[b]) if b<len(ts) else int(ts[-1]+2*HOUR)))
    return out
def aggregate(bars, tf):
    span, offset = TFS[tf]
    ts=bars[:,0].astype(np.int64);keys=(ts-offset)//span
    cuts=np.r_[0,np.flatnonzero(np.diff(keys))+1,len(ts)];result=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        if a==b:continue
        start=int(keys[a]*span+offset)
        if b-a==span//HOUR and ts[a]==start and ts[b-1]+HOUR==start+span:
            result.append([start,bars[a,1],bars[a:b,2].max(),bars[a:b,3].min(),bars[b-1,4]])
    return np.array(result,float).reshape(-1,5)

def raw_state(bars, profile):
    w=aggregate(bars,profile['tf'])
    state=st_direction(w[:,2],w[:,3],w[:,4],profile['length'],profile['mult'])
    times=w[:,0].astype(np.int64)+TFS[profile['tf']][0]+HOUR
    ix=np.searchsorted(times,bars[:,0].astype(np.int64),side='right')-1
    return np.where(ix>=0,state[np.maximum(ix,0)],False) if len(w) else np.zeros(len(bars),bool)

def admitted(raw, qualify, eligible):
    available=qualify&eligible
    out=np.zeros(len(raw),bool);active=False
    for i in range(len(raw)):
        if not raw[i]:active=False
        elif not active and available[i]:active=True
        out[i]=active
    return out

def qualify_hours(ts, days, values):
    known=(ts-HOUR)//DAY-1
    ix=np.searchsorted(days,known)
    valid=(ix<len(days))
    result=np.zeros(len(ts),bool)
    result[valid]=values[ix[valid]]&(days[ix[valid]]==known[valid])
    return result
