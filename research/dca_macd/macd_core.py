import json,math,time,pathlib
import numpy as np
import pandas as pd
if __package__:
 from . import wash21
else:
 import wash21
G=vars(wash21)
DAY=86400000
END=int(pd.Timestamp("2026-10-22",tz="UTC").timestamp()*1000)
def stamp(s): return int(pd.Timestamp(s,tz='UTC').timestamp()*1000)
def jprint(x): print(json.dumps(x,ensure_ascii=False,allow_nan=False,default=lambda z:z.item() if isinstance(z,np.generic) else str(z)),flush=True)

def load_bars(sym,start='2021-01-01',end='2026-10-22'):
 a=np.load(data_root/f'dca_indicator_study/5m/{sym}USDT_5m.npy',mmap_mode='r')
 a=np.asarray(a[(a[:,0]>=stamp(start))&(a[:,0]<stamp(end)),:5])
 return a

def gate_arrays(a,kind,stable=21,fail_on='close'):
 dt=D.index.as_unit('ms').asi8+1
 idx=np.searchsorted(dt,a[:,0].astype('int64'),side='right')-1
 ok=idx>=0; ix=np.maximum(idx,0)
 wk=W.index.as_unit('ms').asi8+1
 wi=np.searchsorted(wk,a[:,0].astype('int64'),side='right')-1
 up=(T.trend.to_numpy()[np.maximum(wi,0)]>0)&(wi>=0)
 if kind=='always':return np.ones(len(a),dtype=bool),np.zeros(len(a),dtype=bool)
 if kind=='weekly':return up,~up
 if kind=='early':
  ea,ef,_=make_early_gate(stable,fail_on)
  entry=ea[ix]&ok
  exits=ef[ix]&ok
  return entry,exits
 events=G['evaluate_gate'](D,W)['events']
 entry=np.zeros(len(a),dtype=bool)
 for e in events:
  ts=stamp(e['effective_at'])
  entry[a[:,0]>=ts]=(e['type']=='enable')
 return entry,(~entry if kind=='wash_exit' else np.zeros(len(a),dtype=bool))

def sim_bar_open(a,entry,exits,path=0,step=.75,gx=1.12,amt=1.3,tp=2.,n=13,base=10.,fee=.00018,cooldown_ms=60000):
 ts=a[:,0].astype('int64');bars=a[:,1:5]
 orders=base*amt**np.arange(n);fracs=1-np.cumsum(step/100*gx**np.arange(n))
 cash=1000.;q=0.;cost=0.;bp=0.;nb=0;opened=-1.;open_idx=-1;ready=-1.;first_idx=None
 fees=0.;mincash=1000.;maxcost=0.;skips=0
 buys=0;tps=0;stops=0;end_exits=0;trades=[];negative=0.;positive=0.
 eq=np.empty((len(a),4)); used=np.empty(len(a))
 def close_out(price,tm,reason):
  nonlocal cash,q,cost,bp,nb,opened,open_idx,ready,fees,tps,stops,end_exits,positive,negative
  sell=q*price;sf=sell*fee;profit=sell-sf-cost-cost*fee
  cash+=sell-sf;fees+=sf
  trades.append((int(opened),int(tm),float(profit),float(cost),nb,reason))
  if profit>=0:positive+=profit
  else:negative+=profit
  if reason=='tp':tps+=1
  elif reason=='end':end_exits+=1
  else:stops+=1
  q=cost=bp=0.;nb=0;opened=-1.;open_idx=-1;ready=tm+cooldown_ms
 for i in range(len(a)):
  t=int(ts[i]);o,h,l,c=map(float,bars[i])
  if exits[i] and q>0:close_out(o,t,'gate')
  if q>0:
   target=cost/q*(1+tp/100)
   if o>=target:close_out(o,t,'tp')
   else:
    while nb<n and o<=bp*fracs[nb]:
     order=float(orders[nb])
     if cash+1e-9<order*(1+fee):skips+=1;break
     cash-=order*(1+fee);fees+=order*fee;q+=order/o;cost+=order;nb+=1;buys+=1
  pts=(o,h,l,c) if path==0 else(o,l,h,c)
  if q==0 and entry[i] and not exits[i] and t>=ready and cash>=base*(1+fee):
   cash-=base*(1+fee);fees+=base*fee;q=base/o;cost=base;bp=o;nb=0;opened=t;open_idx=i;buys+=1
   if first_idx is None:first_idx=i
  eq[i,0]=cash+q*o*(1-fee)
  for sg in range(3):
   p0,p1=pts[sg],pts[sg+1];t0=t+sg*100000;t1=t0+100000
   if False:
    ot=max(float(t0),ready);op=p0+(p1-p0)*((ot-t0)/100000)
    if cash>=base*(1+fee):
     cash-=base*(1+fee);fees+=base*fee;q=base/op;cost=base;bp=op;nb=0;opened=ot;open_idx=i;buys+=1
     p0=op;t0=ot
     if first_idx is None:first_idx=i
   for _ in range(64):
    if q<=0:
     break
     ot=max(float(t0),ready)
     op=p0 if t1==t0 else p0+(p1-p0)*(ot-t0)/(t1-t0)
     if cash<base*(1+fee):break
     cash-=base*(1+fee);fees+=base*fee;q=base/op;cost=base;bp=op;nb=0;opened=ot;open_idx=i;buys+=1;p0=op;t0=ot
     if first_idx is None:first_idx=i
    if p1>=p0:
     target=cost/q*(1+tp/100)
     if target>p1+1e-10:break
     price=max(p0,target)
     tm=t0 if p1==p0 else t0+(t1-t0)*(price-p0)/(p1-p0)
     close_out(price,tm,'tp');p0=price;t0=tm
    else:
     if nb>=n:break
     price=bp*fracs[nb]
     if price<p1-1e-10:break
     price=min(p0,price)
     order=float(orders[nb])
     if cash+1e-9<order*(1+fee):skips+=1;break
     tm=t0+(t1-t0)*(p0-price)/(p0-p1) if p0!=p1 else t0
     cash-=order*(1+fee);fees+=order*fee;q+=order/price;cost+=order;nb+=1;buys+=1;p0=price;t0=tm
    mincash=min(mincash,cash);maxcost=max(maxcost,cost)
    if t0>=t1:break
   eq[i,sg+1]=cash+q*p1*(1-fee)
  used[i]=cost;maxcost=max(maxcost,cost);mincash=min(mincash,cash)
 if q>0:close_out(float(a[-1,4]),int(a[-1,0]+300000),'end')
 flat=eq.ravel();peak=np.maximum.accumulate(np.r_[1000.,flat]);dd=float(np.max(1-np.r_[1000.,flat]/peak))
 holds=np.array([(t[1]-t[0])/DAY for t in trades])
 assert mincash>=-1e-7
 assert abs(sum(t[2]for t in trades)-(cash-1000))<1e-6
 return dict(eq=eq,used=used,trades=trades,cash=cash,ret=(cash/1000-1)*100,dd=dd*100,buys=buys,tp=tps,stops=stops,end_exits=end_exits,max_hold=float(holds.max())if len(holds)else 0.,median_hold=float(np.median(holds))if len(holds)else 0.,p90_hold=float(np.quantile(holds,.9))if len(holds)else 0.,hold_gt90=int(np.count_nonzero(holds>90)),full_pct=float(np.mean(used>=900)*100),avg_used=float(np.mean(used)),max_used=maxcost,mincash=mincash,fees=fees,skips=skips,first=(None if first_idx is None else str(pd.to_datetime(ts[first_idx],unit='ms',utc=True))))

def aggregate(a,rs):
 total=sum(r['eq']for r in rs)
 flat=np.r_[1000*len(rs),total.ravel()]
 dd=float(np.max(1-flat/np.maximum.accumulate(flat))*100)
 dates=a[:,0].astype('int64')//DAY
 ix=np.r_[np.flatnonzero(np.diff(dates)!=0),len(a)-1]
 deq=total[ix,-1];deq=np.r_[1000*len(rs),deq]
 ret=deq[1:]/deq[:-1]-1
 sharpe=float(ret.mean()/ret.std()*np.sqrt(365))if ret.std()>0 else 0.
 years={}
 for year in range(2021,2027):
  mask=(a[:,0]>=stamp(f'{year}-01-01'))&(a[:,0]<stamp(f'{year+1}-01-01'))
  ids=np.flatnonzero(mask)
  if len(ids):
   b=ids[0];e=ids[-1];start_eq=1000*len(rs)if b==0 else total[b-1,-1]
   years[str(year)]=round((total[e,-1]-start_eq)/(1000*len(rs))*100,4)
 return dict(ret=round((sum(r['cash']for r in rs)/(1000*len(rs))-1)*100,4),dd=round(dd,4),sharpe=round(sharpe,4),max_hold=round(max(r['max_hold']for r in rs),2),buys=sum(r['buys']for r in rs),tp=sum(r['tp']for r in rs),stops=sum(r['stops']for r in rs),avg_used=round(float(np.mean(sum(r['used']for r in rs)))/(1000*len(rs))*100,4),full_pct=round(float(np.mean(np.concatenate([r['used']>=900 for r in rs])))*100,4),fees=round(sum(r['fees']for r in rs),4),years=years,symbols=[{k:r[k]for k in ('ret','dd','max_hold','hold_gt90','full_pct','max_used','mincash','stops')}for r in rs])


from functools import lru_cache
@lru_cache(None)
def own_daily(sym):
 # Only reuse the OHLC loader. No ST calculation or ST signal.
 return G['frames_from_npy'](data_root/f'dca_indicator_study/5m/{sym}USDT_5m.npy',END)[0]
def project_closed(a,frame,values):
 ix=np.searchsorted(frame.index.as_unit('ms').asi8+1,a[:,0].astype('int64'),side='right')-1
 vals=np.asarray(values,dtype=bool)
 return (ix>=0)&vals[np.maximum(ix,0)]


import sys



@lru_cache(None)
def own_frames(sym):
 return G['frames_from_npy'](data_root/f'dca_indicator_study/5m/{sym}USDT_5m.npy',END)[:2]
def ema(s,n):
 return s.ewm(span=n,adjust=False,ignore_na=True).mean()
def macd_indicators(f):
 c=f.close;m=ema(c,12)-ema(c,26);sig=ema(m,9);hist=m-sig
 em13=ema(c,13);eh=m-m.rolling(9).mean()
 red=(em13<em13.shift())&(eh<eh.shift())
 green=(em13>em13.shift())&(eh>eh.shift())
 high=G['rma'](f.high,34);low=G['rma'](f.low,34)
 mid=2*ema((f.high+f.low+c)/3,34)-ema(ema((f.high+f.low+c)/3,34),34)
 valid=high.notna()&low.notna()
 md=pd.Series(np.where(mid>high,mid-high,np.where(mid<low,mid-low,0.)),index=f.index).where(valid)
 sb=md.rolling(9).mean()
 return pd.DataFrame({'macd':m,'signal':sig,'hist':hist,'hist_gate':hist>0,
   'impulse_md':md,'impulse_signal':sb,'impulse_gate':(md>sb)&sb.notna(),
   'elder_hist':eh,'elder_ema':em13,'elder_red':red,'elder_green':green,
   'elder_gate':(~red)&eh.notna()},index=f.index)
@lru_cache(None)
def mf(sym,tf):
 return macd_indicators(own_frames(sym)[0 if tf=='D'else 1])

def left_signals(f):
 v=macd_indicators(f);h=v['hist'];m=v['macd']
 raw=(h<0)&(h>h.shift())
 for k in range(1,6):raw &= h.shift(k)<0
 for k in range(1,5):raw &= h.shift(k)<h.shift(k+1)
 # Reproduce prior confirmed pivot / current right=0 candidate from LonesomeTheBlue.
 low=f.low.to_numpy();close=f.close.to_numpy();hist=h.to_numpy()
 div=np.zeros(len(f),dtype=bool);anchor=np.full(len(f),-1,dtype=int);last=-1;p=5
 for i in range(len(f)):
  c=i-p
  if c>=p and low[c]<=low[c-p:c].min() and low[c]<low[c+1:i+1].min():last=c
  anchor[i]=last
  if i<p or last<0 or low[i]>low[i-p:i].min()or low[i]>=low[last]or hist[i]<=hist[last]:continue
  distance=i-last
  line=low[i]-(low[i]-low[last])*np.arange(1,distance)/distance
  div[i]=bool(np.all(close[i-np.arange(1,distance)]>=line))
 v['raw_early5']=raw
 v['raw_divergence']=div
 # Added to the original indicators: both DIF and histogram must still be negative.
 v['early5']=raw&(m<0)
 v['divergence']=div&(m<0)&(h<0)
 v['anchor_i']=anchor
 return v


@lru_cache(None)
def frame(sym,tf):
 d,w=own_frames(sym)
 if tf=='D':return d
 if tf=='W':return w
 x=d.copy();x.index=x.index+pd.Timedelta(milliseconds=1)-pd.Timedelta(days=1)
 out=x.resample('72h',origin=pd.Timestamp('1970-01-01',tz='UTC'),closed='left',label='left').agg({'open':'first','high':'max','low':'min','close':'last'})
 count=x.close.resample('72h',origin=pd.Timestamp('1970-01-01',tz='UTC'),closed='left',label='left').count()
 out=out[count==3].dropna()
 out.index=out.index+pd.Timedelta(days=3)-pd.Timedelta(milliseconds=1)
 return out[out.index<pd.Timestamp(END,unit='ms',tz='UTC')]
def pine_round(s):
 return np.floor(s*100+.5)/100
def stoch(s,n):
 lo=s.rolling(n,min_periods=1).min();span=s.rolling(n,min_periods=1).max()-lo
 return (100*(s-lo)/span.replace(0,np.nan)).ffill().fillna(0)
def variant_indicators(f):
 v=left_signals(f);c=f.close;h=v['hist'];m=v['macd']
 v['basic']=(h<0)&(h>h.shift())&(m<0)
 # Public Impulse MACD's negative md recovers before a zero or signal cross.
 md=v['impulse_md'];v['impulse']=(md<0)&(md>md.shift())&(m<0)
 fast=ema(c,12);slow=ema(c,26)
 leader=(fast+ema(c-fast,12))-(slow+ema(c-slow,26))
 v['leader_line']=leader
 v['leader']=(leader<0)&(leader>leader.shift())&(m<0)
 atr=G['rma'](G['true_range'](f),26)
 val=pine_round(m/atr*100);unrounded_signal=ema(val,9)
 vh=pine_round(val-unrounded_signal)
 v['V']=val;v['VH']=vh;v['Vsignal']=pine_round(unrounded_signal)
 v['v_extreme']=(vh<-40)&(val<0)
 v['v_contract']=(vh<-40)&(vh>vh.shift())&(val<0)
 stcm=ema(c,23)-ema(c,50)
 k=stoch(stcm,10);dd=ema(k,3);kd=stoch(dd,10);stc=ema(kd,3).clip(0,100)
 v['stc_line']=stc
 v['stc']=(stc>0)&(stc<=25)&(stc>stc.shift())&(stcm<0)
 return v
@lru_cache(None)
def vf(sym,tf):
 return variant_indicators(frame(sym,tf))
# All configurations fixed before inspecting this round's performance.
CONFIGS=[
 ('basic_D_Dm','D','basic','D','m','普通MACD 日线 / 日线环境'),
 ('basic_D_3Dm','D','basic','3D','m','普通MACD 日线 / 3日环境'),
 ('basic_D_Wm','D','basic','W','m','普通MACD 日线 / 周线深跌环境'),
 ('basic_3D_Wm','3D','basic','W','m','普通MACD 3日 / 周线深跌环境'),
 ('basic_D_Wh','D','basic','W','h','普通MACD 日线 / 周线回调环境'),
 ('early5_D_Wm','D','early5','W','m','5负柱收缩 日线 / 周线深跌环境'),
 ('early5_D_Wh','D','early5','W','h','5负柱收缩 日线 / 周线回调环境'),
 ('div_D_Wm','D','divergence','W','m','底背离 日线 / 周线深跌环境'),
 ('div_D_Wh','D','divergence','W','h','底背离 日线 / 周线回调环境'),
 ('impulse_D_Wm','D','impulse','W','m','Impulse 日线 / 周线深跌环境'),
 ('impulse_3D_Wm','3D','impulse','W','m','Impulse 3日 / 周线深跌环境'),
 ('impulse_D_Wh','D','impulse','W','h','Impulse 日线 / 周线回调环境'),
 ('leader_D_Wm','D','leader','W','m','Leader 日线 / 周线深跌环境'),
 ('vext_D_Wm','D','v_extreme','W','m','MACD-V 直接提前 日线 / 周线深跌环境'),
 ('vcon_D_Wm','D','v_contract','W','m','MACD-V 收敛 日线 / 周线深跌环境'),
 ('vcon_D_Wh','D','v_contract','W','h','MACD-V 收敛 日线 / 周线回调环境'),
 ('vcon_3D_Wm','3D','v_contract','W','m','MACD-V 收敛 3日 / 周线深跌环境'),
 ('stc_D_Wm','D','stc','W','m','STC低位上升 日线 / 周线深跌环境'),
 ('stc_D_Wh','D','stc','W','h','STC低位上升 日线 / 周线回调环境'),
 ('stc_3D_Wm','3D','stc','W','m','STC低位上升 3日 / 周线深跌环境')]
assert len(CONFIGS)==20
LOOKUP={r[0]:r for r in CONFIGS}
VARIANTS=[('wash_exit','native_exit'),('always','entry_only')]+[(r[0],'btc_flip_exit')for r in CONFIGS]
LABELS={'wash_exit_native_exit':'基线 BTC提前门 wash21','always_entry_only':'对照 无门持仓等TP'}
LABELS.update({r[0]+'_btc_flip_exit':r[-1]for r in CONFIGS})
@lru_cache(None)
def phase(sym,key):
 _,lowtf,kind,htf,macro,_=LOOKUP[key]
 lo=frame(sym,lowtf);lv=vf(sym,lowtf)
 hi=frame(sym,htf);hv=vf(sym,htf)
 lowtm=lo.index.as_unit('ms').asi8+1;hitm=hi.index.as_unit('ms').asi8+1
 cond=lv[kind].to_numpy(dtype=bool)
 # An onset requires a fresh warning; persistent warning conditions do not create daily restart events.
 onset=cond&~np.r_[False,cond[:-1]]
 macro_ok=(hv['macd']<0 if macro=='m'else hv['hist']<0).to_numpy(dtype=bool)
 down=(T.trend<0)&(T.trend.shift()>0);downs=W.index[down].as_unit('ms').asi8+1
 events=[]
 for tm,ok in zip(hitm,macro_ok):events.append((int(tm),0,bool(ok)))
 for tm,tr in zip(lowtm,onset):events.append((int(tm),1,bool(tr)))
 for tm in downs:events.append((int(tm),2,False))
 events.sort()
 active=False;env=False;times=[];states=[];audit=[]
 for tm,typ,flag in events:
  old=active
  if typ==0:
   env=flag
   if not env:active=False
  elif typ==1:
   if flag and env:active=True
  else:active=False
  times.append(tm);states.append(active)
  if active!=old:audit.append((tm,'enable'if active else'disable','macro'if typ==0 else'warning'if typ==1 else'btc_risk_flip'))
 return np.asarray(times,dtype='int64'),np.asarray(states,dtype=bool),downs,audit
def tv_gate(a,sym,kind,mode):
 if kind in ('wash_exit','always'):return gate_arrays(a,kind)
 times,states,downs,_=phase(sym,kind)
 ix=np.searchsorted(times,a[:,0].astype('int64'),side='right')-1
 en=(ix>=0)&states[np.maximum(ix,0)]
 ex=np.isin(a[:,0].astype('int64'),downs)
 return en,ex
RUNS={}
def verify_multitf():
 for sym in ('BNB','SOL'):
  for tf in ('D','3D','W'):
   f=frame(sym,tf);full=vf(sym,tf)
   counts=(400,1100,1800)if tf=='D'else(100,250,500)if tf=='3D'else(80,160,250)
   for n in counts:
    cut=variant_indicators(f.iloc[:n])
    for col in full:np.testing.assert_allclose(cut[col].to_numpy(dtype=float),full[col].iloc[:n].to_numpy(dtype=float),equal_nan=True,atol=1e-10)
   tm=f.index.as_unit('ms').asi8
   probe=np.zeros((2*len(f),5));probe[::2,0]=tm;probe[1::2,0]=tm+1
   alt=np.arange(len(f))%2==0;seen=project_closed(probe,f,alt)
   np.testing.assert_array_equal(seen[1::2],alt);np.testing.assert_array_equal(seen[2::2],alt[:-1])
   assert not seen[0]
  # 3-day alignment and length are defined by UTC epoch, independent of input start.
  three=frame(sym,'3D');starts=three.index.as_unit('ms').asi8+1-3*DAY
  assert np.all(starts%(3*DAY)==0)
  # No lookahead in state machine: disabling at a higher timeframe close happens only from that close forward.
  for cfg in CONFIGS:
   tm,states,_,audit=phase(sym,cfg[0])
   assert np.all(np.diff(tm)>=0)
   for t,act,why in audit:
    if act=='enable':assert why=='warning'
 jprint({'verification':'PASS: prefix invariance across D/3D/W and all variant outputs; confirmed-bar mapping; fixed 3D UTC origin; causal phase transitions; every run cash/PnL assertions',
  'configs':len(CONFIGS),'DCA':[.75,1.12,1.3,2,13]})

# Fixed before measuring this new ST+MACD round. All signal logic is native closed D/W.
PRESETS={'p6':(6,2.5),'p10':(10,3.)}
def strict_st(f,length,mult):
 atr=G['rma'](G['true_range'](f),length).to_numpy()
 out=np.full((len(f),4),np.nan);prev_u=prev_l=prev_line=np.nan
 h=f.high.to_numpy();l=f.low.to_numpy();cl=f.close.to_numpy()
 for i in range(len(f)):
  if not np.isfinite(atr[i]):continue
  bu=(h[i]+l[i])/2+mult*atr[i];bl=(h[i]+l[i])/2-mult*atr[i]
  if not np.isfinite(prev_u):u,lo,tr=bu,bl,-1
  else:
   u=bu if bu<prev_u or cl[i-1]>prev_u else prev_u
   lo=bl if bl>prev_l or cl[i-1]<prev_l else prev_l
   tr=(1 if cl[i]>u else-1)if prev_line==prev_u else(-1 if cl[i]<lo else 1)
  line=lo if tr>0 else u
  out[i]=tr,line,lo,u;prev_u,prev_l,prev_line=u,lo,line
 st=pd.DataFrame(out,index=f.index,columns=['trend','line','lower','upper']);st['atr']=atr
 return st
@lru_cache(None)
def stf(sym,tf,preset):
 return strict_st(frame(sym,tf),*PRESETS[preset])
def project_num(f,values,times):
 ix=np.searchsorted(f.index.as_unit('ms').asi8+1,times,side='right')-1
 val=np.asarray(values);result=np.full(len(times),np.nan)
 valid=ix>=0;result[valid]=val[ix[valid]];return result
def onset(v):
 v=np.asarray(v,dtype=bool);return v&~np.r_[False,v[:-1]]
def masked_stall(st,direction,n=4,atr_fraction=.5):
 series=st['upper'if direction<0 else'lower'].where(st.trend==direction)
 return((series.rolling(n,min_periods=2).max()-series.rolling(n,min_periods=2).min())<=atr_fraction*st.atr)&(st.trend==direction)
FAMILIES=[
 ('pull','大趋势向上＋自身日ST下跌＋日MACD收缩'),
 ('flat','周ST下跌停滞＋日MACD收缩提前'),
 ('extreme','周ST下跌＋日MACD-V极端预警提前'),
 ('early','周ST下跌＋日MACD收缩提前'),
 ('hybrid','提前渠道＋趋势内回调渠道'),
 ('tail','双渠道＋上涨停滞/动能减弱暂停重开'),
 ('wash','保留ST洗盘结构、MACD替代21天等待')]
EXPERIMENTS=[]
for family,label in FAMILIES:
 for source_ in ('BTC','SELF'):
  for preset in PRESETS:
   key=f'{family}_{source_}_{preset}'
   EXPERIMENTS.append((key,family,source_,preset,f'{label} / {source_} / ST{PRESETS[preset]}'))
CONTROL=[('baseline','baseline','BTC','p6','原BTC提前门wash21'),
 ('macd_only','macd_only','SELF','p6','仅日MACD左侧运行窗口')]
for source_ in ('BTC','SELF'):
 for preset in PRESETS:
  CONTROL.append((f'st_only_{source_}_{preset}','st_only',source_,preset,f'仅周ST方向门 / {source_} / ST{PRESETS[preset]}'))
JOBS=CONTROL+EXPERIMENTS
assert len(EXPERIMENTS)==28 and len(JOBS)==34
JL={x[0]:x for x in JOBS}

@lru_cache(None)
def features(sym,channel,preset):
 ctx=sym if channel=='SELF'else'BTC'
 d=frame(sym,'D');times=d.index.as_unit('ms').asi8+1
 sd=frame(ctx,'D');sw=frame(ctx,'W');ss=stf(ctx,'W',preset);sm=vf(ctx,'W');sv=vf(ctx,'D')
 own=vf(sym,'D');ownst=stf(sym,'D',preset)
 z=pd.DataFrame(index=d.index)
 z['macro_up']=project_num(sw,ss.trend>0,times)>0
 z['macro_down']=project_num(sw,ss.trend<0,times)>0
 z['macro_flip_down']=onset(z.macro_down)
 z['flat_down']=project_num(sw,masked_stall(ss,-1),times)>0
 z['tail_weak']=project_num(sw,masked_stall(ss,1)&(sm['hist']>0)&(sm['hist']<sm['hist'].shift()),times)>0
 z['week_right']=project_num(sw,(ss.trend>0)&(sm['hist']>0)&(sm['hist']<sm['hist'].shift()),times)>0
 srcwarn=(sv['hist']<0)&(sv['hist']>sv['hist'].shift())
 z['source_warn']=project_num(sd,srcwarn,times)>0
 z['source_extreme']=project_num(sd,sv['v_extreme'],times)>0
 z['own_warn']=(own['hist']<0)&(own['hist']>own['hist'].shift())
 z['own_down']=ownst.trend<0
 z['day_right']=(ownst.trend>0)&(own['hist']>0)&(own['hist']<own['hist'].shift())
 z['macd_right']=(own['hist']>0)&(own['hist']<own['hist'].shift())
 z['early_flag']=z.macro_down&z.source_warn
 z['flat_flag']=z.early_flag&z.flat_down
 z['extreme_flag']=z.macro_down&z.source_extreme
 z['pull_flag']=z.macro_up&z.own_down&z.own_warn
 z['tail_early_flag']=z.early_flag&~z.tail_weak
 z['tail_pull_flag']=z.pull_flag&~z.tail_weak
 z['risk']=np.isin(times,COMMON_DOWNS)
 for k in('early','flat','extreme','pull','tail_early','tail_pull'):
  z[k+'_on']=onset(z[k+'_flag'])
 z['macd_on']=onset(z.own_warn)
 z['st_up_on']=onset(z.macro_up)
 # Explanatory features retain a rolling history across short ST flips. Never used to optimize thresholds.
 z['context_st_distance_atr']=project_num(sw,(sw.close-ss.line)/ss.atr,times)
 z['context_week_hist']=project_num(sw,sm['hist'],times)
 z['own_day_hist']=own['hist']
 z['own_day_ST_distance_atr']=(d.close-ownst.line)/ownst.atr
 z['context_setup']=False;z['wash_low']=False
 source_di=np.searchsorted(sd.index.as_unit('ms').asi8+1,times,side='right')-1
 source_wi=np.searchsorted(sw.index.as_unit('ms').asi8+1,times,side='right')-1
 da=G['rma'](G['true_range'](sd),14)
 ready=[]
 for di,wi in zip(source_di,source_wi):
  ready.append(G['_setup_conditions'](sd,ss,da,int(di),int(wi))['setup_ready']if di>=0 and wi>=0 else False)
 z['context_setup']=ready
 for i,di in enumerate(source_di):
  if di>=28:z.iloc[i,z.columns.get_loc('wash_low')]=float(sd.low.iloc[di])<=float(sd.low.iloc[di-28:di].min())
 z['source_daily_low']=project_num(sd,sd.low,times)
 return z
def state_trace(sym,key,stable21=False):
 _,family,channel,preset,_=JL[key]
 z=features(sym,channel,preset);times=z.index.as_unit('ms').asi8+1
 active_early=active_pull=active_wash=trend_active=False
 armed=waiting=False;wash_low=np.nan;stable=0;en=[];audit=[]
 for i,r in enumerate(z.itertuples()):
  old=(active_early or active_pull or active_wash or trend_active)
  risk=bool(r.risk);reason=''
  if risk:
   active_early=active_pull=active_wash=trend_active=False
   armed=waiting=False;wash_low=np.nan;stable=0;reason='BTC_weekly_risk_reset'
  else:
   if r.macro_flip_down and family!='macd_only':
    active_early=active_wash=trend_active=False
    armed=waiting=False;wash_low=np.nan;stable=0;reason='context_new_down_phase'
   if r.week_right and family not in('wash','st_only','macd_only'):
    active_early=False;reason='context_up_positive_hist_contracts'
   if not r.macro_up:active_pull=False;trend_active=False
   if r.day_right:active_pull=False;reason='own_daily_right_boundary'
   if family=='macd_only':
    if r.macd_right:active_early=False;reason='daily_positive_hist_contracts'
    if r.macd_on:active_early=True;reason='daily_negative_hist_contracts'
   elif family=='st_only':
    if r.st_up_on:trend_active=True;reason='context_ST_turns_up'
   elif family=='wash':
    if not r.macro_down:
     armed=waiting=False;wash_low=np.nan;stable=0
    elif not active_wash:
     if r.context_setup:armed=True
     if waiting:
      if r.source_daily_low<wash_low:wash_low=r.source_daily_low;stable=0
      else:stable+=1
     elif armed and r.wash_low:
      waiting=True;wash_low=r.source_daily_low;stable=0
     if waiting and(stable>=21 if stable21 else r.source_warn):
      active_wash=True;armed=waiting=False;reason='wash_stable21'if stable21 else'wash_then_negative_hist_contracts'
   else:
    if family in('early','hybrid')and r.early_on:active_early=True;reason='early_warning'
    if family=='flat'and r.flat_on:active_early=True;reason='flat_then_early_warning'
    if family=='extreme'and r.extreme_on:active_early=True;reason='MACDV_extreme_early_warning'
    if family in('pull','hybrid')and r.pull_on:active_pull=True;reason='local_left_pullback'
    if family=='tail':
     if r.tail_weak:
      active_early=active_pull=False;reason='weekly_ST_stall_and_MACD_fades'
     else:
      if r.tail_early_on:active_early=True;reason='early_warning_after_tail_veto'
      if r.tail_pull_on:active_pull=True;reason='local_pullback_after_tail_veto'
  if family=='macd_only':allowed=active_early
  elif family=='st_only':allowed=trend_active
  elif family=='wash':allowed=active_wash
  elif family=='pull':allowed=active_pull
  elif family in('hybrid','tail'):allowed=active_early or active_pull
  else:allowed=active_early
  en.append(allowed)
  if allowed!=old:audit.append({'effective_ms':int(times[i]),'event':'enable'if allowed else'disable','reason':reason})
 return times,np.asarray(en,dtype=bool),audit
@lru_cache(None)
def jt(sym,key):
 return state_trace(sym,key)
def joint_gate(a,sym,key):
 if key=='baseline':return gate_arrays(a,'wash_exit')
 tm,en,_=jt(sym,key);ix=np.searchsorted(tm,a[:,0].astype('int64'),side='right')-1
 return(ix>=0)&en[np.maximum(ix,0)],np.isin(a[:,0].astype('int64'),COMMON_DOWNS)
def verify_joint():
 discrepancies=[];prefix_checks=0
 for sym in('BTC','BNB','SOL'):
  for tf in('D','W'):
   f=frame(sym,tf)
   for preset in PRESETS:
    full=stf(sym,tf,preset)
    for n in((200,700,1400)if tf=='D'else(80,160,250)):
     cut=strict_st(f.iloc[:n],*PRESETS[preset])
     np.testing.assert_allclose(cut.to_numpy(),full.iloc[:n].to_numpy(),atol=1e-10,equal_nan=True);prefix_checks+=1
    old=G['supertrend_trace'](f,*PRESETS[preset])
    valid=full.trend.notna()&old.trend.notna();count=int(((full.trend!=old.trend)&valid).sum())
    discrepancies.append({'symbol':sym,'tf':tf,'preset':preset,'direction_mismatches_vs_old':count})
 for sym in('BNB','SOL'):
  a=load_bars(sym,'2021-01-01')
  # The BTC p6 wash/stable21 reconstruction must reproduce the current baseline before replacing its wait rule.
  tm,en,_=state_trace(sym,'wash_BTC_p6',stable21=True)
  ix=np.searchsorted(tm,a[:,0].astype('int64'),side='right')-1;custom=(ix>=0)&en[np.maximum(ix,0)]
  base=gate_arrays(a,'wash_exit')[0]
  assert np.array_equal(custom,base),f'baseline wash21 reconstruction mismatch {sym}: {np.count_nonzero(custom!=base)}'
  for key,_,_,_,_ in JOBS:
   if key=='baseline':continue
   tm,en,audit=jt(sym,key)
   assert not en[np.isin(tm,COMMON_DOWNS)].any()
   # Events are causal: permission is not visible before the close that creates it.
   for e in audit:
    probe=np.zeros((2,5));probe[:,0]=[e['effective_ms']-1,e['effective_ms']]
    states=joint_gate(probe,sym,key)[0]
    assert states[1]==(e['event']=='enable')
 jprint({'verification':'PASS: strict ST prefix invariance, exact baseline reconstruction, risk-priority reset and confirmed-close event timing','ST_prefix_checks':prefix_checks,'ST_direction_differences':discrepancies})
 return {'strict_ST_direction_comparison':discrepancies,'prefix_checks':prefix_checks,'baseline_reconstruction':'exact match after 2021-01-01 for both BNB/SOL','risk_reset':'PASS','signal_availability':'PASS'}

def initialize(folder):
 global data_root,D,W,T,COMMON_DOWNS
 data_root=pathlib.Path(folder)
 D,W,_=G['frames_from_npy'](data_root/'dca_indicator_study/5m/BTCUSDT_5m.npy',END)
 T=G['supertrend_trace'](W)
 COMMON_DOWNS=W.index[(T.trend<0)&(T.trend.shift()>0)].as_unit('ms').asi8+1
