"""Independent Nautilus matching/accounting; explicit OHLC-derived quote events.

No original DCA engine is used here. The original replay supplies weekly decisions only.
"""
from __future__ import annotations

import sys
import traceback
from functools import wraps
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pathlib import Path

import numpy as np
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggerConfig, StrategyConfig
from nautilus_trader.config import PortfolioConfig
from nautilus_trader.config import ExecutionEngineConfig
from nautilus_trader.common import LogLevel
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import (AccountType, CryptoPerpetual, Currency, CurrencyType,
    FundingRateUpdate, InstrumentId, MarkPriceUpdate, Money, OmsType, OrderSide,
    Price, Quantity, QuoteTick, Symbol, TimeInForce, Venue)
from nautilus_trader.trading import Strategy


HOUR = 3_600_000
NS_HOUR = HOUR*1_000_000
FEE = Decimal('0.0005')
INITIAL = 10000.0
USDT = Currency.from_str('USDT')
BINANCE = Venue('BINANCE')
Q_STEP = Decimal('0.00000001')


def number(value):
    return Decimal(str(value).replace('_',''))


def checked(method):
    @wraps(method)
    def call(self,*args):
        try:return method(self,*args)
        except Exception:
            self.errors.append(traceback.format_exc())
            raise
    return call


def instrument(symbol,price_step='0.000000000001',quantity_step='0.00000001',leverage=1,maintenance_rate=.005):
    # Explicit research metadata, not a historical Binance instrument master.
    # Uniform fine increments avoid applying today's changed filters to old prices.
    base=symbol[:-4]
    return CryptoPerpetual(
        instrument_id=InstrumentId.from_str(symbol+'-PERP.BINANCE'),raw_symbol=Symbol(symbol),
        base_currency=Currency(base,8,0,base,CurrencyType.CRYPTO),quote_currency=USDT,
        settlement_currency=USDT,is_inverse=False,
        price_precision=-Decimal(price_step).as_tuple().exponent,size_precision=-Decimal(quantity_step).as_tuple().exponent,
        price_increment=Price.from_str(price_step),size_increment=Quantity.from_str(quantity_step),
        # Native LeveragedMarginModel divides BOTH rates by leverage. Scale
        # the maintenance coefficient so the requested rate stays on notional.
        margin_init=Decimal('1'),margin_maint=number(maintenance_rate)*number(leverage),ts_event=0,ts_init=0)




class ZigzagDca(Strategy):
    def setup(self,inst,windows,resize=True,step=.008,tp=.012,amount=1.2,levels=14,tp_mode='price',leverage=1,compound=False,step_scale=1):
        self.inst=inst;self.windows=windows;self.resize=resize
        self.step=number(step);self.target=number(tp);self.amount=number(amount);self.levels=levels
        self.step_scale=number(step_scale)
        self.quantity_step=number(inst.size_increment);self.price_step=number(inst.price_increment)
        self.budget_capital=INITIAL
        self.leverage=number(leverage);self.compound=compound;self.cycle_capital=[]
        self.tp_mode=tp_mode;self.notional=Decimal(0)
        self.errors=[]
        self.openings={w['open_ms']*1_000_000+1000:w for w in windows}
        self.ends={w['close_ms']*1_000_000 for w in windows}
        self.current=None;self.restart=False;self.pending_base=False
        self.force_flat=False;self.deferred_window=None
        self.exhausted=False
        self.restart_queued=False
        self.qty=Decimal(0);self.cost=Decimal(0);self.p0=Decimal(0);self.base=Decimal(0)
        self.tp=None;self.seeded=False;self.order_info={}
        self.last_price=Decimal(0);self.last_time=0;self.fills=[];self.rejections=[]
        self.last_outer_window=None
        self.equity=[];self.tp_count=0;self.add_fills=0;self.base_fills=0;self.window_capital=[]

    @checked
    def on_start(self):self.subscribe_quotes(self.inst.id)

    def cash(self):return float(self.cache.account_for_venue(BINANCE).balance_total(USDT))

    def mark(self):
        if not self.last_price:return
        equity=float(self.portfolio.equity(venue=BINANCE).get(USDT,Money.from_str('0 USDT')))
        if self.equity and self.equity[-1][0]==self.last_time:self.equity[-1]=(self.last_time,equity)
        else:self.equity.append((self.last_time,equity))

    @checked
    def on_quote(self,quote):
        self.last_price=number(quote.bid_price);self.last_time=int(quote.ts_init)
        # Cancel known future-window orders at the preceding close, then execute
        # the market exit on the newly observed next-hour opening quote.
        if (self.last_time+1)%NS_HOUR==0 and self.last_time+1 in self.ends:
            self.restart=False;self.cancel_all_orders(self.inst.id)
        if self.current is not None and self.last_time>=self.current['close_ms']*1_000_000+1000:
            self.current=None;self.restart=False;self.cancel_all_orders(self.inst.id)
            self.force_flat=True
            self.close_all_positions(self.inst.id,tags=['WINDOW_EXIT'])
            self.mark();return
        opening=self.openings.get(self.last_time)
        if self.force_flat:
            if self.qty>0:
                if opening is not None and not self.exhausted:self.deferred_window=opening
                # Large compounded positions may exceed one quote's liquidity.
                # Preserve a still-working native close order. Re-submit only
                # when the previous IOC/partial liquidation has no live exit.
                live=self.cache.orders_open(instrument_id=self.inst.id)
                exiting=any(o.side==OrderSide.SELL and 'WINDOW_EXIT' in (o.tags or []) for o in live)
                if not exiting:
                    exit_order=self.order_factory.market(instrument_id=self.inst.id,order_side=OrderSide.SELL,
                        quantity=Quantity.from_str(str(self.qty)),time_in_force=TimeInForce.IOC,
                        reduce_only=True,tags=['WINDOW_EXIT'])
                    self.send(exit_order,'WINDOW_EXIT')
                self.mark();return
            self.force_flat=False
            if self.deferred_window is not None:
                if self.last_time<self.deferred_window['close_ms']*1_000_000:opening=self.deferred_window
                self.deferred_window=None
        if opening is not None and not self.exhausted and self.cash()>1:
            self.current=opening;self.restart=True
            outer=self.current.get('outer_window',self.current['open_ms'])
            capital=self.cash() if self.resize or self.compound else self.budget_capital
            reset=self.compound or self.last_outer_window!=outer
            if reset or not self.resize:
                self.base=number(capital)/(Decimal(1)+sum(self.amount**i for i in range(self.levels)))
            self.last_outer_window=outer
            self.window_capital.append(dict(**self.current,capital=capital,base=float(self.base),reset_base=reset))
            self.open_base()
        elif self.restart_queued and self.restart:
            self.restart_queued=False;self.open_base()
        self.mark()

    def quantity(self,budget,price):
        q=(budget*self.leverage/(price*(1+FEE*self.leverage))/self.quantity_step).to_integral_value(rounding=ROUND_DOWN)*self.quantity_step
        return Quantity.from_str(str(q)) if q>0 else None

    def send(self,order,kind,budget=None):
        self.order_info[str(order.client_order_id)]=dict(kind=kind,budget=str(budget))
        self.submit_order(order)

    def open_base(self):
        if not self.restart or self.qty or self.pending_base or self.base<=0:return
        account=self.cache.account_for_venue(BINANCE)
        if self.cache.orders_open(instrument_id=self.inst.id) or number(float(account.balance_free(USDT)))<self.base:
            self.restart_queued=True
            return
        quantity=self.quantity(self.base,self.last_price)
        if quantity is None:return
        self.cycle_capital.append(dict(time_ns=self.last_time,wallet=self.cash(),margin_base=float(self.base),leverage=float(self.leverage)))
        self.pending_base=True;self.seeded=False
        order=self.order_factory.market(instrument_id=self.inst.id,order_side=OrderSide.BUY,
                                        quantity=quantity,time_in_force=TimeInForce.IOC,tags=['BASE'])
        self.send(order,'BASE',self.base)

    def seed_ladder(self):
        deviation=Decimal(0)
        for level in range(1,self.levels+1):
            if self.step_scale==1:
                price=self.p0*(1-self.step*level)
            else:
                deviation+=self.step*self.step_scale**(level-1)
                price=self.p0*(1-deviation)
            price=(price/self.price_step).to_integral_value(rounding=ROUND_DOWN)*self.price_step
            budget=self.base*self.amount**(level-1)
            quantity=self.quantity(budget,price)
            if quantity is None:continue
            order=self.order_factory.limit(instrument_id=self.inst.id,order_side=OrderSide.BUY,
                quantity=quantity,price=Price(float(price),self.inst.price_precision),time_in_force=TimeInForce.GTC,tags=[f'ADD:{level}'])
            self.send(order,f'ADD:{level}',budget)

    def update_tp(self):
        if self.tp is not None and not self.tp.is_closed:self.cancel_order(self.tp.client_order_id)
        if self.qty<=0 or not self.restart:return
        price=(self.notional*(1+self.target)/self.qty if self.tp_mode=='price'
               else self.cost*(1+self.target)/(self.qty*(1-FEE)))
        price=(price/self.price_step).to_integral_value(rounding=ROUND_UP)*self.price_step
        self.tp=self.order_factory.limit(instrument_id=self.inst.id,order_side=OrderSide.SELL,
            quantity=Quantity.from_str(str(self.qty)),price=Price(float(price),self.inst.price_precision),
            reduce_only=True,time_in_force=TimeInForce.GTC,tags=['TP'])
        self.send(self.tp,'TP')

    @checked
    def on_order_filled(self,event):
        quote=self.cache.quote(self.inst.id)
        fresh=quote is not None and quote.ts_init>=event.ts_event
        if fresh:
            self.last_price=number(quote.bid_price);self.last_time=int(event.ts_event)
        kind=self.order_info.get(str(event.client_order_id),{}).get('kind','WINDOW_EXIT')
        if str(event.client_order_id).startswith('LIQUIDATION-'):
            kind='LIQUIDATION';self.exhausted=True;self.restart=False;self.current=None;self.force_flat=True
        quantity,price=number(event.last_qty),number(event.last_px)
        commission=number(float(event.commission))
        self.fills.append(dict(time_ns=int(event.ts_event),kind=kind,side=str(event.order_side),
            qty=float(quantity),price=float(price),commission=float(commission),order_id=str(event.client_order_id)))
        if event.order_side==OrderSide.BUY:
            self.qty+=quantity;self.cost+=quantity*price+commission;self.notional+=quantity*price
            if kind=='BASE':
                self.pending_base=False;self.base_fills+=1
                if not self.seeded:
                    self.p0=price;self.seeded=True
                    if self.restart:self.seed_ladder()
            else:self.add_fills+=1
            self.update_tp()
        else:
            before=self.qty;self.qty=max(Decimal(0),self.qty-quantity)
            self.cost=self.cost*self.qty/before if before else Decimal(0)
            self.notional=self.notional*self.qty/before if before else Decimal(0)
            if self.qty==0:
                if kind=='TP':self.tp_count+=1
                self.tp=None;self.pending_base=False;self.seeded=False
                self.cancel_all_orders(self.inst.id)
                if kind=='TP' and self.restart:
                    # Re-enter after native cancels and account updates settle.
                    self.restart_queued=True
            elif kind=='TP':self.update_tp()
        if fresh:self.mark()

    def on_order_rejected(self,event):
        self.rejections.append(dict(time_ns=int(event.ts_event),kind='REJECTED',reason=str(event)))
        if self.order_info.get(str(event.client_order_id),{}).get('kind')=='BASE':self.pending_base=False

    def on_order_denied(self,event):
        self.rejections.append(dict(time_ns=int(event.ts_event),kind='DENIED',reason=str(event)))
        if self.order_info.get(str(event.client_order_id),{}).get('kind')=='BASE':self.pending_base=False

    @checked
    def on_position_event(self,event):
        self.last_time=max(self.last_time,int(event.ts_event));self.mark()


def market_events(inst,bars,funding,marks,path='OHL',actual_funding=False):
    data=[]
    for i,row in enumerate(bars):
        t=int(row[0])*1_000_000
        values=[row[1],row[2],row[3],row[4]] if path=='OHL' else [row[1],row[3],row[2],row[4]]
        times=[t+1000,t+NS_HOUR//3,t+2*NS_HOUR//3,t+NS_HOUR-1]
        points=list(zip(times,values))
        volume=max(float(inst.size_increment),float(row[5])/len(points))
        for when,price in points:
            px=Price(float(price),inst.price_precision);sz=Quantity(volume,inst.size_precision)
            data.append(QuoteTick(inst.id,px,px,sz,sz,when,when))
    mark_lookup={int(r[0]):float(r[1]) for r in marks}
    missing_marks=0
    for timestamp,rate,_interval in funding:
        if timestamp<bars[0,0] or timestamp>=bars[-1,0]+HOUR:continue
        when=int(timestamp)*1_000_000
        hour=int(timestamp)//HOUR*HOUR
        price=mark_lookup.get(hour)
        if price is None:
            missing_marks+=1
            index=int(np.searchsorted(bars[:,0],timestamp,side='right')-1)
            price=float(bars[max(index,0),1])
        data.append(MarkPriceUpdate(inst.id,Price(price,inst.price_precision),when,when))
        data.append(FundingRateUpdate(inst.id,Decimal(str(rate if actual_funding else max(rate,0))),when,when,
                                     interval=None,next_funding_ns=when))
    # Stable order keeps mark before funding at the same timestamp.
    data.sort(key=lambda item:item.ts_init)
    return data,missing_marks


def replay_continuous(engine,strategy,inst,data,crossings=True):
    """Replay a declared continuous OHLC segment at live order crossings.

    Prices jump across hourly boundaries. Inside each declared segment, insert
    the next active TP/add price, let Nautilus fill it, then read updated orders.
    Batch anchors which cannot touch an order; no arbitrary price mesh is used.
    """
    pending=[];record=[];buy_prices=[];tp_price=None;liquidation_price=None
    def append_quote(q):
        if record and record[-1][0]==q.ts_init:record[-1]=(int(q.ts_init),float(q.bid_price))
        else:record.append((int(q.ts_init),float(q.bid_price)))
        pending.append(q)
    def flush():
        nonlocal tp_price,buy_prices,liquidation_price
        if not pending:return
        engine.add_data(pending);engine.run(streaming=True);engine.clear_data();pending.clear()
        # A fill callback can precede cancel/account settlement. Advance only
        # event time at the already-observed price, never to a future price.
        # Three passes cover exchange, portfolio and strategy callback phases;
        # an unaffordable base stays queued for the next real quote.
        for _ in range(3):
            if not (strategy.restart_queued and strategy.restart and not strategy.qty and not strategy.pending_base):break
            q=engine.cache.quote(inst.id)
            when=int(q.ts_init)+1
            if when//NS_HOUR!=int(q.ts_init)//NS_HOUR:break
            account=engine.cache.account_for_venue(BINANCE)
            if not engine.cache.orders_open(instrument_id=inst.id) and number(float(account.balance_free(USDT)))<strategy.base:break
            retry=QuoteTick(inst.id,q.bid_price,q.ask_price,q.bid_size,q.ask_size,when,when)
            append_quote(retry)
            engine.add_data(pending);engine.run(streaming=True);engine.clear_data();pending.clear()
        live=engine.cache.orders_open(instrument_id=inst.id)
        tp_price=next((float(o.price) for o in live if o.side==OrderSide.SELL and o.price is not None),None)
        buy_prices=[float(o.price) for o in live if o.side==OrderSide.BUY and o.price is not None]
        liquidation_price=None
        if strategy.leverage>1 and strategy.qty>0:
            account=engine.cache.account_for_venue(BINANCE)
            maintenance=account.maintenance_margin(inst.id)
            if maintenance is not None:
                # Use the maintenance balance calculated by the native engine,
                # not a second independent margin model. For a long position,
                # wallet + qty*price - entry_notional == maintenance at trigger.
                threshold=(strategy.notional-number(strategy.cash())+number(float(maintenance)))/strategy.qty
                threshold=(threshold/strategy.price_step).to_integral_value(rounding=ROUND_DOWN)*strategy.price_step
                if threshold>0:liquidation_price=float(threshold)
    previous=None
    for item in data:
        if not isinstance(item,QuoteTick):pending.append(item);flush();continue
        when=int(item.ts_init);end=float(item.bid_price)
        if crossings and previous is not None and previous.ts_init//NS_HOUR==when//NS_HOUR:
            start=float(previous.bid_price);cur=start;last=int(previous.ts_init)
            while end!=cur:
                if end>cur:
                    target=tp_price if tp_price is not None and cur<tp_price<=end else None
                else:
                    candidates=[p for p in buy_prices if end<=p<cur]
                    if liquidation_price is not None and end<=liquidation_price<=cur:
                        point=min(liquidation_price,float(np.nextafter(cur,-np.inf)))
                        if point>=end:candidates.append(point)
                    target=max(candidates) if candidates else None
                if target is None:break
                # A queued boundary/fee may have updated native state.
                if pending:flush();continue
                t=max(last+1,min(when,int(previous.ts_init+(when-previous.ts_init)*(target-start)/(end-start))))
                if t>when:break
                px=Price(target,inst.price_precision)
                q=QuoteTick(inst.id,px,px,item.bid_size,item.ask_size,t,t)
                append_quote(q);flush();cur=target;last=max(t,record[-1][0])
                if t==when:break
        if not record or record[-1][0]<when or record[-1][1]!=end:append_quote(item)
        boundary=(when in strategy.openings or when-1000 in strategy.ends
                  or when+1 in strategy.ends)
        if boundary or strategy.restart_queued or not crossings:flush()
        previous=item
    flush()
    return np.array([v[0] for v in record],dtype=np.int64),np.array([v[1] for v in record])


def run_native(symbol,bars,funding,marks,resize=True,path='OHL',windows=None,initial_cash=INITIAL,purge_history=True,
               step=.008,tp=.012,amount=1.2,levels=14,actual_funding=False,
               price_step='0.000000000001',quantity_step='0.00000001',tp_mode='price',continuous_path=True,
               leverage=1,compound=False,maintenance_rate=.005,step_scale=1):
    if leverage<1 or leverage>100:raise ValueError('INVALID_RESEARCH_LEVERAGE')
    if maintenance_rate<=0 or maintenance_rate*leverage>=1:raise ValueError('INVALID_MAINTENANCE_RATE')
    coverage=sum(number(step)*number(step_scale)**i for i in range(levels))
    if number(step)<=0 or number(step_scale)<=0 or not 0<=coverage<1:raise ValueError('INVALID_GRID_COVERAGE')
    inst=instrument(symbol,price_step,quantity_step,leverage,maintenance_rate)
    if windows is None:raise ValueError("Explicit confirmed gate windows are required")
    else:plan={}
    engine=BacktestEngine(BacktestEngineConfig(run_analysis=False,portfolio=PortfolioConfig(use_mark_prices=False),
        exec_engine=ExecutionEngineConfig(purge_account_events_interval_mins=60,purge_account_events_lookback_mins=0) if purge_history else None,
        logging=LoggerConfig(stdout_level=LogLevel.ERROR,print_config=False,bypass_logging=False)))
    engine.add_venue(venue=BINANCE,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,
        starting_balances=[Money.from_str(f'{initial_cash:.8f} USDT')],default_leverage=number(leverage),
        fee_model=MakerTakerFeeModel(maker_rate=FEE,taker_rate=FEE),
        bar_execution=False,trade_execution=False,liquidity_consumption=True,
        liquidation_enabled=True)
    engine.add_instrument(inst)
    strategy=ZigzagDca(StrategyConfig(log_events=False,log_commands=False))
    strategy.setup(inst,windows,resize,step,tp,amount,levels,tp_mode,leverage,compound,step_scale)
    strategy.budget_capital=initial_cash
    engine.add_strategy(strategy)
    data,missing=market_events(inst,bars,funding,marks,path,actual_funding)
    quote_times,quote_prices=replay_continuous(engine,strategy,inst,data,crossings=continuous_path)
    if strategy.errors:
        errors=strategy.errors.copy();engine.dispose()
        raise RuntimeError('Nautilus callback failed: '+errors[0])
    strategy.mark()
    equity=np.array(strategy.equity,dtype=float).reshape(-1,2)
    wallet=strategy.cash();final=float(equity[-1,1]) if len(equity) else wallet
    open_positions=engine.cache.positions_open(instrument_id=inst.id)
    fees=sum(v['commission'] for v in strategy.fills)
    cashflow=sum(v['qty']*v['price']*(1 if 'SELL' in v['side'] else -1) for v in strategy.fills)
    inferred_funding=initial_cash+cashflow-fees+float(strategy.qty*strategy.last_price)-final
    peak=np.maximum.accumulate(np.r_[initial_cash,equity[:,1]])[1:]
    result=dict(symbol=symbol,path=path,resize=resize,ret=final/initial_cash-1,final=final,wallet=wallet,
        tp_mode=tp_mode,path_mode='continuous_order_crossings' if continuous_path else 'four_quotes',
        leverage=leverage,compound=compound,compound_scope='gate' if compound else 'none',maintenance_rate=maintenance_rate,
        step=step,step_scale=step_scale,levels=levels,amount=amount,tp=tp,grid_coverage=float(coverage),
        open_quantity=float(strategy.qty),open_cost_including_buy_fees=float(strategy.cost),last_price=float(strategy.last_price),
        maxdd=float((equity[:,1]/peak-1).min()) if len(equity) else 0,
        windows=len(windows),fills=len(strategy.fills),tps=strategy.tp_count,
        base_fills=strategy.base_fills,add_fills=strategy.add_fills,
        fees=fees,native_funding_from_cashflow=inferred_funding,missing_mark_events=missing,
        liquidations=len({v['order_id'] for v in strategy.fills if v['kind']=='LIQUIDATION'}),
        open_positions=len(open_positions),native_order_count=engine.cache.orders_total_count(),
        rejected=len(strategy.rejections),planning=plan,
        instrument_metadata=f'price step={price_step} quantity step={quantity_step}; historical exchange filters unknown')
    trace=dict(equity=equity,fills=strategy.fills,rejections=strategy.rejections,
               quote_times=quote_times,quote_prices=quote_prices,
               time_ns=np.array([v[0] for v in strategy.equity],dtype=np.int64),
               window_capital=strategy.window_capital,cycle_capital=strategy.cycle_capital)
    engine.dispose()
    return result,trace
