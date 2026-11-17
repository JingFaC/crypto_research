"""Frozen 20x shared-collateral ST channel/ATR grids. No new parameter fitting."""
from pathlib import Path
from types import SimpleNamespace
import math,json,heapq,itertools
import numpy as np
import pandas as pd
if __package__:
    from . import market_core
    from .grid_core import supertrend_bands
else:
    import market_core
    from grid_core import supertrend_bands
CAPITAL,ALLOCATION,LEVERAGE=40000.,20000.,20.
SYMBOL='SOXLUSDT';SIDES=('LONG','SHORT');ATR_LENGTH=14
RISK={}
run=SimpleNamespace(stamp=lambda t:str(pd.Timestamp(t,unit='ms',tz='UTC').tz_convert('Asia/Shanghai')))
def maintenance(symbol, notional):
    for tier in RISK[symbol]["tiers"]:
        if notional <= tier["bracketNotionalCap"]:
            return max(0., notional*tier["bracketMaintenanceMarginRate"]-tier["cumFastMaintenanceAmount"])
    raise ValueError(f"notional beyond public risk table: {symbol}, {notional}")
risk=SimpleNamespace(RISK=RISK,maintenance=maintenance)
class SharedAccount:
    def __init__(self):
        self.legs = {}
        self.marks = {}
        self.curve = []
        self.states = []
        self.liquidations = []
        self.insurance_credit = 0.
        self.stopped = False
        self.rejections = []
        self.min_available = float("inf")
        self.risk_minutes = 0
        self.min_low_buffer = float("inf")

    def wallet(self):
        return sum(g.cash for g in self.legs.values())+self.insurance_credit

    def equity(self):
        return self.wallet()+sum(g.position*self.marks[s]-g.cost for s, g in self.legs.items())

    def maintenance(self):
        return sum(risk.maintenance(s, g.position*self.marks[s]) for s, g in self.legs.items())

    def initial_required(self):
        # All logical buys, not just the native 169-order window: conservative reserve.
        return sum((g.position*self.marks[s]+(g.qty*g.price_prefix[g.frontier] if not g.stopped else 0.))/LEVERAGE for s, g in self.legs.items())

    def remember_state(self):
        if len(self.legs) != 2:
            return
        state = (self.wallet(), *[v for g in self.legs.values() for v in (g.position, g.cost)])
        if not self.states or self.states[-1] != state:
            self.states.append(state)

    def can_buy(self, grid, timestamp, slot, price, taker):
        mark = self.marks[grid.symbol]
        fee = price*grid.qty*(.0004 if taker else 0.)
        prospective_equity = self.equity()+grid.qty*(mark-price)-fee
        prospective_required = self.initial_required()+grid.qty*(mark-grid.levels[slot])/LEVERAGE
        if prospective_equity+1e-8 >= prospective_required:
            return True
        self.rejections.append({"time_bjt": run.stamp(timestamp), "symbol": grid.symbol,
            "equity_after_fill": prospective_equity, "required_after_fill": prospective_required})
        return False

    def liquidate(self, timestamp):
        before = self.equity()
        required = self.maintenance()
        active = [(s, g, g.position*self.marks[s]*risk.RISK[s]["liquidation_fee_rate"]) for s, g in self.legs.items() if g.position > 1e-8]
        fee_due = sum(v for s, g, v in active)
        total_fee = min(max(before, 0.), fee_due)
        positions = []
        for s, g, due in active:
            price, qty = self.marks[s], g.position
            pnl = qty*price-g.cost
            fee = total_fee*due/fee_due if fee_due else 0.
            g.cash += pnl-fee
            g.realized += pnl
            g.fees += fee
            g.clearance_fees += fee
            g.fills.append((timestamp, "LIQUIDATE", -1, price, qty, fee, pnl, False, 0.))
            positions.append({"symbol": s, "mark": price, "quantity": qty, "clearance_fee": fee})
            g.position = g.cost = 0.
            g.frontier = g.count
            g.costs.fill(0.)
        credit = max(0., -self.wallet())
        self.insurance_credit += credit
        self.stopped = True
        for g in self.legs.values():
            g.stopped = {"reason": "shared_account_liquidation", "time_bjt": run.stamp(timestamp)}
        self.liquidations.append({"timestamp": timestamp, "time_bjt": run.stamp(timestamp),
            "account_equity_before": before, "maintenance_required": required,
            "clearance_fee": total_fee, "bankruptcy_writeoff": credit,
            "wallet_after": self.wallet(), "positions": positions})

    def sample_marks(self, timestamp, marks):
        assert set(marks) == set(self.legs), "both contemporaneous marks are required"
        # Updating only one side before checking risk would invent a hedge loss.
        self.marks.update({s: pair[0] for s, pair in marks.items()})
        self.remember_state()
        if self.states:
            low_buffer = float("inf")
            for state in self.states:
                eq, mm = state[0], 0.
                for i, s in enumerate(self.legs):
                    qty, cost = state[1+i*2:3+i*2]
                    low = marks[s][1]
                    eq += qty*low-cost
                    mm += risk.maintenance(s, qty*low)
                low_buffer = min(low_buffer, eq-mm)
            self.min_low_buffer = min(self.min_low_buffer, low_buffer)
            self.risk_minutes += int(low_buffer <= 0)
        if not self.stopped and any(g.position > 1e-8 for g in self.legs.values()) and self.equity() <= self.maintenance():
            self.liquidate(timestamp)
        equity, wallet, mm, required = self.equity(), self.wallet(), self.maintenance(), self.initial_required()
        self.min_available = min(self.min_available, equity-required)
        realised = sum(g.realized for g in self.legs.values())
        fees = sum(g.fees for g in self.legs.values())
        funding = sum(g.funding for g in self.legs.values())
        assert math.isclose(equity-CAPITAL, realised-fees+funding+equity-wallet+self.insurance_credit, abs_tol=1e-5)
        self.curve.append((timestamp, equity, wallet, equity-wallet, mm, required, equity-required,
            realised, fees, funding, self.insurance_credit))
        for s, g in self.legs.items():
            mark = self.marks[s]
            unrealized = g.position*mark-g.cost
            g.max_notional = max(g.max_notional, g.position*mark)
            assert g.position*mark <= risk.RISK[s]["max_notional_20x"]
            g.curve.append((timestamp, g.cash-g.capital+unrealized, g.cash-g.capital,
                g.position, g.cost, mark, g.realized, g.fees, g.funding))
        self.states = []
        self.remember_state()
cross=SimpleNamespace(SharedAccount=SharedAccount)
class TrendGrid:
    def __init__(self, account, symbol):
        self.account, self.symbol = account, symbol
        self.cash = ALLOCATION
        self.position = self.cost = self.realized = self.fees = self.funding = self.clearance_fees = 0.
        self.qty = self.max_notional = 0.
        self.side, self.count, self.frontier, self.regime = 1, 0, 0, -1
        self.levels = self.costs = np.array([])
        self.pending = None
        self.started = self.blocked = False
        self.fills, self.curve, self.activations, self.funding_logs = [], [], [], []

    def reserve(self, mark, *, position=None, frontier=None):
        position = self.position if position is None else position
        frontier = self.frontier if frontier is None else frontier
        orders = 0.
        if not self.blocked and self.pending is None:
            prices = self.levels[:frontier] if self.side == 1 else np.maximum(mark, self.levels[frontier+1:])
            orders = self.qty*float(np.sum(prices))
        return (abs(position)*mark+orders)/LEVERAGE

    def fill(self, timestamp, slot, price, quantity, opening, purpose, *, fee_rate=None):
        taker = purpose in ("INITIAL", "FLIP_CLOSE")
        fee = quantity*price*((.0004 if taker else 0.) if fee_rate is None else fee_rate)
        pnl = matched = 0.
        trade_side = self.side if opening else -self.side
        if opening:
            self.costs[slot] = price
            self.position += self.side*quantity
            self.cost += self.side*quantity*price
        else:
            assert quantity <= abs(self.position)+1e-7
            average = self.cost/self.position
            pnl = self.side*(price-average)*quantity
            matched = (self.side*float(np.sum(price-self.costs[self.costs > 0]))*self.qty
                if slot < 0 else self.side*(price-self.costs[slot])*quantity)
            self.position -= self.side*quantity
            self.cost -= self.side*quantity*average
            if abs(self.position) < 1e-7:
                self.position = self.cost = 0.
            if slot >= 0:
                self.costs[slot] = 0.
            self.realized += pnl
        self.cash += pnl-fee
        self.fees += fee
        self.fills.append((timestamp, "BUY" if trade_side == 1 else "SELL", purpose, self.regime,
            slot, price, quantity, fee, pnl, matched, self.position, self.cost))
        self.account.remember_state()

    def activate(self, timestamp, price, plan):
        if abs(self.position) > 1e-7:
            self.fill(timestamp, -1, price, abs(self.position), False, "FLIP_CLOSE")
        self.side, self.regime, self.count = plan["side"], plan["id"], plan["count"]
        self.levels = np.array(plan["ticks"], dtype=float)/100
        self.costs = np.zeros(self.count)
        self.frontier = self.count if self.side == 1 else 0
        self.started = self.blocked = False
        mark = self.account.marks[self.symbol]
        entries = self.levels[:-1] if self.side == 1 else self.levels[1:]
        assuming = entries if self.side == 1 else np.maximum(mark, entries)
        denom = np.sum(assuming+LEVERAGE*np.maximum(-self.side*(mark-entries), 0.))
        self.qty = math.floor(.8*ALLOCATION*LEVERAGE/denom*100+1e-9)/100
        assert self.qty*self.levels[0] >= 5
        self.pending = None
        self.activations.append(dict(plan, execution_time=timestamp, execution_price=price,
            sizing_mark=mark, qty_per_grid=self.qty, allocated_margin=ALLOCATION))
        if self.account.equity()+1e-8 < self.account.initial_required():
            self.blocked = True
            self.account.rejections.append(dict(timestamp=timestamp, symbol=self.symbol,
                regime=self.regime, reason="insufficient_shared_margin_at_creation"))

    def open(self, timestamp, slot, price, initial=False):
        new_frontier = self.frontier-self.side
        mark = self.account.marks[self.symbol]
        fee = price*self.qty*(.0004 if initial else 0.)
        new_position = self.position+self.side*self.qty
        equity = self.account.equity()+self.side*self.qty*(mark-price)-fee
        required = self.account.initial_required()-self.reserve(mark)+self.reserve(mark, position=new_position, frontier=new_frontier)
        reason = ""
        if abs(new_position)*mark > risk.RISK[self.symbol]["max_notional_20x"]:
            reason = "public_20x_position_cap"
        elif equity+1e-8 < required:
            reason = "insufficient_shared_initial_margin"
        if reason:
            self.blocked = True
            self.account.rejections.append(dict(timestamp=timestamp, symbol=self.symbol,
                regime=self.regime, reason=reason, equity_after_fill=equity, required_after_fill=required))
            return False
        self.fill(timestamp, slot, price, self.qty, True, "INITIAL" if initial else "GRID_OPEN")
        self.frontier = new_frontier
        return True

    def trade(self, timestamp, price):
        if self.account.stopped:
            return
        if self.pending is not None:
            self.activate(timestamp, price, self.pending)
        if self.regime < 0:
            return
        if not self.started:
            if (self.side == 1 and price < self.levels[0]) or (self.side == -1 and price > self.levels[-1]):
                return
            self.started = True
            if self.levels[0] <= price <= self.levels[-1] and not self.blocked:
                target = int(np.searchsorted(self.levels[:-1], price, side="right")) if self.side == 1 else int(np.searchsorted(self.levels[1:], price, side="left"))
                while self.frontier != target:
                    slot = self.frontier-1 if self.side == 1 else self.frontier
                    if not self.open(timestamp, slot, price, initial=True):
                        break
        if self.side == 1:
            while not self.blocked and self.frontier > 0 and price < self.levels[self.frontier-1]-1e-9:
                if not self.open(timestamp, self.frontier-1, self.levels[self.frontier-1]):
                    break
            while self.frontier < self.count and price > self.levels[self.frontier+1]+1e-9:
                self.fill(timestamp, self.frontier, self.levels[self.frontier+1], self.qty, False, "GRID_CLOSE")
                self.frontier += 1
        else:
            while not self.blocked and self.frontier < self.count and price > self.levels[self.frontier+1]+1e-9:
                if not self.open(timestamp, self.frontier, self.levels[self.frontier+1]):
                    break
            while self.frontier > 0 and price < self.levels[self.frontier-1]-1e-9:
                slot = self.frontier-1
                self.fill(timestamp, slot, self.levels[slot], self.qty, False, "GRID_CLOSE")
                self.frontier -= 1

    def funding_event(self, timestamp, rate, mark):
        payment = -self.position*mark*rate
        self.funding_logs.append((timestamp, rate, mark, self.position, payment))
        self.cash += payment
        self.funding += payment
        self.account.remember_state()

class Account(cross.SharedAccount):
    def maintenance(self):
        return sum(risk.maintenance(s, abs(g.position)*self.marks[s]) for s, g in self.legs.items())

    def initial_required(self):
        return sum(g.reserve(self.marks[s]) for s, g in self.legs.items())

    def liquidate(self, timestamp):
        before, mm = self.equity(), self.maintenance()
        due = {s: abs(g.position)*self.marks[s]*risk.RISK[s]["liquidation_fee_rate"] for s, g in self.legs.items()}
        total_fee = min(max(before, 0.), sum(due.values()))
        positions = []
        for s, g in self.legs.items():
            quantity, price = abs(g.position), self.marks[s]
            fee = total_fee*due[s]/sum(due.values()) if sum(due.values()) else 0.
            if quantity > 1e-7:
                positions.append(dict(symbol=s, signed_position=g.position, price=price))
                g.fill(timestamp, -1, price, quantity, False, "LIQUIDATE", fee_rate=fee/(quantity*price))
                g.clearance_fees += fee
            g.costs.fill(0.)
            g.frontier = g.count if g.side == 1 else 0
            g.blocked, g.pending = True, None
        credit = max(0., -self.wallet())
        self.insurance_credit += credit
        self.stopped = True
        self.liquidations.append(dict(timestamp=timestamp, time_bjt=run.stamp(timestamp),
            equity_before=before, maintenance_required=mm, clearance_fee=total_fee,
            bankruptcy_writeoff=credit, positions=positions))

    def sample_marks(self, timestamp, marks):
        assert set(marks) == set(self.legs)
        self.marks.update({s: v[0] for s, v in marks.items()})
        self.remember_state()
        bound = float("inf")
        for state in self.states:
            eq, mm = state[0], 0.
            for i, s in enumerate(self.legs):
                quantity, cost = state[1+i*2:3+i*2]
                worst = marks[s][2] if quantity >= 0 else marks[s][1]
                eq += quantity*worst-cost
                mm += risk.maintenance(s, abs(quantity)*worst)
            bound = min(bound, eq-mm)
        self.min_low_buffer = min(self.min_low_buffer, bound)
        self.risk_minutes += int(bound <= 0)
        if not self.stopped and any(abs(g.position) > 1e-7 for g in self.legs.values()) and self.equity() <= self.maintenance():
            self.liquidate(timestamp)
        eq, wallet, mm, required = self.equity(), self.wallet(), self.maintenance(), self.initial_required()
        self.min_available = min(self.min_available, eq-required)
        realized, fees, funding = (sum(getattr(g, k) for g in self.legs.values()) for k in ("realized", "fees", "funding"))
        assert math.isclose(eq-CAPITAL, realized-fees+funding+eq-wallet+self.insurance_credit, abs_tol=1e-5)
        self.curve.append((timestamp, eq, wallet, eq-wallet, mm, required, eq-required, realized, fees, funding, self.insurance_credit, bound))
        for s, g in self.legs.items():
            mark = self.marks[s]
            g.max_notional = max(g.max_notional, abs(g.position)*mark)
            g.curve.append((timestamp, g.cash-ALLOCATION+g.position*mark-g.cost, g.cash-ALLOCATION,
                g.position, g.cost, mark, g.realized, g.fees, g.funding, g.regime, g.side))
        self.states = []
        self.remember_state()
engine=SimpleNamespace(Account=Account,TrendGrid=TrendGrid)
class HedgeAccount(engine.Account):
    def __init__(self):
        super().__init__()
        self.config = dict(name="soxl_hedge", contract=SYMBOL, hedge=True,
            stop=False, sizing=False, guard=False)
        self.halt_request = None
        self.stop_events = []

    def sample_common_mark(self, timestamp, close, high, low):
        # Same contract: one observed mark updates both sides atomically.
        self.marks.update({s: close for s in SIDES})
        self.remember_state()
        bound = float("inf")
        for state in self.states:
            # The price is common, including the intraminute stress endpoints.
            # Using a long low and an independent short high would invent loss.
            for price in (low, high):
                equity, mm = state[0], 0.
                for i, side in enumerate(SIDES):
                    quantity, cost = state[1+i*2:3+i*2]
                    equity += quantity*price-cost
                    mm += risk.maintenance(side, abs(quantity)*price)
                bound = min(bound, equity-mm)
        self.min_low_buffer = min(self.min_low_buffer, bound)
        self.risk_minutes += int(bound <= 0)
        if not self.stopped and any(abs(g.position) > 1e-7 for g in self.legs.values()) and self.equity() <= self.maintenance():
            self.liquidate(timestamp)
        equity, wallet, mm, required = self.equity(), self.wallet(), self.maintenance(), self.initial_required()
        self.min_available = min(self.min_available, equity-required)
        realized, fees, funding = (sum(getattr(g, key) for g in self.legs.values()) for key in ("realized", "fees", "funding"))
        assert math.isclose(equity-40000., realized-fees+funding+equity-wallet+self.insurance_credit, abs_tol=1e-5)
        self.curve.append((timestamp, equity, wallet, equity-wallet, mm, required,
            equity-required, realized, fees, funding, self.insurance_credit, bound))
        for side, g in self.legs.items():
            g.max_notional = max(g.max_notional, abs(g.position)*close)
            g.curve.append((timestamp, g.cash-20000.+g.position*close-g.cost, g.cash-20000.,
                g.position, g.cost, close, g.realized, g.fees, g.funding, g.regime, g.side))
        self.states = []
        self.remember_state()

class HedgeGrid(engine.TrendGrid):
    def __init__(self, account, side):
        super().__init__(account, side)
        self.grid_side = 1 if side == "LONG" else -1
        self.side = self.grid_side
        self.enabled = False

    def reserve(self, mark, *, position=None, frontier=None):
        # Binance hedge-mode IM: sum of side-specific max exposure branches.
        position = self.position if position is None else position
        frontier = self.frontier if frontier is None else frontier
        N = position*mark
        pending = self.pending is None
        opening = pending and self.enabled and not self.blocked
        closing = pending and abs(position) > 1e-7
        bids = self.qty*float(self.levels[:frontier].sum()) if ((self.side == 1 and opening) or (self.side == -1 and closing)) else 0.
        asks = self.qty*float(self.levels[frontier+1:].sum()) if ((self.side == -1 and opening) or (self.side == 1 and closing)) else 0.
        return max(abs(N+bids), abs(N-asks))/20.

    def activate(self, timestamp, price, plan):
        if plan["side"] == 1:
            self.enabled = True
            real = dict(plan, side=self.grid_side, signal_side=1, grid_side=self.grid_side)
            super().activate(timestamp, price, real)
            self.activations[-1]["mode"] = "long" if self.grid_side == 1 else "short"
            return
        if abs(self.position) > 1e-7:
            self.fill(timestamp, -1, price, abs(self.position), False, "ST_DOWN_CLOSE", fee_rate=.0004)
        self.enabled = False
        self.side, self.regime, self.count, self.frontier = self.grid_side, plan["id"], 0, 0
        self.levels, self.costs = np.array([]), np.array([])
        self.qty = 0.
        self.pending = None
        self.started, self.blocked = False, True
        self.activations.append(dict(plan, grid_side=self.grid_side, signal_side=-1,
            execution_time=timestamp, execution_price=price,
            sizing_mark=self.account.marks[self.symbol], qty_per_grid=0., allocated_margin=0., mode="flat"))

    def open(self, timestamp, slot, price, initial=False):
        mark = self.account.marks[self.symbol]
        gross = (sum(abs(g.position) for g in self.account.legs.values())+self.qty)*mark
        if gross > risk.RISK[self.symbol]["max_notional_20x"]+1e-7:
            self.blocked = True
            self.account.rejections.append(dict(timestamp=timestamp, symbol=SYMBOL,
                side=self.symbol, regime=self.regime, reason="combined_absolute_20x_position_cap", proposed_gross_notional=gross))
            return False
        return super().open(timestamp, slot, price, initial)

    def trade(self, timestamp, price):
        if self.account.stopped:
            return
        if self.pending is not None:
            self.activate(timestamp, price, self.pending)
        if self.enabled:
            super().trade(timestamp, price)
hedge=SimpleNamespace(HedgeAccount=HedgeAccount,HedgeGrid=HedgeGrid)
class SplitAccount(hedge.HedgeAccount):
    def __init__(self):
        super().__init__()
        self.config["name"] = "soxl_split"
        self.config["zones"] = "four user-specified ST-relative intervals"

    def sample_common_mark(self, timestamp, close, high, low):
        # A flat bankrupt account has no remaining price exposure. Do not count
        # its subsequent zero balances as additional intraminute risk events.
        exposed = any(abs(g.position) > 1e-7 for g in self.legs.values()) or any(
            abs(s[1])+abs(s[3]) > 1e-7 for s in self.states)
        previous = self.risk_minutes
        super().sample_common_mark(timestamp, close, high, low)
        if not exposed:
            self.risk_minutes = previous

class SplitGrid(hedge.HedgeGrid):
    def activate(self, timestamp, price, plan):
        self.enabled = True
        real = dict(plan, **plan["zones"][self.symbol])
        real.update(side=self.grid_side, signal_side=plan["side"], grid_side=self.grid_side)
        # Both ST states are active; bypass the earlier bullish-only gate.
        engine.TrendGrid.activate(self, timestamp, price, real)
        self.activations[-1]["mode"] = "long" if self.grid_side == 1 else "short"
split=SimpleNamespace(SplitAccount=SplitAccount,SplitGrid=SplitGrid)
def atr14(rows):
    a = np.asarray(rows, dtype=float)
    h, l, c = a[:, 2], a[:, 3], a[:, 4]
    previous = np.r_[c[0], c[:-1]]
    tr = np.maximum(h-l, np.maximum(abs(h-previous), abs(l-previous)))
    out = np.full(len(rows), np.nan)
    if len(out) >= ATR_LENGTH:
        out[ATR_LENGTH-1] = tr[:ATR_LENGTH].mean()
        for i in range(ATR_LENGTH, len(out)):
            out[i] = (out[i-1]*(ATR_LENGTH-1)+tr[i])/ATR_LENGTH
    return out

def geometry(st, width, direction):
    pairs = ({"LONG": (st, st+width), "SHORT": (st+width, st+2*width)} if direction == 1
             else {"LONG": (st-2*width, st-width), "SHORT": (st-width, st)})
    if not np.isfinite(width) or width <= 0 or any(lo <= 0 or hi <= lo for lo, hi in pairs.values()):
        return None, "nonpositive_or_invalid_bounds"
    zones = {}
    for side, (lo, hi) in pairs.items():
        count = round(math.log(hi/lo)/math.log(1.0025))
        if count < 1 or count > 1000:
            return None, "grid_count_unexecutable"
        ticks = np.floor(np.geomspace(lo, hi, count+1)*100+1e-7).astype(np.int64)
        if not np.all(np.diff(ticks) > 0):
            return None, "duplicate_rounded_price_levels"
        zones[side] = dict(lower=lo, upper=hi, count=count, ticks=ticks.tolist())
    return zones, None

class ReviewAccount(split.SplitAccount):
    def __init__(self, name):
        super().__init__()
        self.config["name"] = name
        self.skip_events = []

class ReviewGrid(split.SplitGrid):
    def idle(self, timestamp, price, plan, reason):
        if abs(self.position) > 1e-7:
            self.fill(timestamp, -1, price, abs(self.position), False, "FLIP_CLOSE")
        self.enabled = False
        self.regime, self.count, self.frontier, self.qty = plan["id"], 0, 0, 0.
        self.levels, self.costs = np.array([]), np.array([])
        self.pending = None
        self.started, self.blocked = False, True
        self.activations.append(dict(plan, execution_time=timestamp, execution_price=price,
            mode="skipped", reason=reason, qty_per_grid=0., lower=None, upper=None, count=0))
        self.account.skip_events.append(dict(timestamp=timestamp, side=self.symbol, regime=self.regime, reason=reason))

    def activate(self, timestamp, price, plan):
        if not plan["valid"]:
            self.idle(timestamp, price, plan, plan["invalid_reason"])
            return
        if abs(self.position) > 1e-7:
            self.fill(timestamp, -1, price, abs(self.position), False, "FLIP_CLOSE")
        z = plan["zones"][self.symbol]
        levels = np.array(z["ticks"])/100.
        mark = self.account.marks[self.symbol]
        entries = levels[:-1] if self.grid_side == 1 else levels[1:]
        assuming = entries if self.grid_side == 1 else np.maximum(mark, entries)
        qty = math.floor(.8*20000.*20/np.sum(assuming+20*np.maximum(-self.grid_side*(mark-entries), 0.))*100+1e-9)/100
        if qty*levels[0] < 5:
            self.idle(timestamp, price, plan, "minimum_notional_below_5U")
            return
        super().activate(timestamp, price, plan)
        assert abs(self.qty-qty) < 1e-9
def plans_from_ohlc(rows,strategy,start,end):
    a=np.asarray(rows,float);t=a[:,0].astype(np.int64);atr,line,up,lo,hi=supertrend_bands(a,6,2.5);at=atr14(a)
    valid=np.isfinite(line)&np.isfinite(at);prior=np.flatnonzero(valid&(t+14400000<=start))
    indices=[int(prior[-1])]+np.flatnonzero(valid&(t+14400000>start)&(t+14400000<end)).tolist();plans=[]
    for i in indices:
        side=1 if up[i] else -1
        if plans and plans[-1]['side']==side:continue
        st=float(line[i]);width=float(1.5*(hi[i]-lo[i]) if strategy=='channel20x' else 2.087419701989316*at[i]);zones,reason=geometry(st,width,side)
        plans.append(dict(id=len(plans),side=side,st=st,width=width,source_bar_open=int(t[i]),signal_time=max(start,int(t[i]+14400000)),valid=zones is not None,invalid_reason=reason,zones=zones or {},ticks=[] if zones is None else sorted(set(v for z in zones.values() for v in z['ticks']))))
    return plans

def filtered_quotes(times,prices,ticks,start,end):
    # Same level-code/minute-preserving filter as the original public trade replay,
    # applied to the SHA-verified all-price-change raw tape (not old fills).
    i0=int(np.searchsorted(times,start));i1=int(np.searchsorted(times,end));previous_code=previous_minute=None
    for begin in range(i0,i1,400000):
        t=np.asarray(times[begin:min(begin+400000,i1)]);p=np.asarray(prices[begin:min(begin+400000,i1)]);left=np.searchsorted(ticks,p,'left');right=np.searchsorted(ticks,p,'right');code=left*2+right-left;minutes=t//60000
        mask=np.ones(len(t),bool);mask[1:]=(code[1:]!=code[:-1])|(minutes[1:]!=minutes[:-1]);mask[0]=previous_code is None or code[0]!=previous_code or minutes[0]!=previous_minute;mask[-1]=True;previous_code,previous_minute=int(code[-1]),int(minutes[-1])
        for j in np.flatnonzero(mask):yield int(t[j]),0,'trade',float(p[j])/100.,0.

def evaluate(legacy_root,strategy):
    data,hashes=market_core.load_market(legacy_root);times,prices,marks,_,_,_,funding,tiers,cap,clear=data;start=1789272000000;end=1794326400000
    old=Path(legacy_root);p=old/'backtest/reports/sox_grid_20x_cross_st4h_2026-11-11/SOXLUSDT.klines_4h.json';rows=json.loads(p.read_text(encoding='utf-8'));plans=plans_from_ohlc(rows,strategy,start,end)
    raw=json.loads((old/'backtest/reports/soxl_snowball_2026-11-11/risk_snapshot.json').read_text(encoding='utf-8'));tiers=raw['response']['data']['brackets'][0]['riskBrackets'];cap=max(t['bracketNotionalCap'] for t in tiers if t['maxOpenPosLeverage']>=20)
    for side in SIDES:RISK[side]=dict(tiers=tiers,liquidation_fee_rate=clear,max_notional_20x=cap)
    marks=marks[(marks[:,0]>=start)&(marks[:,0]<end)];account=ReviewAccount(strategy)
    for side in SIDES:account.marks[side]=float(marks[0,1]);account.legs[side]=ReviewGrid(account,side)
    account.remember_state();ticks=np.unique(np.concatenate([p['ticks'] for p in plans if p['ticks']])).astype(np.int64)
    streams=[filtered_quotes(times,prices,ticks,start,end),((int(m[5]),1,'mark',float(m[4]),(float(m[2]),float(m[3]))) for m in marks),((int(f['fundingTime']),-1,'funding',float(f['fundingRate']),float(f['markPrice'])) for f in funding if start<=int(f['fundingTime'])<end),((p['signal_time'],-2,'signal',p,0.) for p in plans)]
    for timestamp,g in itertools.groupby(heapq.merge(*streams,key=lambda x:x[0]),key=lambda x:x[0]):
        batch=list(g)
        for ts,_,kind,value,extra in batch:
            if kind=='funding':
                for leg in account.legs.values():leg.funding_event(ts,value,extra)
            elif kind=='signal' and not account.stopped:
                for leg in account.legs.values():leg.pending=value
        for ts,_,kind,value,extra in batch:
            if kind=='trade':
                for leg in account.legs.values():leg.trade(ts,value)
            elif kind=='mark':account.sample_common_mark(ts,value,*extra)
        if account.stopped:break
    curve=np.asarray(account.curve);e=np.r_[40000.,curve[:,1]];result=dict(return_pct=float((account.equity()/40000-1)*100),dd_pct=float(100*(1-e/np.maximum.accumulate(e)).max()),final_equity=float(account.equity()),liquidations=len(account.liquidations),stopped=account.stopped,rejections=len(account.rejections),fees=float(sum(g.fees for g in account.legs.values())),funding=float(sum(g.funding for g in account.legs.values())),last_event=int(curve[-1,0]),plans=len(plans),note='Fresh raw-price-change-tape replay with original classes, recomputed 4h ST/ATR, same frozen sizing. Stops after account liquidation; later flat equity remains fixed.')
    return result,hashes
