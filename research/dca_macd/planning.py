"""Frozen causal weekly ZigZag planning; no execution engine or legacy imports."""
import numpy as np
import types
WEEK=604800000
WEEK_OFF=316800000
LEFT,RIGHT,RIGHT_HI,DEV,LEFT_LO=4,1,4,.05,4
HOUR=3600000
def _impulse_retrace(pivots, this_lo):
    """当前低点相对「上一波上涨」的回撤比例。没有完整上涨则 None。"""
    last_high = None
    low_before = None
    prev_low = None
    for _, side, px, _ in pivots:
        if side < 0:
            prev_low = px
        else:
            last_high = px
            low_before = prev_low
    if last_high is None or low_before is None:
        return None
    impulse = last_high - low_before
    if impulse <= 0:
        return None
    return (last_high - this_lo) / impulse

def replay(wc, left=LEFT, right=RIGHT, right_hi=RIGHT_HI, dev=DEV, left_lo=LEFT_LO,
           morph=""):
    """逐根回放。只使用 0..t 已收盘的周。先处理本根开盘该执行的单，再用本根收盘更新。

    返回 (signals, rounds)。
    signals: [(p, side, price, confirm_week)]，白线用这份，已被更高收盘换掉的高点不保留。
    rounds:  [{week, exit, ref, broken, pivot, side}]，exit 为 None 表示收到最后一根仍持仓。

    未定格高点被更高收盘换掉，只改折点，不取消已经在开盘成交的卖出，也不回场。
    回场只发生在高点已被低点定格之后，收盘重新站上卖掉的那个高点。
    """
    wc = [float(x) for x in wc]
    n = len(wc)
    pivots = []
    hi = None
    hi_px = None
    last_lo = None
    pend = None
    pos = None
    pending_exit = None
    pending_entry = None
    last_exit = -1
    re_hp = None
    rounds = []
    lows_since_high = 0
    saw_high = False

    def drop_high():
        for j in range(len(pivots) - 1, -1, -1):
            if pivots[j][1] > 0:
                del pivots[j]
                break

    def arm_exit(exec_i, broken, hp):
        nonlocal pending_exit
        if pending_exit is None or exec_i < pending_exit[0]:
            pending_exit = (exec_i, broken, hp)

    def arm_entry(exec_i, ref, pivot):
        nonlocal pending_entry
        if pos is not None or exec_i <= last_exit:
            return
        if pending_entry is None or exec_i < pending_entry[0]:
            pending_entry = (exec_i, ref, pivot)

    def left_high(i):
        if i < left:
            return False
        c = wc[i]
        if c < max(wc[i - left:i]):
            return False
        if last_lo is not None and not (c > last_lo and (c - last_lo) / last_lo >= dev):
            return False
        return True

    for t in range(n):
        if pending_exit is not None and pending_exit[0] == t and pos is not None:
            rounds.append(dict(week=pos[0], exit=t, ref=pos[1],
                               broken=pending_exit[1], pivot=pos[2], side=-1))
            re_hp = None if pending_exit[1] else pending_exit[2]
            last_exit = t
            pos = None
            pending_exit = None
        if pending_entry is not None and pending_entry[0] == t and pos is None and t > last_exit:
            pos = (t, pending_entry[1], pending_entry[2])
            pending_entry = None
        c = wc[t]
        if t >= 1:
            p = t - 1
            is_lo = (p >= left_lo and c > wc[p]
                     and wc[p] <= min(wc[p - left_lo:p]))
            last_hi_px = next((x[2] for x in reversed(pivots) if x[1] > 0), None)
            if is_lo and last_hi_px is not None and (last_hi_px - wc[p]) / last_hi_px < dev:
                is_lo = False
            if is_lo:
                take = True
                if morph == "two_b" and last_lo is not None and wc[p] < last_lo and c <= last_lo:
                    take = False
                if morph in ("fib", "fib_break", "fib_deep"):
                    retr = _impulse_retrace(pivots, wc[p])
                    if retr is not None:
                        if morph == "fib" and (retr < 0.382 or retr > 0.618):
                            take = False
                        if morph == "fib_break" and retr > 1.0:
                            take = False
                        if morph == "fib_deep" and retr > 0.618:
                            take = False
                if morph == "first_pb" and lows_since_high >= 1:
                    take = False
                if morph == "skip_first" and saw_high and lows_since_high == 0:
                    take = False
                if hi is not None:
                    hi["locked"] = True
                pend = None
                pivots.append((p, -1, wc[p], t))
                last_lo = wc[p]
                lows_since_high += 1
                if take:
                    arm_entry(t + 1, wc[p], p)
        if hi is not None and not hi["locked"] and c > hi["price"]:
            on = bool(hi.get("on"))
            if on:
                drop_high()
                pivots.append((t, 1, c, t))
                lows_since_high = 0
                saw_high = True
            hi = {"i": t, "price": c, "locked": False, "armed": False, "on": on}
            hi_px = c
            re_hp = None
            if pending_exit is not None and not pending_exit[1] and pending_exit[0] > t:
                pending_exit = None
        elif hi is None and left_high(t):
            hi = {"i": t, "price": c, "locked": False, "armed": False, "on": False}
            hi_px = c
        elif hi is not None and hi["locked"] and left_high(t) and (pend is None or c > pend[1]):
            pend = (t, c)
            hi_px = c
        if (hi is not None and not hi["armed"] and t == hi["i"] + right_hi
                and all(x < hi["price"] for x in wc[hi["i"] + 1:t + 1])):
            if not hi.get("on"):
                pivots.append((hi["i"], 1, hi["price"], t))
                hi["on"] = True
                lows_since_high = 0
                saw_high = True
            hi["armed"] = True
            if pos is not None:
                arm_exit(t + 1, False, hi["price"])
        if (pend is not None and t == pend[0] + right_hi
                and all(x < pend[1] for x in wc[pend[0] + 1:t + 1])):
            pivots.append((pend[0], 1, pend[1], t))
            lows_since_high = 0
            saw_high = True
            hi = {"i": pend[0], "price": pend[1], "locked": False, "armed": True, "on": True}
            hi_px = pend[1]
            pend = None
            if pos is not None:
                arm_exit(t + 1, False, hi["price"])
        if pos is not None and c < pos[1]:
            arm_exit(t + 1, True, None)
        if re_hp is not None and pos is None and hi is not None and hi["locked"] and c > re_hp:
            arm_entry(t + 1, re_hp, t)
            re_hp = None
    if pos is not None:
        rounds.append(dict(week=pos[0], exit=None, ref=pos[1],
                           broken=False, pivot=pos[2], side=-1))
    return pivots, rounds
Z=types.SimpleNamespace(replay=replay,WEEK=WEEK,WEEK_OFF=WEEK_OFF)
def planning(bars):
    ts = bars[:, 0].astype(np.int64)
    if len(ts) < 2 or not (np.diff(ts) == HOUR).all(): raise ValueError('NONCONTIGUOUS_1H')
    keys = (ts - Z.WEEK_OFF) // Z.WEEK
    cuts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(ts)]
    weeks = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        start = int(keys[a] * Z.WEEK + Z.WEEK_OFF)
        if ts[a] == start and ts[b - 1] + HOUR == start + Z.WEEK:
            weeks.append([start, bars[b - 1, 4]])
    if not weeks: return [], dict(weeks=0)
    weeks = np.array(weeks); wc = weeks[:, 1]
    state = np.zeros(len(ts), dtype=bool)
    for t in range(len(wc)):
        pivots, _ = Z.replay(wc[:t + 1])
        i = int(np.searchsorted(ts, int(weeks[t, 0]) + Z.WEEK + HOUR))
        if pivots and i < len(ts): state[i:] = pivots[-1][1] < 0
    _, plans = Z.replay(np.r_[wc, wc[-1]])
    windows = []; last = -1; skips = []
    for plan in plans:
        start = int(np.searchsorted(ts, int(weeks[0, 0]) + plan['week'] * Z.WEEK + HOUR))
        if start <= last or start >= len(ts) - 1: continue
        if not state[start]: skips.append(dict(plan, reason='GATE_CLOSED')); continue
        end = len(ts) - 1 if plan['exit'] is None else min(len(ts) - 1, int(np.searchsorted(ts, int(weeks[0, 0]) + plan['exit'] * Z.WEEK + HOUR)))
        allowed = state[start:end].copy()
        splits = np.r_[0, np.flatnonzero(np.diff(allowed.astype(int))) + 1, len(allowed)]
        for a, b in zip(splits[:-1], splits[1:]):
            if allowed[a]:
                windows.append(dict(open_i=start + int(a), close_i=start + int(b), open_ms=int(ts[start + a]), close_ms=int(ts[start + b]),
                                    outer_window=plan['week'], reference=plan['ref'], broken=plan['broken']))
        last = end
    return windows, dict(weeks=len(wc), skipped=skips)