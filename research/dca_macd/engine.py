"""DCA 引擎（唯一实现）：门作用在 entry / add / exit 三个决策点。

口径（M5）
----------
* 全仓逐仓统一：W = 可用现金，spent = 已投保证金，权益 = W + spent + (qty·P − spent·lev)。
* 爆仓：权益 = MMR·名义 → P* = (spent·(lev−1) − W) / (qty·(1−MMR))。爆仓后现金记 0（保守，实际会剩 MMR·名义）。
* bar 内路径两种，默认 opt（保守下界）：
    opt  = O → H → L：先给止盈（用未加仓的高均价），再走加仓 —— 收益更低、回撤更深
    pess = O → L → H：先加仓（压低均价 → tpp 下移），再给止盈

  重要更正（2026-10-29 复审）：此前把 pess 标为"对多头最不利"是错的。V 型 bar 里
  pess 会先加仓把均价压下去，同一个 high 就够到更低的 tpp，于是「本来出不来的一轮
  被算成止盈」，结果是投入更大、止盈更多、收益更高。实测同一参数全量：
  pess +1336% / DD −51.7%，opt +825% / DD −70.8%，差 511pp。
  真实 5m 内部路径未知，故真值落在这两条之间；**对外只报 opt，pess 仅作上界参考**。
* 费用：双边 taker FEE；资金费按持仓名义逐 bar 扣，**按事件时刻的 bar 开盘价计价**
  （复审 2026-10-30：此前用当根收盘价 `c[i]`，那是未来函数）。
* 滑点：`slip` 只作用在**市价单**上（开仓 / 加仓 / 门关平仓），方向一律不利；
  止盈是限价单，成交价就是 tpp，不再加滑点。默认 0，灵敏度另测。
* 档位跌幅按 docs/dca.md 公式 p·(gx^k−1)/(gx−1)，绝对累计自首仓价。
* 逐轮 `pnl` 一律是**净盈亏**（相对本轮开始时的现金，不含退回的本金）：
  `tp` / `gate_exit` = `qty·px·(1−FEE) − spent·lev`（平仓费已扣）；
  **期末未平 `eof` = `qty·c[-1] − spent·lev`（不扣平仓费 —— 没平就不该收费）**；
  `bust = f_round − w_open`（本轮现金全损，加回本轮已付资金费以避免与 `funding` 重复计）。
  复审 2026-10-30：门关平仓此前存 `spent + 净盈亏`，把本金当成盈利，8 次门平
  记成 +4490.68U（真实 −4590.35U），Σ明细与 ΔW 差 9329U。
  复审 2026-10-30 第二轮：爆仓轮原记 `−spent`，但爆仓把现金 `W` 直接清零 —— 亏掉的是
  **整份现金**而不只是已投保证金，账目差 `W_open − spent`（mock M4 实测 −600 vs −1000）。
* 回撤：`maxdd` 同时看收盘权益 `eq` 与 **bar 内最低价权益 `eq_low`**，取更差的那个。
* 余量 `worst` = min(low/bpx − 1)。**`bpx <= 0` 的 bar 数学上不可能强平**，不计入；
  若全程都 `bpx <= 0`，`worst` 返回 `inf`。打印一律用 `wstr()`，别直接 `f"{w*100:+.1f}%"`——
  那会打出 `+inf%`（2026-10-30 复审在 168 组扫描表里真实出现过两行）。

审计（C5）：opens / blocked_opens / adds / blocked_adds / gate_exits / funding / short_cash
全部计数并随结果返回。`short_cash` = 因现金不足而被截断的加仓次数（「梯子断了」，
满档 = 100% 权益时必现；交付组实测 0 次）。
`blocked_opens` 与 `blocked_adds` **都按「意图」计次**（连续关门只算 1 次），
关门/拦加仓的 bar 数分别存 `blocked_open_bars` / `bars_hold_off*`
（复审 2026-10-30 第三轮：`blocked_adds` 原按 bar 计，与 `blocked_opens` 单位不一致，
差分测试实测同段 361 vs 8）。原 `bars_off` 与 `blocked_open_bars` 恒等，已删除。
`rounds` = **已平轮的笔数** = tps + gate_exits + busts（复审 2026-10-30 第二轮：此前只在止盈时 +1，
把 1349 个止盈轮当成总轮数，漏了 8 个门平轮；期末未平的那一轮不算已平）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEE = 5e-4
MMR = 0.005


@dataclass(frozen=True)
class Params:
    step: float          # 首档间距
    gx: float            # 间距倍数
    amt: float           # 加仓量倍数
    tp: float            # 止盈（相对均价）
    n: int               # 加仓次数（不含首仓）
    lev: float
    base: float          # 首仓保证金
    intrabar: str = "opt"
    slip: float = 0.0    # 市价单滑点（开仓/加仓/门关平仓），一律不利方向
    compound: bool = False
    """复利：每轮开仓时把「满档」重算为**当前现金 W 的 cap 倍**。
    止盈/门平后本金+盈利全部回到 W，下一轮满档随之变大 —— 这才是「TP 成功算作
    保证金」的口径。`compound=False`（默认）是旧的固定 base，仅用于历史数字复现。"""
    cap: float = 1.0
    """复利时满档占当前权益的比例。1.0 = 满仓（资金费一扣就会让末档现金不足，
    `short_cash` 按 bar 反复计次）；<1 留缓冲。仅 compound=True 时生效。"""
    fb: float = 1.0
    """首仓 / 第一次加仓 的比例。1.0 = 1:1（历史默认）；2.0 = 首仓是加仓单的 2 倍。
    `amt` 只管第 2 次及以后的递增，两者独立。"""
    side: str = "long"
    """持仓方向。"long" = 向下加仓、TP 在均价上方；"short" = 向上加仓、TP 在均价下方。
    实现方式：所有价格公式统一乘 sgn（long=+1 / short=−1），见 simulate() 开头。"""
    sweep_tp: bool = False
    """止盈利润立刻抽到总账户，不再留在仓里当保证金。爆仓只清仓内现金，已抽走的利润保留。"""


def ladder(step, gx, amt, n, base, fb=1.0):
    """返回 (gaps, sizes)。gaps[m-1] = 第 m 档离首仓的累计跌幅；sizes[m-1] = 第 m 档保证金。

    首仓 = base，第 1 次加仓 = base/fb，之后每次乘 amt。fb=1.0 时等价于旧的 1:1 阶梯。
    """
    if abs(gx - 1.0) < 1e-12:
        gaps = [step * (m + 1) for m in range(n)]
    else:
        gaps = [step * (gx ** (m + 1) - 1.0) / (gx - 1.0) for m in range(n)]
    sizes = [base / fb * amt ** m for m in range(n)]
    return np.array(gaps, dtype=float), np.array(sizes, dtype=float)


def full_margin(amt, n, base, fb=1.0):
    """打满一轮所需保证金（首仓 + n 次加仓）。"""
    if abs(amt - 1.0) < 1e-12:
        return base * (1.0 + n / fb)
    return base * (1.0 + (amt ** n - 1.0) / (amt - 1.0) / fb)


def base_for(account, cap, amt, n, fb=1.0):
    """给定账户本金与「满档占本金比例」，反解首仓。"""
    if abs(amt - 1.0) < 1e-12:
        return cap * account / (1.0 + n / fb)
    return cap * account / (1.0 + (amt ** n - 1.0) / (amt - 1.0) / fb)


def wstr(w, width=7, prec=1):
    """余量格式化（唯一入口）。`worst=inf` 表示全程 bpx<=0、任何正价都无法强平，
    显示为「∞」；判断仍是 inf（语义正确），但**不允许**直接 `f"{w*100:+.1f}%"`。"""
    if not np.isfinite(w):
        return f"{'∞':>{width}}"
    return f"{w * 100:+{width}.{prec}f}%"


def simulate(o, h, l, c, fund, gate, p: Params, w0: float, detail: bool = False,
             trace: bool = False):
    """跑一遍。detail=True 时额外返回 `rounds_detail`（逐轮开/加/平的价格与盈亏），
    供 trade_log 直接消费 —— 不允许再出现第二套撮合循环（复审 P3）。"""
    gaps, _ = ladder(p.step, p.gx, p.amt, p.n, p.base, p.fb)
    sizes_ratio = np.array([(p.amt ** m) / p.fb for m in range(p.n)], dtype=float)
    sizes = sizes_ratio * p.base          # compound=True 时每轮开仓按新 base 重算
    sgn = 1.0 if p.side == "long" else -1.0   # 做空 = 全套公式乘 −1，价格比较随之翻转
    nl = p.n
    L = len(c)
    W = float(w0)
    W0 = W
    bank = 0.0                         # sweep_tp：已抽到总账户、爆仓动不到的利润
    qty = spent = p0 = 0.0
    nbuys = 0
    w_open = 0.0                       # 本轮开仓时的现金（爆仓全损的基准）
    last_base = p.base                 # 最近一轮的首仓保证金（复利时每轮不同）
    f_round = 0.0                      # 本轮已付的资金费（爆仓轮 pnl 要把它加回来，见下）
    stopped = False
    eq = np.empty(L)
    eq_low = np.empty(L)

    au = dict(opens=0, blocked_opens=0, adds=0, blocked_adds=0,
              gate_exits=0, tps=0, busts=0,
              bars_hold_off=0, bars_hold_off_with_levels=0, bars_hold_off_full=0,
              blocked_open_bars=0, funding=0.0, short_cash=0, exit_pnl=[])
    ev = []                                  # (bar, kind, 档数, 已投保证金)
    peak = W
    maxdd = 0.0
    worst, worst_i = np.inf, -1        # inf = 从未接近爆仓（bpx<=0，数学上不会爆）
    was_blocked = False                # 上一根 bar 是否处在「空仓且门关」状态
    was_add_blocked = False            # 上一根 bar 是否已在「价格到位但门拦加仓」状态
    rds = []                           # 逐轮明细（detail=True 时填）
    cur = None
    tr = []                            # 逐 bar 风险轨迹（trace=True 时填）：(i,W,qty,spent,bpx,low,档数)

    def mark(i, e, el):
        """记一根 bar 的权益（收盘 + bar 内最低价）并更新峰值与回撤。"""
        nonlocal peak, maxdd
        e, el = e + bank, el + bank
        eq[i], eq_low[i] = e, el
        if e > peak:
            peak = e
        if peak > 0.0:
            d = e / peak - 1.0
            if d < maxdd:
                maxdd = d
            dl = el / peak - 1.0          # bar 内最低价口径，回撤取更差的那个
            if dl < maxdd:
                maxdd = dl

    def close_round(i, kind, px, pnl):
        """收一轮：记录明细。pnl = 本轮净盈亏（不含退回的本金，相对本轮开始时的现金）。

        必须在 `qty/spent` 清零**之前**调用 —— `avg_px` 由引擎自己的 qty 反解，
        不给下游留第二套均价公式（复审 P3）。
        """
        nonlocal cur
        if not detail or cur is None:
            cur = None
            return
        rds.append({**cur, "close_i": i, "close_kind": kind, "close_px": float(px),
                    "pnl": float(pnl), "W": float(W), "nlegs": cur["nlegs"],
                    "avg_px": float(spent * p.lev * (1.0 - sgn * FEE) / qty)
                    if qty > 0.0 else float("nan")})
        cur = None

    def walk_down(i, ext):
        """沿不利方向逐档加仓；门关则本 bar 不再加。ext = 不利极值（多头 low / 空头 high）。"""
        nonlocal W, qty, spent, p0, nbuys, was_add_blocked
        while qty > 0.0 and nbuys - 1 < nl:
            m = nbuys                       # 第 m 档加仓（1-based）
            lim = p0 * (1.0 - sgn * gaps[m - 1])
            if sgn * (ext - lim) > 0.0:      # 多头要 ext<=lim，空头要 ext>=lim
                return
            if not gate.add[i]:             # ← 门的加仓决策点（A2）
                if not was_add_blocked:     # 同 blocked_opens：连续关门只算 1 次加仓意图
                    au["blocked_adds"] += 1
                was_add_blocked = True
                return
            sz = sizes[m - 1]
            if sz > W + 1e-12:              # 现金不够这一档：梯子断了（满档=100%权益时会命中）
                au["short_cash"] += 1
                return
            fill = lim * (1.0 + sgn * p.slip)     # 市价加仓，滑点不利
            W -= sz
            qty += sz * p.lev * (1.0 - sgn * FEE) / fill
            spent += sz
            nbuys += 1
            au["adds"] += 1
            was_add_blocked = False
            if detail and cur is not None:
                cur["legs"].append((i, float(fill), float(sz)))
                cur["nlegs"] = nbuys
            ev.append((i, "add", nbuys, spent))

    def check_bust(i, ext):
        nonlocal W, qty, spent, p0, nbuys, stopped, worst, worst_i, was_add_blocked
        if qty <= 0.0:
            return
        # 强平价：权益 = W + spent + sgn·(qty·p − spent·lev) = 0 解出 p。
        # long  → qty·p = spent·(lev−1) − W
        # short → qty·p = spent·(lev+1) + W   （空头亏损随价格上行，W 是同向加项）
        bpx = (spent * (p.lev - sgn) - sgn * W) / (qty * (1.0 - sgn * MMR))
        if trace:
            tr.append((i, float(W), float(qty), float(spent), float(bpx), float(ext), nbuys))
        if bpx <= 0.0:                       # 满档 ≤ 权益/lev：数学上永不爆仓
            return
        marg = (ext / bpx - 1.0) if sgn > 0 else (bpx / ext - 1.0)
        if marg < worst:
            worst, worst_i = marg, i
        busted = (ext <= bpx) if sgn > 0 else (ext >= bpx)
        if busted:
            au["busts"] += 1
            ev.append((i, "bust", nbuys, spent))
            W = 0.0                          # 全仓强平：现金清零（保守，实际会剩 MMR·名义）
            # 本轮现金从 w_open 直接归零 → 该轮对现金的贡献就是 −w_open（不含资金费）。
            # 资金费已单独记在 au["funding"] 里，故 pnl 要把本轮已付的 f_round 加回，
            # 才能让 Σpnl − funding == ΔW 在爆仓段也成立（实测曾差 512.27U）。
            close_round(i, "bust", bpx, f_round - w_open)
            qty = spent = p0 = 0.0
            nbuys = 0
            was_add_blocked = False
            stopped = True

    def open_round(i, px):
        """以 px 开新一轮首仓（Binance 的 Base Order）。

        金额固定 = `p.base`（`compound=False`）。Binance 官方 FAQ：Base Order Margin
        是「每轮开始时预留的初始保证金」，设置后不随上一轮盈利放大 —— 所以默认口径
        就是固定金额，`compound=True` 只是对照用的非实盘变体。

        返回 False 表示现金不足、本轮开不出来。
        """
        nonlocal W, qty, spent, p0, nbuys, w_open, f_round, last_base, cur, sizes
        b_open = base_for(W, p.cap, p.amt, p.n, p.fb) if p.compound else p.base
        if W < b_open - 1e-9:
            return False
        fill = px * (1.0 + sgn * p.slip)        # 市价开仓，滑点不利
        w_open = W
        f_round = 0.0
        last_base = b_open
        if p.compound:
            sizes = sizes_ratio * b_open
        W -= b_open
        qty = b_open * p.lev * (1.0 - sgn * FEE) / fill
        spent, p0, nbuys = b_open, fill, 1
        au["opens"] += 1
        if detail:
            cur = dict(open_i=i, open_px=float(fill),
                       legs=[(i, float(fill), float(b_open))], nlegs=1, mfd=0.0,
                       min_px=float(fill), need=0.0)
        ev.append((i, "open", 1, spent))
        return True

    def try_tp(i, fav, ext):
        """TP 打中即全平，并**在同一根 bar 内立刻开下一轮**。

        Binance 官方 FAQ（Start Condition = Instant）：TP 打中后立即开始下一轮，
        没有 bar 级延迟。新首仓成交价取 tpp（止盈限价触发价），滑点照旧不利。
        一根 bar 的 high 余量够时，可连续完成多轮（每多一轮需再涨一个 tp）。
        """
        nonlocal W, qty, spent, p0, nbuys, was_add_blocked, bank
        while qty > 0.0:
            tpp = spent * p.lev * (1.0 + sgn * p.tp) / (qty * (1.0 - sgn * FEE))
            hit = (fav >= tpp) if sgn > 0 else (fav <= tpp)
            if not hit:
                return
            pnl = sgn * (qty * tpp * (1.0 - sgn * FEE) - spent * p.lev)
            W += spent + pnl
            if p.sweep_tp:
                extra = W - W0
                if extra > 0.0:
                    bank += extra
                    W -= extra
            au["tps"] += 1
            ev.append((i, "tp", nbuys, spent))
            close_round(i, "tp", tpp, pnl)
            qty = spent = p0 = 0.0
            nbuys = 0
            was_add_blocked = False
            # ← 关键：同一根 bar 内立刻开下一轮（不等到下一根）
            if not gate.entry[i] or not open_round(i, tpp):
                return

    for i in range(L):
        if stopped:
            mark(i, W, W)
            continue
        fi = fund[i]
        if fi != 0.0 and qty > 0.0:
            fee = qty * o[i] * fi            # 按事件时刻（bar 开盘）计价，不用当根收盘
            W -= fee
            f_round += fee
            au["funding"] += fee

        # ① 门：已持仓的处置
        if qty > 0.0 and gate.exit_now[i]:
            nb = nbuys
            amt_in = spent
            _pnl = sgn * (qty * o[i] * (1.0 - sgn * p.slip) * (1.0 - sgn * FEE)
                          - spent * p.lev)                     # 净盈亏，不含本金
            W += spent + _pnl
            au["gate_exits"] += 1
            ev.append((i, "gate_exit", nb, amt_in))
            au["exit_pnl"].append(_pnl)
            close_round(i, "gate_exit", o[i] * (1.0 - sgn * p.slip), _pnl)
            qty = spent = p0 = 0.0
            nbuys = 0
            was_add_blocked = False

        # ② 门：开新一轮（空仓时）
        if qty <= 0.0:
            if gate.entry[i]:
                if not open_round(i, o[i]):        # 市价开仓在 bar 开盘价
                    mark(i, W, W)
                    was_blocked = False
                    continue
                was_blocked = False
                was_add_blocked = False
            else:
                au["blocked_open_bars"] += 1
                if not was_blocked:                   # 连续关门只算 1 次开仓意图
                    au["blocked_opens"] += 1
                was_blocked = True

        if qty > 0.0 and not gate.add[i]:
            au["bars_hold_off"] += 1
            if nbuys - 1 < nl:
                au["bars_hold_off_with_levels"] += 1
            else:
                au["bars_hold_off_full"] += 1

        # ext = 不利极值（多头 low / 空头 high），fav = 有利极值
        ext = l[i] if sgn > 0 else h[i]
        fav = h[i] if sgn > 0 else l[i]
        if p.intrabar == "pess":
            walk_down(i, ext)
            check_bust(i, ext)
            if not stopped:
                try_tp(i, fav, ext)
        else:
            try_tp(i, fav, ext)
            if qty > 0.0:
                walk_down(i, ext)
                check_bust(i, ext)

        if qty > 0.0:
            mark(i, W + spent + sgn * (qty * c[i] - spent * p.lev),
                 W + spent + sgn * (qty * ext - spent * p.lev))
            if detail and cur is not None:
                # MFD：本轮持仓期间最深浮亏 / 本轮已投保证金（3Commas 口径）。
                # 分母是 spent 不是 spent·lev —— 杠杆亏损要体现在比率里。
                fl = sgn * (qty * ext - spent * p.lev) / spent
                if fl < cur["mfd"]:
                    cur["mfd"] = float(fl)
                # need：本轮最苛刻时刻，从当根最低价回到 TP 价还需要多少反弹（含手续费）。
                # 这是判断「普通反弹能否 TP 出去」的直接指标。
                worse = (ext < cur["min_px"]) if sgn > 0 else (ext > cur["min_px"])
                if worse:
                    cur["min_px"] = float(ext)
                tp_now = spent * p.lev * (1.0 + sgn * p.tp) / (qty * (1.0 - sgn * FEE))
                nd = (tp_now / ext - 1.0) if sgn > 0 else (1.0 - tp_now / ext)
                if nd > cur["need"]:
                    cur["need"] = float(nd)
        else:
            mark(i, W, W)

    cover = float(gaps[-1]) if len(gaps) else 0.0
    if detail and cur is not None and qty > 0.0:          # 期末未平：按收盘价 mark-to-market
        fpnl = sgn * (qty * c[-1] - spent * p.lev)         # 没平仓就不该计平仓手续费
        rds.append({**cur, "close_i": L - 1, "close_kind": "eof", "close_px": float(c[-1]),
                    "pnl": float(fpnl),
                    "W": float(W + spent + fpnl), "nlegs": nbuys, "float": float(fpnl),
                    "avg_px": float(spent * p.lev * (1.0 - sgn * FEE) / qty)})
    return dict(ret=eq[-1] / W0 - 1.0, W=float(eq[-1]), W0=W0, bank=float(bank),
                eq=eq, eq_low=eq_low, maxdd=maxdd,
                cover=cover, worst=worst, worst_i=worst_i,
                full_margin=full_margin(p.amt, p.n, last_base, p.fb),
                rounds=au["tps"] + au["gate_exits"] + au["busts"],
                ev=ev, rounds_detail=rds, trace=tr, **au)
