#!/usr/bin/env python3
"""Report whether the locked BTC wash21 gate currently permits a long DCA cycle.

The calculation is pure once the completed daily and weekly candles are fixed.
Use ``--source npy --as-of ...`` for a reproducible historical replay, or use
the default Binance source for the latest completed UTC daily candle.
"""

import argparse
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


API = "https://data-api.binance.vision/api/v3/klines"
SYMBOL = "BTCUSDT"
DAY_MS = 86_400_000
FIVE_MINUTES_MS = 300_000
HISTORY_START_MS = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)
DEFAULT_NPY = Path("backtest/data/dca_indicator_study/5m/BTCUSDT_5m.npy")

ST_LENGTH = 6
ST_MULTIPLIER = 2.5
FLAT_WEEKS = 8
FLAT_ATR_MULTIPLIER = 0.5
STALL_WEEKS = 4
DAILY_RANGE_DAYS = 28
DAILY_RANGE_ATR_MULTIPLIER = 1.5
DAILY_ATR_LENGTH = 14
DAILY_ATR_CONTRACTION = 0.8
STABLE_DAYS = 21

DCA_PARAMETERS = {
    "step_pct": 0.75,
    "spacing_multiplier": 1.12,
    "amount_multiplier": 1.30,
    "take_profit_pct": 2.0,
    "max_dca_orders": 13,
    "base_order_usdt": 10.0,
    "dca_order_usdt": 10.0,
    "planned_notional_usdt": 986.2503553075103,
    "coverage_pct": 21.021831947828847,
}


def iso_utc(timestamp_ms):
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def parse_as_of(value):
    if value:
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")
        return int(parsed.value // 1_000_000)
    now_ms = int(time.time() * 1000)
    return now_ms - now_ms % DAY_MS


def canonical_json(value, *, pretty=True):
    separators = None if pretty else (",", ":")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=separators,
        allow_nan=False,
    )


def _request_json(params, attempts=5):
    url = API + "?" + urllib.parse.urlencode(params)
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "btc-wash21-signal/1"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Binance kline request failed: {last_error}") from last_error


def fetch_completed_klines(interval, as_of_ms):
    rows = []
    cursor = HISTORY_START_MS
    while cursor < as_of_ms:
        page = _request_json({
            "symbol": SYMBOL,
            "interval": interval,
            "startTime": cursor,
            "endTime": as_of_ms - 1,
            "limit": 1000,
        })
        if not page:
            break
        rows.extend(row for row in page if int(row[6]) < as_of_ms)
        next_cursor = int(page[-1][6]) + 1
        if next_cursor <= cursor or len(page) < 1000:
            break
        cursor = next_cursor
    deduplicated = {int(row[0]): row for row in rows}
    return [deduplicated[key] for key in sorted(deduplicated)]


def frame_from_binance(rows):
    if not rows:
        raise ValueError("no completed Binance candles were returned")
    frame = pd.DataFrame({
        "open": [float(row[1]) for row in rows],
        "high": [float(row[2]) for row in rows],
        "low": [float(row[3]) for row in rows],
        "close": [float(row[4]) for row in rows],
        "volume": [float(row[5]) for row in rows],
    }, index=pd.to_datetime([int(row[6]) for row in rows], unit="ms", utc=True))
    frame.index.name = "close_time"
    return frame


def resample_completed(values, rule, period, as_of_ms):
    columns = ["timestamp", "open", "high", "low", "close"]
    frame = pd.DataFrame(np.asarray(values[:, :5]), columns=columns)
    frame.index = pd.to_datetime(frame.pop("timestamp"), unit="ms", utc=True)
    result = frame.resample(rule, closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()
    result.index = result.index + period - pd.Timedelta(milliseconds=1)
    cutoff = pd.Timestamp(as_of_ms, unit="ms", tz="UTC")
    return result.loc[result.index < cutoff]


def frames_from_npy(path, as_of_ms):
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2 or values.shape[1] < 5:
        raise ValueError("npy input must contain timestamp/open/high/low/close columns")
    timestamps = values[:, 0].astype(np.int64)
    stop = int(np.searchsorted(timestamps + FIVE_MINUTES_MS, as_of_ms, side="right"))
    selected = np.asarray(values[:stop, :5], dtype=np.float64)
    if len(selected) < 2 or np.any(np.diff(selected[:, 0]) <= 0):
        raise ValueError("npy input is empty or timestamps are not strictly increasing")
    daily = resample_completed(selected, "1D", pd.Timedelta(days=1), as_of_ms)
    weekly = resample_completed(selected, "W-MON", pd.Timedelta(days=7), as_of_ms)
    digest = hashlib.sha256(selected.tobytes(order="C")).hexdigest()
    return daily, weekly, {
        "kind": "npy",
        "path": str(Path(path).resolve()),
        "bars_used": len(selected),
        "sha256_used_rows": digest,
    }


def frames_from_binance(as_of_ms):
    daily_rows = fetch_completed_klines("1d", as_of_ms)
    weekly_rows = fetch_completed_klines("1w", as_of_ms)
    fingerprint_rows = [
        [int(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]), int(row[6])]
        for row in daily_rows + weekly_rows
    ]
    digest = hashlib.sha256(
        canonical_json(fingerprint_rows, pretty=False).encode("utf-8")
    ).hexdigest()
    return frame_from_binance(daily_rows), frame_from_binance(weekly_rows), {
        "kind": "binance_spot",
        "endpoint": API,
        "daily_candles": len(daily_rows),
        "weekly_candles": len(weekly_rows),
        "sha256_completed_candles": digest,
    }


def rma(series, length):
    values = series.to_numpy(dtype=np.float64)
    output = np.full(len(values), np.nan)
    seed_index = None
    for index in range(length - 1, len(values)):
        window = values[index - length + 1:index + 1]
        if np.all(np.isfinite(window)):
            seed_index = index
            output[index] = float(np.mean(window))
            break
    if seed_index is not None:
        for index in range(seed_index + 1, len(values)):
            output[index] = (
                (output[index - 1] * (length - 1) + values[index]) / length
                if np.isfinite(values[index]) else output[index - 1]
            )
    return pd.Series(output, index=series.index)


def true_range(frame):
    previous = frame["close"].shift(1)
    return pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - previous).abs(),
        (frame["low"] - previous).abs(),
    ], axis=1).max(axis=1)


def supertrend_trace(frame, length=ST_LENGTH, multiplier=ST_MULTIPLIER):
    atr = rma(true_range(frame), length).to_numpy()
    high, low, close = (frame[column].to_numpy() for column in ("high", "low", "close"))
    output = np.full((len(frame), 4), np.nan)
    upper = lower = 0.0
    trend = -1
    started = False
    for index in range(len(frame)):
        if not np.isfinite(atr[index]):
            continue
        middle = (high[index] + low[index]) / 2
        basic_lower = middle - multiplier * atr[index]
        basic_upper = middle + multiplier * atr[index]
        if not started:
            lower, upper, started = basic_lower, basic_upper, True
            trend = 1 if close[index] > upper else -1
        else:
            lower = max(basic_lower, lower) if close[index - 1] > lower else basic_lower
            upper = min(basic_upper, upper) if close[index - 1] < upper else basic_upper
            if close[index] > upper:
                trend = 1
            elif close[index] < lower:
                trend = -1
        output[index] = trend, lower if trend > 0 else upper, lower, upper
    result = pd.DataFrame(
        output, index=frame.index, columns=["trend", "line", "lower", "upper"]
    )
    result["atr"] = atr
    return result


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _setup_conditions(daily, trace, daily_atr, day_index, week_index):
    unavailable = {
        "flat_8_weeks": False,
        "four_weeks_down": False,
        "upper_stall": False,
        "daily_range_contracted": False,
        "daily_atr_contracted": False,
        "setup_ready": False,
    }
    if week_index < FLAT_WEEKS or day_index < DAILY_RANGE_DAYS:
        return unavailable
    current_atr = trace["atr"].iloc[week_index]
    reference_atr = trace["atr"].iloc[week_index - STALL_WEEKS]
    current_daily_atr = daily_atr.iloc[day_index]
    previous_daily_atr = daily_atr.iloc[day_index - DAILY_RANGE_DAYS]
    required = (current_atr, reference_atr, current_daily_atr, previous_daily_atr)
    if not all(_finite(value) for value in required):
        return unavailable

    upper_window = trace["upper"].iloc[week_index - FLAT_WEEKS + 1:week_index + 1]
    flat = float(upper_window.max() - upper_window.min()) <= (
        FLAT_ATR_MULTIPLIER * float(current_atr)
    )
    four_down = bool(np.all(
        trace["trend"].iloc[week_index - STALL_WEEKS + 1:week_index + 1].to_numpy() < 0
    ))
    upper_stall = (
        float(trace["upper"].iloc[week_index - STALL_WEEKS])
        - float(trace["upper"].iloc[week_index])
        <= FLAT_ATR_MULTIPLIER * float(reference_atr)
    )
    day_window = daily.iloc[day_index - DAILY_RANGE_DAYS + 1:day_index + 1]
    daily_range = float(day_window["high"].max() - day_window["low"].min())
    range_contracted = daily_range <= DAILY_RANGE_ATR_MULTIPLIER * float(reference_atr)
    atr_contracted = float(current_daily_atr) <= (
        DAILY_ATR_CONTRACTION * float(previous_daily_atr)
    )
    return {
        "flat_8_weeks": bool(flat),
        "four_weeks_down": four_down,
        "upper_stall": bool(upper_stall),
        "daily_range_contracted": bool(range_contracted),
        "daily_atr_contracted": bool(atr_contracted),
        "setup_ready": bool(flat and four_down and upper_stall and range_contracted and atr_contracted),
    }


def evaluate_gate(daily, weekly):
    if len(daily) < 60 or len(weekly) < 20:
        raise ValueError("at least 60 completed daily and 20 completed weekly candles are required")
    trace = supertrend_trace(weekly)
    daily_atr = rma(true_range(daily), DAILY_ATR_LENGTH)
    weekly_times = weekly.index.as_unit("ms").asi8
    daily_times = daily.index.as_unit("ms").asi8

    active = False
    armed = False
    waiting = False
    wash_low = None
    wash_low_time = None
    stable_days = 0
    armed_at = None
    events = []
    latest_conditions = None
    latest_week_index = None

    for day_index, day_close_ms in enumerate(daily_times):
        week_index = int(np.searchsorted(weekly_times, day_close_ms, side="right") - 1)
        if week_index < 0 or not _finite(trace["trend"].iloc[week_index]):
            continue
        latest_week_index = week_index
        trend_down = bool(trace["trend"].iloc[week_index] < 0)
        flipped_down = bool(
            week_index > 0
            and trace["trend"].iloc[week_index - 1] > 0
            and trend_down
        )
        if active and flipped_down:
            active = False
            armed = False
            waiting = False
            wash_low = None
            wash_low_time = None
            stable_days = 0
            armed_at = None
            events.append({
                "type": "disable",
                "effective_at": iso_utc(int(day_close_ms) + 1),
                "reason": "weekly_st_up_to_down",
            })

        latest_conditions = _setup_conditions(
            daily, trace, daily_atr, day_index, week_index
        )
        if active:
            continue
        if not trend_down:
            armed = False
            waiting = False
            wash_low = None
            wash_low_time = None
            stable_days = 0
            armed_at = None
            continue

        if latest_conditions["setup_ready"]:
            if not armed:
                armed_at = int(day_close_ms) + 1
            armed = True

        current_low = float(daily["low"].iloc[day_index])
        if waiting:
            if current_low < wash_low:
                wash_low = current_low
                wash_low_time = int(day_close_ms)
                stable_days = 0
            else:
                stable_days += 1
        elif armed and day_index >= DAILY_RANGE_DAYS:
            prior_low = float(
                daily["low"].iloc[day_index - DAILY_RANGE_DAYS:day_index].min()
            )
            if current_low <= prior_low:
                waiting = True
                wash_low = current_low
                wash_low_time = int(day_close_ms)
                stable_days = 0

        if waiting and stable_days >= STABLE_DAYS:
            active = True
            armed = False
            waiting = False
            effective_at = int(day_close_ms) + 1
            events.append({
                "type": "enable",
                "effective_at": iso_utc(effective_at),
                "wash_low": wash_low,
                "wash_low_date": iso_utc(wash_low_time),
                "stable_days": stable_days,
            })
            wash_low = None
            wash_low_time = None
            stable_days = 0
            armed_at = None

    if latest_week_index is None:
        raise ValueError("weekly Supertrend did not have enough history")
    weekly_row = trace.iloc[latest_week_index]
    blocking = []
    if not active:
        if weekly_row["trend"] >= 0:
            blocking.append("weekly_st_is_not_down_for_a_new_setup")
        if not armed and not waiting:
            blocking.append("setup_is_not_armed")
        elif armed and not waiting:
            blocking.append("waiting_for_a_new_28_day_low")
        elif waiting:
            blocking.append(f"only_{stable_days}_of_{STABLE_DAYS}_stable_days_completed")

    last_enable = next((event for event in reversed(events) if event["type"] == "enable"), None)
    last_disable = next((event for event in reversed(events) if event["type"] == "disable"), None)
    return {
        "can_start_dca": active,
        "decision": "OPEN_DCA_ALLOWED" if active else "DO_NOT_OPEN_DCA",
        "reason": (
            "enabled_until_the_next_completed_weekly_st_up_to_down_transition"
            if active else "entry_state_machine_has_not_completed"
        ),
        "blocking_conditions": blocking,
        "state": {
            "setup_armed": armed,
            "setup_armed_at": iso_utc(armed_at) if armed_at is not None else None,
            "waiting_after_wash_low": waiting,
            "wash_low": wash_low,
            "wash_low_date": iso_utc(wash_low_time) if wash_low_time is not None else None,
            "stable_days_completed": stable_days,
        },
        "latest_completed_candles": {
            "daily_close_time": iso_utc(int(daily_times[-1])),
            "weekly_close_time": iso_utc(int(weekly_times[latest_week_index])),
        },
        "weekly_st": {
            "length": ST_LENGTH,
            "multiplier": ST_MULTIPLIER,
            "direction": "up" if weekly_row["trend"] > 0 else "down",
            "line": float(weekly_row["line"]),
            "atr": float(weekly_row["atr"]),
        },
        "latest_setup_conditions": latest_conditions,
        "last_enable_event": last_enable,
        "last_disable_event": last_disable,
        "events": events,
    }


def build_output(daily, weekly, source, as_of_ms, include_history):
    gate = evaluate_gate(daily, weekly)
    result = {
        "schema_version": 1,
        "strategy": "btc_wash21_long_dca_gate",
        "symbol": SYMBOL,
        "as_of": iso_utc(as_of_ms),
        "can_start_dca": gate["can_start_dca"],
        "decision": gate["decision"],
        "reason": gate["reason"],
        "blocking_conditions": gate["blocking_conditions"],
        "state": gate["state"],
        "latest_completed_candles": gate["latest_completed_candles"],
        "weekly_st": gate["weekly_st"],
        "latest_setup_conditions": gate["latest_setup_conditions"],
        "last_enable_event": gate["last_enable_event"],
        "last_disable_event": gate["last_disable_event"],
        "dca_parameters": DCA_PARAMETERS,
        "source": source,
    }
    if include_history:
        result["events"] = gate["events"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("binance", "npy"), default="binance")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_NPY)
    parser.add_argument(
        "--as-of",
        help="UTC cutoff; only candles closed before this instant are used",
    )
    parser.add_argument("--history", action="store_true", help="include all enable/disable events")
    parser.add_argument(
        "--verify-idempotence",
        action="store_true",
        help="calculate twice and fail unless canonical outputs are identical",
    )
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    as_of_ms = parse_as_of(args.as_of)
    if args.source == "npy":
        daily, weekly, source = frames_from_npy(args.data_file, as_of_ms)
    else:
        daily, weekly, source = frames_from_binance(as_of_ms)
    output = build_output(daily, weekly, source, as_of_ms, args.history)
    if args.verify_idempotence:
        repeated = build_output(daily.copy(), weekly.copy(), dict(source), as_of_ms, args.history)
        if canonical_json(output, pretty=False) != canonical_json(repeated, pretty=False):
            raise AssertionError("same completed candles produced different output")
        output["idempotence_verified"] = True
    print(canonical_json(output, pretty=not args.compact))


if __name__ == "__main__":
    main()
