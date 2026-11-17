package main

import (
	"math"
	"testing"

	"crypto_research/st_boll/types"
)

func testMatrixIndicators(close float64) []indicatorBar {
	return []indicatorBar{
		{close: 100, mid: 100, upper: 110, lower: 90, st: 95, atr: 3, dir: 1, valid: true},
		{close: close, mid: 100, upper: 110, lower: 90, st: 95, atr: 3, dir: 1, valid: true},
	}
}

func TestMatrixStateTracksNearestSupportAndResistance(t *testing.T) {
	ind := testMatrixIndicators(96)
	state := matrixStateAt(ind, 1, config{MatrixTouchFrac: 0.35})
	if state.supportName != "st" || state.resistanceName != "mid" {
		t.Fatalf("support/resistance = %s/%s, want st/mid", state.supportName, state.resistanceName)
	}
	if state.zone != "lower_mid" {
		t.Fatalf("zone = %q, want lower_mid", state.zone)
	}
}

func TestMatrixSignalKeepsDirectionsSeparatedByMiddleBand(t *testing.T) {
	ind := testMatrixIndicators(96)
	longCandle := types.Candle{Low: 94, High: 98, Close: 96}
	prevLong := types.Candle{Low: 101, High: 103, Close: 102}
	if dir, _ := matrixSignal(longCandle, prevLong, ind, 1, "touch", config{MatrixTouchFrac: 0.35}); dir != 1 {
		t.Fatalf("long touch direction = %d, want 1", dir)
	}

	ind[1].close = 104
	shortCandle := types.Candle{Low: 102, High: 111, Close: 104}
	prevShort := types.Candle{Low: 98, High: 99, Close: 98}
	if dir, _ := matrixSignal(shortCandle, prevShort, ind, 1, "touch", config{MatrixTouchFrac: 0.35}); dir != -1 {
		t.Fatalf("short touch direction = %d, want -1", dir)
	}

	ind[1].close = 100
	ind[0].close = 101
	middleCandle := types.Candle{Low: 89, High: 111, Close: 100}
	if dir, _ := matrixSignal(middleCandle, types.Candle{Low: 101, High: 109, Close: 101}, ind, 1, "touch", config{MatrixTouchFrac: 0.35}); dir != 0 {
		t.Fatalf("middle touch direction = %d, want 0", dir)
	}
}

func TestMatrixExitPrioritizesStopOverTarget(t *testing.T) {
	b := indicatorBar{mid: 100, upper: 110, lower: 90, st: 95, atr: 4, valid: true}
	c := types.Candle{Low: 88, High: 112, Close: 100}
	ev := matrixExitForBar(1, c, b, 96, "outer", false, 1, 2, 18, config{})
	if !ev.trigger || !ev.stop || ev.reason != "sl" {
		t.Fatalf("exit = %+v, want stop", ev)
	}
}

func TestPhaseT2ManagedExitPartialThenRunner(t *testing.T) {
	pos := phaseT2Position{dir: 1, entry: 100, stop: 90, target: 105, residual: 1}
	b := indicatorBar{st: 101, upper: 115, lower: 90, atr: 4, valid: true}
	if reason, trigger, partial := phaseT2ManagedExit(pos, types.Candle{Low: 100, High: 106, Close: 104}, b, 2, 24, config{T2Exit: "hybrid"}); !trigger || !partial || reason != "tp_mid_partial" {
		t.Fatalf("initial exit = %q/%t/%t, want mid partial", reason, trigger, partial)
	}
	pos.armed = true
	pos.residual = 0.5
	if reason, trigger, partial := phaseT2ManagedExit(pos, types.Candle{Low: 103, High: 116, Close: 114}, b, 3, 24, config{T2Exit: "hybrid"}); !trigger || partial || reason != "tp_outer" {
		t.Fatalf("runner exit = %q/%t/%t, want outer exit", reason, trigger, partial)
	}
}

func TestPhaseT2AllocationHonorsRiskCap(t *testing.T) {
	got := phaseT2Allocation(100, 10, config{T2RiskPct: 0.005, T2MaxAlloc: 0.25})
	if math.Abs(got-0.05) > 1e-9 {
		t.Fatalf("allocation = %.4f, want 0.05", got)
	}
	got = phaseT2Allocation(100, 0.1, config{T2RiskPct: 0.005, T2MaxAlloc: 0.25})
	if got != 0.25 {
		t.Fatalf("allocation cap = %.4f, want 0.25", got)
	}
}

func TestSymbolMaxDDSortsByExitTime(t *testing.T) {
	ts := []trade{
		{entryTime: 20, exitTime: 30, net: -0.10},
		{entryTime: 10, exitTime: 20, net: 0.10},
	}
	if got := symbolMaxDD(ts); math.Abs(got-0.10) > 1e-9 {
		t.Fatalf("max DD = %.4f, want 0.10 after chronological ordering", got)
	}
}
