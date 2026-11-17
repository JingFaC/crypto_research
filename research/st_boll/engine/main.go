package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"crypto_research/st_boll/types"
)

// st_boll_ema is an independent ST/Bollinger research strategy. The phase_t2
// mode treats the nearest Bollinger/ST line as support or resistance, enters
// after a closed-bar reaction, takes half at the middle band, and trails the
// remainder toward the opposite outer band. Signals use closed bars and
// next-open fills; legacy modes remain available for comparison.
type config struct {
	STPeriod        int
	STFactor        float64
	BBPeriod        int
	BBK             float64
	EMAPeriod       int
	MaxHold         int
	FeeBps          float64
	SlipBps         float64
	EntryMode       string
	SqueezeLB       int
	SqueezeW        int
	CrossW          int
	ExitMode        string
	EMASlope        bool
	RunnerExit      string
	RunnerStopPct   float64
	RunnerTrailPct  float64
	TWeight         float64
	TGap            int
	THold           int
	TStopPct        float64
	TTargetFrac     float64
	TMinMovePct     float64
	PhaseLB         int
	RunnerSetup     string
	TStopMode       string
	TSide           string
	TRegime         string
	TAlloc          float64
	TIndependent    bool
	TPhaseLB        int
	TPhaseMin       int
	T2RiskPct       float64
	T2MaxAlloc      float64
	T2StopATR       float64
	T2TargetATR     float64
	T2StopOnClose   bool
	T2Gap           int
	T2Hold          int
	T2Family        string
	T2Exit          string
	MatrixTouchFrac float64
	MatrixATRStop   float64
	MatrixATRTake   float64
	MatrixFilter    string
	MatrixReport    string
	MatrixReverse   bool
	MarketRegime    map[int64]int
}

type indicatorBar struct {
	close, ema, mid, upper, lower, st, atr float64
	dir                                    int
	squeezeAge                             int
	crossLongAge, crossShortAge            int
	lowerOuterAge, upperOuterAge           int
	midUpAge, midDownAge                   int
	valid                                  bool
}

type trade struct {
	sym       string
	dir       int
	entryTime int64
	exitTime  int64
	entry     float64
	exit      float64
	net       float64
	hold      int
	reason    string
	tCycles   int
	kind      string
	family    string
	exitModel string
	state     string
}

type symbolResult struct {
	sym    string
	bars   int
	trades []trade
	maxDD  float64
}

type hybridPosition struct {
	dir       int
	entryIdx  int
	entryTime int64
	entry     float64
	tOpen     bool
	rOpen     bool
	tEntry    float64
	tEntryIdx int
	tCycles   int
	tNet      float64
	rNet      float64
	tHold     int
	rHold     int
	tReason   string
	rReason   string
	exitTime  int64
}

type summary struct {
	name       string
	symbols    int
	bars       int
	trades     []trade
	perSymbol  []symbolResult
	monthlyNet map[string]float64
}

func main() {
	dataDir := flag.String("data", "data/k63_te", "directory containing *_4h.json")
	interval := flag.String("interval", "4h", "candle interval")
	feeBps := flag.Float64("fee_bps", 10, "one-way fee in basis points")
	slipBps := flag.Float64("slip_bps", 5, "one-way slippage in basis points")
	maxHold := flag.Int("max_hold", 18, "maximum holding bars")
	entryMode := flag.String("entry", "state", "entry mode: phase_ttrend|phase_t2|hfregime|state|t_matrix|stboll|stboll_t|hybrid|cross_t|breakout|touch")
	squeezeLB := flag.Int("squeeze_lb", 300, "Bollinger bandwidth squeeze lookback")
	squeezeWindow := flag.Int("squeeze_window", 50, "bars allowed after a squeeze")
	crossWindow := flag.Int("cross_window", 6, "bars allowed after an ST/Bollinger cross")
	exitMode := flag.String("exit", "hybrid", "exit mode: hybrid|mid_st|trend|trend_st|trend_ema|opposite")
	runnerExit := flag.String("runner_exit", "st", "hybrid runner exit: st|st_ema")
	runnerStopPct := flag.Float64("runner_stop_pct", 0, "phase trend runner hard stop as adverse price percentage; 0 disables")
	runnerTrailPct := flag.Float64("runner_trail_pct", 0, "phase trend runner trailing stop from best close percentage; 0 disables")
	tWeight := flag.Float64("t_weight", 0.80, "fraction allocated to the Bollinger T leg; remainder follows the confirmed breakout runner")
	tGap := flag.Int("t_gap", 3, "minimum bars between high-frequency T entries")
	tHold := flag.Int("t_hold", 18, "maximum bars for each high-frequency T leg")
	tStopPct := flag.Float64("t_stop_pct", 0, "phase T hard stop as adverse price percentage; 0 uses Bollinger outer stop")
	tTargetFrac := flag.Float64("t_target_frac", 1, "phase T target fraction of the signal-to-middle move")
	tMinMovePct := flag.Float64("t_min_move_pct", 0, "minimum signal-to-middle distance percentage for a phase T entry")
	phaseLB := flag.Int("phase_lb", 42, "lookback bars for the phase regime filter")
	phaseMarket := flag.String("phase_market", "local", "phase source for phase_ttrend: local|btc")
	runnerSetup := flag.String("runner_setup", "band", "phase trend retest setup: band|mid")
	tStopMode := flag.String("t_stop_mode", "outer", "phase T stop line: outer|st")
	tSide := flag.String("t_side", "both", "T direction filter: both|long|short")
	tRegime := flag.String("t_regime", "up", "T phase filter: both|trend|up|down")
	tAlloc := flag.Float64("t_alloc", 0.75, "state/phase T allocation fraction; 1 preserves the baseline")
	tIndependent := flag.Bool("t_independent", false, "allow a new T leg while a zero-capital trend gate is waiting")
	tPhaseLB := flag.Int("t_phase_lb", 10, "lookback bars for the state phase gate")
	tPhaseMin := flag.Int("t_phase_min", 6, "minimum closes on the phase side")
	t2RiskPct := flag.Float64("t2_risk_pct", 0.0025, "phase_t2 maximum equity risk per trade")
	t2MaxAlloc := flag.Float64("t2_max_alloc", 0.10, "phase_t2 maximum allocation per trade")
	t2StopATR := flag.Float64("t2_stop_atr", 2.0, "phase_t2 fixed stop distance in ATR")
	t2TargetATR := flag.Float64("t2_target_atr", 0, "phase_t2 fixed target distance in ATR; 0 uses Bollinger middle")
	t2StopOnClose := flag.Bool("t2_stop_on_close", true, "phase_t2 trigger fixed stop only on candle close")
	t2Gap := flag.Int("t2_gap", 0, "phase_t2 minimum bars between entries")
	t2Hold := flag.Int("t2_hold", 24, "phase_t2 maximum holding bars")
	t2Family := flag.String("t2_family", "hybrid", "phase_t2 signal family: selected|paired|state|hybrid|reclaim")
	t2Exit := flag.String("t2_exit", "hybrid", "phase_t2 exit: fixed|mid|outer|hybrid")
	matrixTouchFrac := flag.Float64("matrix_touch_frac", 0.35, "matrix state distance as a fraction of Bollinger width")
	matrixATRStop := flag.Float64("matrix_atr_stop", 1.0, "matrix ATR stop multiple")
	matrixATRTake := flag.Float64("matrix_atr_take", 1.5, "matrix ATR target multiple")
	matrixFilter := flag.String("matrix_filter", "", "matrix state filter: robust")
	matrixReport := flag.String("matrix_report", "", "optional CSV path for t_matrix per-symbol and per-state results")
	matrixReverse := flag.Bool("matrix_reverse", true, "t_matrix: close and reverse on an opposite confirmed signal")
	emaSlope := flag.Bool("ema_slope", true, "require EMA rising for long and falling for short")
	minBars := flag.Int("min_bars", 250, "minimum bars per symbol")
	start := flag.String("start", "", "optional UTC date, YYYY-MM-DD")
	end := flag.String("end", "", "optional UTC date, YYYY-MM-DD")
	analyzeCross := flag.Bool("analyze_cross", false, "analyze ST/Bollinger cross and support reactions")
	analyzeHF := flag.Bool("analyze_hf", false, "analyze high-frequency ST/Bollinger reaction filters")
	analyzeState := flag.Bool("analyze_state", false, "analyze ST/Bollinger state combinations")
	symbolsFilter := flag.String("symbols", "", "comma-separated symbols for read-only analysis, e.g. BTCUSDT,ETHUSDT")
	includeBTC := flag.Bool("include_btc", false, "include BTCUSDT in analysis or backtest")
	flag.Parse()

	files, err := filepath.Glob(filepath.Join(*dataDir, "*_"+*interval+".json"))
	if err != nil || len(files) == 0 {
		fatal("no candle files in %s", *dataDir)
	}
	sort.Strings(files)
	var startMs, endMs int64
	if *start != "" {
		startMs = parseDate(*start)
	}
	if *end != "" {
		endMs = parseDate(*end) + 24*60*60*1000 - 1
	}
	cfg := config{
		STPeriod:        10,
		STFactor:        3,
		BBPeriod:        20,
		BBK:             2,
		EMAPeriod:       63,
		MaxHold:         *maxHold,
		FeeBps:          *feeBps,
		SlipBps:         *slipBps,
		EntryMode:       *entryMode,
		SqueezeLB:       *squeezeLB,
		SqueezeW:        *squeezeWindow,
		CrossW:          *crossWindow,
		ExitMode:        *exitMode,
		EMASlope:        *emaSlope,
		RunnerExit:      *runnerExit,
		RunnerStopPct:   *runnerStopPct,
		RunnerTrailPct:  *runnerTrailPct,
		TWeight:         *tWeight,
		TGap:            *tGap,
		THold:           *tHold,
		TStopPct:        *tStopPct,
		TTargetFrac:     *tTargetFrac,
		TMinMovePct:     *tMinMovePct,
		PhaseLB:         *phaseLB,
		RunnerSetup:     *runnerSetup,
		TStopMode:       *tStopMode,
		TSide:           *tSide,
		TRegime:         *tRegime,
		TAlloc:          *tAlloc,
		TIndependent:    *tIndependent,
		TPhaseLB:        *tPhaseLB,
		TPhaseMin:       *tPhaseMin,
		T2RiskPct:       *t2RiskPct,
		T2MaxAlloc:      *t2MaxAlloc,
		T2StopATR:       *t2StopATR,
		T2TargetATR:     *t2TargetATR,
		T2StopOnClose:   *t2StopOnClose,
		T2Gap:           *t2Gap,
		T2Hold:          *t2Hold,
		T2Family:        *t2Family,
		T2Exit:          *t2Exit,
		MatrixTouchFrac: *matrixTouchFrac,
		MatrixATRStop:   *matrixATRStop,
		MatrixATRTake:   *matrixATRTake,
		MatrixFilter:    *matrixFilter,
		MatrixReport:    *matrixReport,
		MatrixReverse:   *matrixReverse,
	}
	if *analyzeCross {
		analyzeCrossEvents(files, *dataDir, *interval, startMs, endMs, *minBars, cfg, *symbolsFilter, *includeBTC)
		return
	}
	if *analyzeHF {
		analyzeHighFrequencyFilters(files, *dataDir, *interval, startMs, endMs, *minBars, cfg)
		return
	}
	if *analyzeState {
		analyzeStateCombinations(files, *dataDir, *interval, startMs, endMs, *minBars, cfg, *symbolsFilter, *includeBTC)
		return
	}
	if cfg.EntryMode == "t_matrix" {
		runTMatrix(files, *dataDir, *interval, startMs, endMs, *minBars, cfg, *symbolsFilter, *includeBTC)
		return
	}
	if cfg.EntryMode == "phase_ttrend" && strings.EqualFold(*phaseMarket, "btc") {
		for _, f := range files {
			sym := strings.TrimSuffix(filepath.Base(f), "_"+*interval+".json")
			if strings.EqualFold(sym, "BTCUSDT") {
				if btc, loadErr := loadCandles(f); loadErr == nil {
					cfg.MarketRegime = buildMarketRegime(btc, cfg, cfg.PhaseLB)
				}
				break
			}
		}
	}

	var all summary
	if cfg.EntryMode == "stboll" || cfg.EntryMode == "hfregime" || cfg.EntryMode == "state" || cfg.EntryMode == "phase_ttrend" || cfg.EntryMode == "phase_t2" {
		all.name = fmt.Sprintf("ST(%d,%.1f)+Boll(%d,%.1f)", cfg.STPeriod, cfg.STFactor, cfg.BBPeriod, cfg.BBK)
	} else {
		all.name = fmt.Sprintf("ST(%d,%.1f)+Boll(%d,%.1f)+EMA(%d)", cfg.STPeriod, cfg.STFactor, cfg.BBPeriod, cfg.BBK, cfg.EMAPeriod)
	}
	all.monthlyNet = make(map[string]float64)
	for _, f := range files {
		sym := strings.TrimSuffix(filepath.Base(f), "_"+*interval+".json")
		if strings.EqualFold(sym, "BTCUSDT") && !*includeBTC {
			continue
		}
		if selected := parseSymbolFilter(*symbolsFilter); len(selected) > 0 && !selected[strings.ToUpper(sym)] {
			continue
		}
		cs, err := loadCandles(f)
		if err != nil || len(cs) < *minBars {
			continue
		}
		cs = filterTime(cs, startMs, endMs)
		if len(cs) < *minBars {
			continue
		}
		var r symbolResult
		if cfg.EntryMode == "hfregime" {
			r = runHighFrequencySymbol(sym, cs, cfg)
		} else if cfg.EntryMode == "phase_ttrend" {
			r = runPhaseTTrendSymbol(sym, cs, cfg)
		} else if cfg.EntryMode == "state" {
			// State mode uses the reaction engine with a fresh-compression runner.
			r = runHighFrequencySymbol(sym, cs, cfg)
		} else if cfg.EntryMode == "phase_t2" {
			r = runPhaseT2Symbol(sym, cs, cfg)
		} else if cfg.EntryMode == "hybrid" || cfg.EntryMode == "stboll" {
			r = runHybridSymbol(sym, cs, cfg)
		} else {
			r = runSymbol(sym, cs, cfg)
		}
		all.symbols++
		all.bars += r.bars
		all.trades = append(all.trades, r.trades...)
		all.perSymbol = append(all.perSymbol, r)
		for _, tr := range r.trades {
			all.monthlyNet[time.UnixMilli(tr.exitTime).UTC().Format("2006-01")] += tr.net
		}
	}
	printSummary(all, cfg, *dataDir, *interval)
}

func runSymbol(sym string, cs []types.Candle, cfg config) symbolResult {
	ind := buildIndicators(cs, cfg)
	r := symbolResult{sym: sym, bars: len(cs)}
	minWarm := 0
	if cfg.EntryMode != "stboll" {
		minWarm = cfg.EMAPeriod
	}
	if cfg.BBPeriod > minWarm {
		minWarm = cfg.BBPeriod
	}
	if cfg.STPeriod > minWarm {
		minWarm = cfg.STPeriod
	}
	if cfg.EntryMode == "breakout" && cfg.SqueezeLB > minWarm {
		minWarm = cfg.SqueezeLB
	}
	pos := 0
	for i := minWarm; i < len(cs)-1; {
		if !ind[i].valid {
			i++
			continue
		}
		if pos == 0 {
			if longSignal(cs[i], ind[i], ind[i-1], cfg) {
				entry := cs[i+1].Open * (1 + cfg.SlipBps/10000)
				if entry > 0 {
					pos = 1
					entryIdx := i + 1
					for j := entryIdx; j < len(cs)-1; j++ {
						if !ind[j].valid {
							continue
						}
						if exitLong(cs[j], ind[j], j-entryIdx, cfg) {
							exit := cs[j+1].Open * (1 - cfg.SlipBps/10000)
							r.trades = append(r.trades, makeTrade(sym, 1, cs[entryIdx].OpenTime, cs[j+1].OpenTime, entry, exit, j-entryIdx, exitReasonLong(cs[j], ind[j], j-entryIdx, cfg), cfg))
							pos = 0
							i = j + 1
							break
						}
					}
					if pos == 1 {
						// Mark positions still open at the data boundary to the last
						// available close instead of silently dropping the trade.
						last := len(cs) - 1
						exit := cs[last].Close * (1 - cfg.SlipBps/10000)
						exitTime := cs[last].CloseTime
						if exitTime == 0 {
							exitTime = cs[last].OpenTime
						}
						r.trades = append(r.trades, makeTrade(sym, 1, cs[entryIdx].OpenTime, exitTime, entry, exit, last-entryIdx, "data_end", cfg))
						pos = 0
						break
					}
					continue
				}
			}
			if shortSignal(cs[i], ind[i], ind[i-1], cfg) {
				entry := cs[i+1].Open * (1 - cfg.SlipBps/10000)
				if entry > 0 {
					pos = -1
					entryIdx := i + 1
					for j := entryIdx; j < len(cs)-1; j++ {
						if !ind[j].valid {
							continue
						}
						if exitShort(cs[j], ind[j], j-entryIdx, cfg) {
							exit := cs[j+1].Open * (1 + cfg.SlipBps/10000)
							r.trades = append(r.trades, makeTrade(sym, -1, cs[entryIdx].OpenTime, cs[j+1].OpenTime, entry, exit, j-entryIdx, exitReasonShort(cs[j], ind[j], j-entryIdx, cfg), cfg))
							pos = 0
							i = j + 1
							break
						}
					}
					if pos == -1 {
						last := len(cs) - 1
						exit := cs[last].Close * (1 + cfg.SlipBps/10000)
						exitTime := cs[last].CloseTime
						if exitTime == 0 {
							exitTime = cs[last].OpenTime
						}
						r.trades = append(r.trades, makeTrade(sym, -1, cs[entryIdx].OpenTime, exitTime, entry, exit, last-entryIdx, "data_end", cfg))
						pos = 0
						break
					}
					continue
				}
			}
			i++
			continue
		}
	}
	r.maxDD = symbolMaxDD(r.trades)
	return r
}

func runHybridSymbol(sym string, cs []types.Candle, cfg config) symbolResult {
	ind := buildIndicators(cs, cfg)
	r := symbolResult{sym: sym, bars: len(cs)}
	minWarm := 0
	if cfg.EntryMode != "stboll" {
		minWarm = cfg.EMAPeriod
	}
	if cfg.BBPeriod > minWarm {
		minWarm = cfg.BBPeriod
	}
	if cfg.STPeriod > minWarm {
		minWarm = cfg.STPeriod
	}
	tWeight := cfg.TWeight
	if tWeight <= 0 || tWeight >= 1 {
		tWeight = 0.70
	}
	runnerWeight := 1 - tWeight

	for i := minWarm; i < len(cs)-1; {
		if !ind[i].valid {
			i++
			continue
		}
		dir := 0
		if longSignal(cs[i], ind[i], ind[i-1], cfg) {
			dir = 1
		} else if shortSignal(cs[i], ind[i], ind[i-1], cfg) {
			dir = -1
		}
		if dir == 0 || cs[i+1].Open <= 0 {
			i++
			continue
		}

		entry := cs[i+1].Open
		if dir > 0 {
			entry *= 1 + cfg.SlipBps/10000
		} else {
			entry *= 1 - cfg.SlipBps/10000
		}
		p := hybridPosition{
			dir:       dir,
			entryIdx:  i + 1,
			entryTime: cs[i+1].OpenTime,
			entry:     entry,
			tOpen:     true,
			rOpen:     true,
			tEntry:    entry,
			tEntryIdx: i + 1,
			tCycles:   1,
			tReason:   "open",
			rReason:   "open",
		}
		closed := false
		for j := p.entryIdx; j < len(cs)-1; j++ {
			if !ind[j].valid {
				continue
			}
			hold := j - p.entryIdx
			b := ind[j]
			c := cs[j]
			tClosed := false
			if dir > 0 {
				tStop := c.Close < b.lower
				if cfg.EntryMode != "stboll" {
					tStop = tStop || c.Close < b.ema
				}
				if p.tOpen && (c.Close >= b.mid || tStop) {
					exit := cs[j+1].Open * (1 - cfg.SlipBps/10000)
					p.tNet = compoundLeg(p.tNet, tWeight, netReturn(dir, p.tEntry, exit, cfg))
					if hold-p.tEntryIdx > p.tHold {
						p.tHold = hold - p.tEntryIdx
					}
					p.tReason = appendHybridReason(p.tReason, hybridTReasonLong(c, b))
					p.tOpen = false
					tClosed = true
					p.exitTime = cs[j+1].OpenTime
				}
				runnerStop := c.Close < b.st || hold >= cfg.MaxHold
				if cfg.RunnerExit == "st_ema" {
					runnerStop = c.Close < b.st || c.Close < b.ema || hold >= cfg.MaxHold
				}
				if p.rOpen && runnerStop {
					exit := cs[j+1].Open * (1 - cfg.SlipBps/10000)
					p.rNet = runnerWeight * netReturn(dir, p.entry, exit, cfg)
					p.rHold = hold
					p.rReason = hybridRunnerReasonLong(c, b, hold, cfg)
					p.rOpen = false
					p.exitTime = cs[j+1].OpenTime
				}
			} else {
				tStop := c.Close > b.upper
				if cfg.EntryMode != "stboll" {
					tStop = tStop || c.Close > b.ema
				}
				if p.tOpen && (c.Close <= b.mid || tStop) {
					exit := cs[j+1].Open * (1 + cfg.SlipBps/10000)
					p.tNet = compoundLeg(p.tNet, tWeight, netReturn(dir, p.tEntry, exit, cfg))
					if hold-p.tEntryIdx > p.tHold {
						p.tHold = hold - p.tEntryIdx
					}
					p.tReason = appendHybridReason(p.tReason, hybridTReasonShort(c, b))
					p.tOpen = false
					tClosed = true
					p.exitTime = cs[j+1].OpenTime
				}
				runnerStop := c.Close > b.st || hold >= cfg.MaxHold
				if cfg.RunnerExit == "st_ema" {
					runnerStop = c.Close > b.st || c.Close > b.ema || hold >= cfg.MaxHold
				}
				if p.rOpen && runnerStop {
					exit := cs[j+1].Open * (1 + cfg.SlipBps/10000)
					p.rNet = runnerWeight * netReturn(dir, p.entry, exit, cfg)
					p.rHold = hold
					p.rReason = hybridRunnerReasonShort(c, b, hold, cfg)
					p.rOpen = false
					p.exitTime = cs[j+1].OpenTime
				}
			}
			if p.rOpen && !p.tOpen && !tClosed && j+1 < len(cs)-1 {
				if (dir > 0 && longSignal(c, b, ind[j-1], cfg)) || (dir < 0 && shortSignal(c, b, ind[j-1], cfg)) {
					p.tEntry = cs[j+1].Open
					if dir > 0 {
						p.tEntry *= 1 + cfg.SlipBps/10000
					} else {
						p.tEntry *= 1 - cfg.SlipBps/10000
					}
					p.tEntryIdx = j + 1
					p.tOpen = true
					p.tCycles++
				}
			}
			if !p.tOpen && !p.rOpen {
				r.trades = append(r.trades, finishHybridTrade(sym, p))
				i = j + 1
				closed = true
				break
			}
		}
		if closed {
			continue
		}

		// Close surviving legs at the last available close. No next open exists.
		last := len(cs) - 1
		lastTime := cs[last].CloseTime
		if lastTime == 0 {
			lastTime = cs[last].OpenTime
		}
		exit := cs[last].Close
		if dir > 0 {
			exit *= 1 - cfg.SlipBps/10000
		} else {
			exit *= 1 + cfg.SlipBps/10000
		}
		if p.tOpen {
			p.tNet = compoundLeg(p.tNet, tWeight, netReturn(dir, p.tEntry, exit, cfg))
			if last-p.tEntryIdx > p.tHold {
				p.tHold = last - p.tEntryIdx
			}
			p.tReason = appendHybridReason(p.tReason, "data_end")
			p.tOpen = false
		}
		if p.rOpen {
			p.rNet = runnerWeight * netReturn(dir, p.entry, exit, cfg)
			p.rHold = last - p.entryIdx
			p.rReason = "data_end"
			p.rOpen = false
		}
		p.exitTime = lastTime
		r.trades = append(r.trades, finishHybridTrade(sym, p))
		break
	}
	r.maxDD = symbolMaxDD(r.trades)
	return r
}

// runHighFrequencySymbol keeps one ST-regime gate, treats reaction signals as
// independent Bollinger T trades, and starts the funded runner only after a
// same-direction Bollinger breakout. Both legs use closed-bar signals and the
// next candle open, so frequency does not come from look-ahead fills.
func runHighFrequencySymbol(sym string, cs []types.Candle, cfg config) symbolResult {
	ind := buildIndicators(cs, cfg)
	r := symbolResult{sym: sym, bars: len(cs)}
	minWarm := cfg.BBPeriod
	if cfg.STPeriod > minWarm {
		minWarm = cfg.STPeriod
	}
	if minWarm < 3 {
		minWarm = 3
	}
	tWeight := cfg.TWeight
	if tWeight <= 0 || tWeight >= 1 {
		tWeight = 0.70
	}
	tAllocFraction := cfg.TAlloc
	if tAllocFraction <= 0 || tAllocFraction > 1 {
		tAllocFraction = 1
	}
	tGap := cfg.TGap
	if tGap < 0 {
		tGap = 0
	}
	tHold := cfg.THold
	if tHold < 1 {
		tHold = 12
	}

	runnerOpen := false
	dir := 0
	runnerEntryIdx := 0
	runnerEntryTime := int64(0)
	runnerEntry := 0.0
	runnerBest := 0.0
	tOpen := false
	tEntryIdx := 0
	tEntry := 0.0
	tAlloc := 0.0
	lastTExitIdx := -tGap
	runnerAlloc := 0.0

	openPrice := func(i int, direction int) float64 {
		if i < 0 || i >= len(cs) || cs[i].Open <= 0 {
			return 0
		}
		if direction > 0 {
			return cs[i].Open * (1 + cfg.SlipBps/10000)
		}
		return cs[i].Open * (1 - cfg.SlipBps/10000)
	}
	closePrice := func(i int, direction int) float64 {
		if i < 0 || i >= len(cs) || cs[i].Close <= 0 {
			return 0
		}
		if direction > 0 {
			return cs[i].Open * (1 - cfg.SlipBps/10000)
		}
		return cs[i].Open * (1 + cfg.SlipBps/10000)
	}

	for i := minWarm; i < len(cs)-1; i++ {
		if !ind[i].valid || cs[i+1].Open <= 0 {
			continue
		}
		b := ind[i]
		c := cs[i]
		tClosedThisBar := false

		if tOpen {
			hold := i - tEntryIdx
			tStop := false
			if strings.EqualFold(cfg.TStopMode, "st") {
				tStop = (dir > 0 && c.Close < b.st) || (dir < 0 && c.Close > b.st)
			} else {
				tStop = (dir > 0 && c.Close < b.lower) || (dir < 0 && c.Close > b.upper)
			}
			hardStop := false
			if cfg.TStopPct > 0 {
				stopPct := cfg.TStopPct / 100
				hardStop = (dir > 0 && c.Close <= tEntry*(1-stopPct)) || (dir < 0 && c.Close >= tEntry*(1+stopPct))
				tStop = tStop || hardStop
			}
			tTarget := (dir > 0 && c.Close >= b.mid) || (dir < 0 && c.Close <= b.mid)
			if tStop || tTarget || hold >= tHold {
				exit := closePrice(i+1, dir)
				if exit > 0 {
					net := tAlloc * netReturn(dir, tEntry, exit, cfg)
					reason := "timeout"
					if tStop {
						reason = "outer_stop"
						if hardStop {
							reason = "hard_stop"
						}
					} else if tTarget {
						reason = "bb_mid"
					}
					r.trades = append(r.trades, makeLegTrade(sym, "t", dir, cs[tEntryIdx].OpenTime, cs[i+1].OpenTime, tEntry, exit, hold, reason, net))
				}
				tOpen = false
				lastTExitIdx = i
				tClosedThisBar = true
			}
		}

		if runnerOpen {
			hold := i - runnerEntryIdx
			if dir > 0 && c.Close > runnerBest {
				runnerBest = c.Close
			}
			if dir < 0 && (runnerBest == 0 || c.Close < runnerBest) {
				runnerBest = c.Close
			}
			runnerStop := false
			switch strings.ToLower(cfg.RunnerExit) {
			case "mid":
				runnerStop = (dir > 0 && c.Close < b.mid) || (dir < 0 && c.Close > b.mid)
			case "outer":
				runnerStop = (dir > 0 && c.Close < b.lower) || (dir < 0 && c.Close > b.upper)
			case "none":
				runnerStop = false
			case "st_ema":
				runnerStop = (dir > 0 && (c.Close < b.st || c.Close < b.ema)) || (dir < 0 && (c.Close > b.st || c.Close > b.ema))
			default:
				runnerStop = (dir > 0 && c.Close < b.st) || (dir < 0 && c.Close > b.st)
			}
			hardStop := false
			if cfg.RunnerStopPct > 0 {
				stopPct := cfg.RunnerStopPct / 100
				hardStop = (dir > 0 && c.Close <= runnerEntry*(1-stopPct)) || (dir < 0 && c.Close >= runnerEntry*(1+stopPct))
				runnerStop = runnerStop || hardStop
			}
			trailStop := false
			if cfg.RunnerTrailPct > 0 && runnerBest > 0 {
				trail := cfg.RunnerTrailPct / 100
				trailStop = (dir > 0 && c.Close <= runnerBest*(1-trail)) || (dir < 0 && c.Close >= runnerBest*(1+trail))
				runnerStop = runnerStop || trailStop
			}
			timedOut := hold >= cfg.MaxHold
			runnerStop = runnerStop || timedOut
			if runnerStop {
				exit := closePrice(i+1, dir)
				if exit > 0 && runnerAlloc > 0 {
					net := runnerAlloc * netReturn(dir, runnerEntry, exit, cfg)
					reason := "stop_st"
					if hardStop {
						reason = "hard_stop"
					} else if trailStop {
						reason = "trail_stop"
					} else if cfg.RunnerExit == "mid" {
						reason = "stop_mid"
					} else if cfg.RunnerExit == "outer" {
						reason = "stop_outer"
					} else if cfg.RunnerExit == "st_ema" {
						reason = "stop_st_ema"
					} else if timedOut {
						reason = "timeout"
					}
					r.trades = append(r.trades, makeLegTrade(sym, "runner", dir, runnerEntryTime, cs[i+1].OpenTime, runnerEntry, exit, hold, reason, net))
				}
				runnerOpen = false
				tClosedThisBar = true
				if tOpen {
					tOpen = false
					lastTExitIdx = i
				}
			}
		}

		signalDir := highFrequencySignal(c, ind, i)
		if !tSideAllowed(signalDir, cfg.TSide) {
			signalDir = 0
		}
		if !tRegimeAllowed(ind, i, signalDir, cfg.TRegime, cfg.TPhaseLB, cfg.TPhaseMin) {
			signalDir = 0
		}
		if runnerOpen && runnerAlloc == 0 && !tOpen && !tClosedThisBar && trendBreakoutSignal(c, ind, i, dir, cfg) {
			entry := openPrice(i+1, dir)
			if entry > 0 {
				runnerEntry = entry
				runnerBest = entry
				runnerAlloc = 1 - tWeight
				runnerEntryIdx = i + 1
				runnerEntryTime = cs[i+1].OpenTime
			}
		}
		if !runnerOpen || (cfg.TIndependent && runnerAlloc == 0) {
			if signalDir == 0 || tClosedThisBar {
				continue
			}
			entryPrice := openPrice(i+1, signalDir)
			if entryPrice <= 0 {
				continue
			}
			dir = signalDir
			activeTWeight := tAllocFraction
			tEntry = entryPrice
			tAlloc = activeTWeight
			tEntryIdx = i + 1
			tOpen = true
			// Keep a zero-capital runner as the ST-regime gate when the
			// trend filter rejects the signal. This preserves T cadence while
			// preventing weak signals from opening independent regimes.
			runnerEntry = entryPrice
			runnerBest = entryPrice
			runnerOpen = true
			runnerAlloc = 0
			runnerEntryIdx = i + 1
			runnerEntryTime = cs[i+1].OpenTime
			continue
		}

		if !tOpen && !tClosedThisBar && i-lastTExitIdx >= tGap && signalDir == dir {
			tEntry = openPrice(i+1, dir)
			if tEntry > 0 {
				if runnerAlloc == 0 {
					tAlloc = tAllocFraction
				} else {
					tAlloc = tWeight
				}
				tEntryIdx = i + 1
				tOpen = true
			}
		}
	}

	last := len(cs) - 1
	lastTime := cs[last].CloseTime
	if lastTime == 0 {
		lastTime = cs[last].OpenTime
	}
	if runnerOpen {
		exit := cs[last].Close
		if dir > 0 {
			exit *= 1 - cfg.SlipBps/10000
		} else {
			exit *= 1 + cfg.SlipBps/10000
		}
		if tOpen {
			r.trades = append(r.trades, makeLegTrade(sym, "t", dir, cs[tEntryIdx].OpenTime, lastTime, tEntry, exit, last-tEntryIdx, "data_end", tAlloc*netReturn(dir, tEntry, exit, cfg)))
		}
		if runnerAlloc > 0 {
			r.trades = append(r.trades, makeLegTrade(sym, "runner", dir, runnerEntryTime, lastTime, runnerEntry, exit, last-runnerEntryIdx, "data_end", runnerAlloc*netReturn(dir, runnerEntry, exit, cfg)))
		}
	} else if tOpen {
		exit := cs[last].Close
		if dir > 0 {
			exit *= 1 - cfg.SlipBps/10000
		} else {
			exit *= 1 + cfg.SlipBps/10000
		}
		r.trades = append(r.trades, makeLegTrade(sym, "t", dir, cs[tEntryIdx].OpenTime, lastTime, tEntry, exit, last-tEntryIdx, "data_end", tAlloc*netReturn(dir, tEntry, exit, cfg)))
	}
	r.maxDD = symbolMaxDD(r.trades)
	return r
}

// runPhaseTTrendSymbol separates the local Bollinger T from the trend leg.
// Direction comes from the 12-bar phase state; ST is only a support/resistance
// location and an exit line, never a standalone long/short switch.
func runPhaseTTrendSymbol(sym string, cs []types.Candle, cfg config) symbolResult {
	ind := buildIndicators(cs, cfg)
	marketRegime := cfg.MarketRegime
	if marketRegime == nil {
		marketRegime = buildMarketRegime(cs, cfg, cfg.PhaseLB)
	}
	r := symbolResult{sym: sym, bars: len(cs)}
	minWarm := cfg.BBPeriod
	if cfg.STPeriod > minWarm {
		minWarm = cfg.STPeriod
	}
	if minWarm < 15 {
		minWarm = 15
	}
	tWeight := cfg.TWeight
	if tWeight <= 0 || tWeight >= 1 {
		tWeight = 0.50
	}
	tGap := cfg.TGap
	if tGap < 0 {
		tGap = 0
	}
	tHold := cfg.THold
	if tHold < 1 {
		tHold = 12
	}

	tOpen, runnerOpen := false, false
	tDir, runnerDir := 0, 0
	tEntryIdx, runnerEntryIdx := 0, 0
	tEntryTime, runnerEntryTime := int64(0), int64(0)
	tEntry, runnerEntry := 0.0, 0.0
	tTarget := 0.0
	runnerBest := 0.0
	tAlloc, runnerAlloc := 0.0, 0.0
	lastTExitIdx := -tGap
	trendArmDir, trendArmIdx := 0, 0

	openPrice := func(i, direction int) float64 {
		if i < 0 || i >= len(cs) || cs[i].Open <= 0 {
			return 0
		}
		if direction > 0 {
			return cs[i].Open * (1 + cfg.SlipBps/10000)
		}
		return cs[i].Open * (1 - cfg.SlipBps/10000)
	}
	closePrice := func(i, direction int) float64 {
		if i < 0 || i >= len(cs) || cs[i].Open <= 0 {
			return 0
		}
		if direction > 0 {
			return cs[i].Open * (1 - cfg.SlipBps/10000)
		}
		return cs[i].Open * (1 + cfg.SlipBps/10000)
	}
	appendExit := func(kind string, direction int, entryIdx int, entryTime int64, entry float64, alloc float64, exitIdx int, reason string) {
		exit := closePrice(exitIdx, direction)
		if exit <= 0 || alloc <= 0 {
			return
		}
		net := alloc * netReturn(direction, entry, exit, cfg)
		r.trades = append(r.trades, makeLegTrade(sym, kind, direction, entryTime, cs[exitIdx].OpenTime, entry, exit, exitIdx-entryIdx, reason, net))
	}

	for i := minWarm; i < len(cs)-1; i++ {
		if !ind[i].valid || cs[i+1].Open <= 0 {
			continue
		}
		b := ind[i]
		c := cs[i]
		tClosed := false

		if tOpen {
			hold := i - tEntryIdx
			stop := false
			if strings.EqualFold(cfg.TStopMode, "st") {
				stop = (tDir > 0 && c.Close < b.st) || (tDir < 0 && c.Close > b.st)
			} else {
				stop = (tDir > 0 && c.Close < b.lower) || (tDir < 0 && c.Close > b.upper)
			}
			if cfg.TStopPct > 0 {
				hardStop := cfg.TStopPct / 100
				stop = stop || (tDir > 0 && c.Close <= tEntry*(1-hardStop)) || (tDir < 0 && c.Close >= tEntry*(1+hardStop))
			}
			target := (tDir > 0 && c.Close >= tTarget) || (tDir < 0 && c.Close <= tTarget)
			if stop || target || hold >= tHold {
				reason := "timeout"
				if stop {
					reason = "outer_stop"
					if cfg.TStopPct > 0 && ((tDir > 0 && c.Close <= tEntry*(1-cfg.TStopPct/100)) || (tDir < 0 && c.Close >= tEntry*(1+cfg.TStopPct/100))) {
						reason = "hard_stop"
					}
				} else if target {
					reason = "bb_mid"
				}
				appendExit("t", tDir, tEntryIdx, tEntryTime, tEntry, tAlloc, i+1, reason)
				tOpen = false
				lastTExitIdx = i
				tClosed = true
			}
		}

		if runnerOpen {
			hold := i - runnerEntryIdx
			if runnerDir > 0 && c.Close > runnerBest {
				runnerBest = c.Close
			}
			if runnerDir < 0 && (runnerBest == 0 || c.Close < runnerBest) {
				runnerBest = c.Close
			}
			stop := false
			switch cfg.RunnerExit {
			case "mid":
				stop = (runnerDir > 0 && c.Close < b.mid) || (runnerDir < 0 && c.Close > b.mid)
			case "opposite":
				stop = (runnerDir > 0 && c.Close < b.lower) || (runnerDir < 0 && c.Close > b.upper)
			case "none":
				stop = false
			default:
				stop = (runnerDir > 0 && c.Close < b.st) || (runnerDir < 0 && c.Close > b.st)
			}
			if cfg.RunnerStopPct > 0 {
				hardStop := cfg.RunnerStopPct / 100
				stop = stop || (runnerDir > 0 && c.Close <= runnerEntry*(1-hardStop)) || (runnerDir < 0 && c.Close >= runnerEntry*(1+hardStop))
			}
			if cfg.RunnerTrailPct > 0 && runnerBest > 0 {
				trail := cfg.RunnerTrailPct / 100
				stop = stop || (runnerDir > 0 && c.Close <= runnerBest*(1-trail)) || (runnerDir < 0 && c.Close >= runnerBest*(1+trail))
			}
			if stop || hold >= cfg.MaxHold {
				reason := "stop_st"
				if !stop {
					reason = "timeout"
				} else if cfg.RunnerExit == "mid" {
					reason = "stop_mid"
				} else if cfg.RunnerExit == "opposite" {
					reason = "stop_outer"
				} else if cfg.RunnerStopPct > 0 && ((runnerDir > 0 && c.Close <= runnerEntry*(1-cfg.RunnerStopPct/100)) || (runnerDir < 0 && c.Close >= runnerEntry*(1+cfg.RunnerStopPct/100))) {
					reason = "hard_stop"
				} else if cfg.RunnerTrailPct > 0 && ((runnerDir > 0 && c.Close <= runnerBest*(1-cfg.RunnerTrailPct/100)) || (runnerDir < 0 && c.Close >= runnerBest*(1+cfg.RunnerTrailPct/100))) {
					reason = "trail_stop"
				}
				appendExit("runner", runnerDir, runnerEntryIdx, runnerEntryTime, runnerEntry, runnerAlloc, i+1, reason)
				runnerOpen = false
				runnerAlloc = 0
			}
		}

		regime := phaseRegime(ind, i)
		if marketRegime != nil {
			if phase, ok := marketRegime[c.OpenTime]; ok {
				regime = phase
			}
		}
		signal := phaseTSignal(c, ind, i, regime, cfg.CrossW)
		if !tSideAllowed(signal, cfg.TSide) {
			signal = 0
		}
		if !tRegimeAllowed(ind, i, signal, cfg.TRegime, cfg.TPhaseLB, cfg.TPhaseMin) {
			signal = 0
		}
		if signal != 0 && cfg.TMinMovePct > 0 {
			minMove := cfg.TMinMovePct / 100
			if (signal > 0 && (b.mid-c.Close)/c.Close < minMove) || (signal < 0 && (c.Close-b.mid)/c.Close < minMove) {
				signal = 0
			}
		}

		// First record a confirmed outer-band move, then wait for a retest of
		// the broken band before entering. A T reaction never opens the runner.
		if trendArmDir != 0 {
			invalid := regime != trendArmDir || i-trendArmIdx > 12
			if trendArmDir > 0 {
				invalid = invalid || c.Close < b.mid
			} else {
				invalid = invalid || c.Close > b.mid
			}
			if invalid {
				trendArmDir = 0
			}
		}
		if arm := phaseBreakoutSignal(c, ind, i, regime); arm != 0 {
			trendArmDir = arm
			trendArmIdx = i
		}
		pullback := 0
		if trendArmDir > 0 && i > trendArmIdx {
			if strings.EqualFold(cfg.RunnerSetup, "mid") {
				if i >= 3 && c.Low <= b.mid && c.Close > b.mid && c.Close >= c.Open && ind[i].dir == 1 && b.mid >= ind[i-3].mid {
					pullback = 1
				}
			} else if c.Low <= ind[i-1].upper && c.Close > b.upper && c.Close >= c.Open {
				pullback = 1
			}
		}
		if trendArmDir < 0 && i > trendArmIdx {
			if strings.EqualFold(cfg.RunnerSetup, "mid") {
				if i >= 3 && c.High >= b.mid && c.Close < b.mid && c.Close <= c.Open && ind[i].dir == -1 && b.mid <= ind[i-3].mid {
					pullback = -1
				}
			} else if c.High >= ind[i-1].lower && c.Close < b.lower && c.Close <= c.Open {
				pullback = -1
			}
		}
		if !runnerOpen && pullback != 0 && (!tOpen || tDir == pullback) {
			entry := openPrice(i+1, pullback)
			if entry > 0 {
				runnerOpen = true
				runnerDir = pullback
				runnerEntryIdx = i + 1
				runnerEntryTime = cs[i+1].OpenTime
				runnerEntry = entry
				runnerBest = entry
				runnerAlloc = 1
				if tOpen {
					runnerAlloc = 1 - tWeight
				}
				trendArmDir = 0
			}
		}

		if signal == 0 || tClosed || i-lastTExitIdx < tGap {
			continue
		}
		if runnerOpen && signal != runnerDir {
			continue
		}

		if !tOpen {
			entry := openPrice(i+1, signal)
			if entry <= 0 {
				continue
			}
			tOpen = true
			tDir = signal
			tEntryIdx = i + 1
			tEntryTime = cs[i+1].OpenTime
			tEntry = entry
			frac := cfg.TTargetFrac
			if frac <= 0 || frac > 1 {
				frac = 1
			}
			tTarget = entry + frac*(b.mid-entry)
			tAlloc = tWeight
			if runnerOpen {
				runnerAlloc = 1 - tWeight
			}
			continue
		}

		if runnerOpen || signal != tDir {
			continue
		}
	}

	last := len(cs) - 1
	lastTime := cs[last].CloseTime
	if lastTime == 0 {
		lastTime = cs[last].OpenTime
	}
	if tOpen {
		exit := cs[last].Close
		if tDir > 0 {
			exit *= 1 - cfg.SlipBps/10000
		} else {
			exit *= 1 + cfg.SlipBps/10000
		}
		r.trades = append(r.trades, makeLegTrade(sym, "t", tDir, tEntryTime, lastTime, tEntry, exit, last-tEntryIdx, "data_end", tAlloc*netReturn(tDir, tEntry, exit, cfg)))
	}
	if runnerOpen {
		exit := cs[last].Close
		if runnerDir > 0 {
			exit *= 1 - cfg.SlipBps/10000
		} else {
			exit *= 1 + cfg.SlipBps/10000
		}
		r.trades = append(r.trades, makeLegTrade(sym, "runner", runnerDir, runnerEntryTime, lastTime, runnerEntry, exit, last-runnerEntryIdx, "data_end", runnerAlloc*netReturn(runnerDir, runnerEntry, exit, cfg)))
	}
	r.maxDD = symbolMaxDD(r.trades)
	return r
}

func phaseRegime(ind []indicatorBar, i int) int {
	up, down, flat := stateBias(ind, i)
	if up && !down {
		return 1
	}
	if down && !up {
		return -1
	}
	if flat {
		return 2
	}
	return 0
}

func tSideAllowed(dir int, side string) bool {
	if dir == 0 {
		return false
	}
	switch strings.ToLower(side) {
	case "long":
		return dir > 0
	case "short":
		return dir < 0
	default:
		return true
	}
}

func tRegimeAllowed(ind []indicatorBar, i, dir int, regime string, lookback, minCount int) bool {
	if dir == 0 || strings.EqualFold(regime, "both") {
		return dir != 0
	}
	up, down := phaseBias(ind, i, lookback, minCount)
	switch strings.ToLower(regime) {
	case "up":
		return dir > 0 && up
	case "down":
		return dir < 0 && down
	case "trend":
		return (dir > 0 && up) || (dir < 0 && down)
	default:
		return true
	}
}

func phaseBias(ind []indicatorBar, i, lookback, minCount int) (up, down bool) {
	if lookback < 2 {
		lookback = 2
	}
	if minCount < 1 {
		minCount = lookback
	}
	if minCount > lookback {
		minCount = lookback
	}
	if i < lookback || !ind[i].valid || !ind[i-lookback].valid {
		return false, false
	}
	above, below := 0, 0
	start := i - lookback + 1
	for j := start; j <= i; j++ {
		if ind[j].close >= ind[j].mid {
			above++
		}
		if ind[j].close <= ind[j].mid {
			below++
		}
	}
	up = ind[i].mid > ind[i-lookback].mid && above >= minCount
	down = ind[i].mid < ind[i-lookback].mid && below >= minCount
	return up, down
}

func phaseSTFlat(ind []indicatorBar, i int) bool {
	if i < 3 || !ind[i].valid || bandWidth(ind[i]) <= 0 {
		return false
	}
	width := bandWidth(ind[i])
	for j := i - 2; j <= i; j++ {
		if math.Abs(ind[j].st-ind[j-1].st) > 0.25*width {
			return false
		}
	}
	return true
}

func phaseTSignal(c types.Candle, ind []indicatorBar, i, regime, crossWindow int) int {
	if i < 1 || !ind[i].valid || bandWidth(ind[i]) <= 0 {
		return 0
	}
	b := ind[i]
	stSupport := b.dir == 1 && c.Low <= b.st && c.Close > b.st && c.Close < b.mid
	stResistance := b.dir == -1 && c.High >= b.st && c.Close < b.st && c.Close > b.mid
	lowerReclaim := c.Low <= b.lower && c.Close > b.lower && b.st >= b.lower
	upperReject := c.High >= b.upper && c.Close < b.upper && b.st <= b.upper
	if regime == 1 && (stSupport || (lowerReclaim && b.st < b.lower)) {
		return 1
	}
	if regime == -1 && (stResistance || (upperReject && b.st > b.upper)) {
		return -1
	}
	if regime == 2 {
		if (stSupport || lowerReclaim) && b.lowerOuterAge > crossWindow {
			return 1
		}
		if (stResistance || upperReject) && b.upperOuterAge > crossWindow {
			return -1
		}
	}
	return 0
}

func phaseBreakoutSignal(c types.Candle, ind []indicatorBar, i, regime int) int {
	if i < 1 || !ind[i].valid || !ind[i-1].valid || regime == 0 || regime == 2 {
		return 0
	}
	if bandWidth(ind[i]) <= 0 || bandWidth(ind[i-1]) <= 0 {
		return 0
	}
	if regime == 1 && ind[i].dir == 1 && c.Close > ind[i].upper && ind[i-1].close > ind[i-1].upper && bandWidth(ind[i]) >= bandWidth(ind[i-1]) {
		return 1
	}
	if regime == -1 && ind[i].dir == -1 && c.Close < ind[i].lower && ind[i-1].close < ind[i-1].lower && bandWidth(ind[i]) >= bandWidth(ind[i-1]) {
		return -1
	}
	return 0
}

func phasePullbackSignal(c types.Candle, ind []indicatorBar, i, regime int) int {
	if i < 6 || !ind[i].valid || !ind[i-1].valid || !ind[i-6].valid {
		return 0
	}
	b := ind[i]
	if regime == 1 && b.dir == 1 && b.mid > ind[i-6].mid && c.Low <= b.lower && c.Close > b.lower && c.Close >= c.Open {
		return 1
	}
	if regime == -1 && b.dir == -1 && b.mid < ind[i-6].mid && c.High >= b.upper && c.Close < b.upper && c.Close <= c.Open {
		return -1
	}
	return 0
}

func phaseTrendEligible(c types.Candle, ind []indicatorBar, i, regime, signal int) bool {
	if i < 3 || regime == 2 || signal == 0 {
		return false
	}
	b := ind[i]
	if signal > 0 && regime == 1 {
		return c.Low <= b.lower && c.Close > b.lower && b.st >= b.lower && b.mid > ind[i-3].mid
	}
	if signal < 0 && regime == -1 {
		return c.High >= b.upper && c.Close < b.upper && b.st <= b.upper && b.mid < ind[i-3].mid
	}
	return false
}

// runStateSymbol implements the state-machine strategy. Outer-band touches are
// short T trades, ST/middle-line flattening is a separate pulse setup, and
// confirmed outer-band expansion opens a trend runner. ST direction is never a
// global long/short gate.
func runStateSymbol(sym string, cs []types.Candle, cfg config) symbolResult {
	ind := buildIndicators(cs, cfg)
	r := symbolResult{sym: sym, bars: len(cs)}
	minWarm := cfg.BBPeriod
	if cfg.STPeriod > minWarm {
		minWarm = cfg.STPeriod
	}
	if minWarm < 5 {
		minWarm = 5
	}
	tWeight := cfg.TWeight
	if tWeight <= 0 || tWeight >= 1 {
		tWeight = 0.40
	}
	tGap := cfg.TGap
	if tGap < 0 {
		tGap = 0
	}
	tHold := cfg.THold
	if tHold < 1 {
		tHold = 12
	}

	runnerOpen := false
	runnerDir := 0
	runnerEntryIdx := 0
	runnerEntryTime := int64(0)
	runnerEntry := 0.0
	runnerAlloc := 0.0
	phaseOpen := false
	phaseDir := 0
	phaseEntryIdx := 0
	tOpen := false
	tDir := 0
	tEntryIdx := 0
	tEntryTime := int64(0)
	tEntry := 0.0
	tAlloc := 0.0
	lastTExitIdx := -tGap

	openPrice := func(i, direction int) float64 {
		if i < 0 || i >= len(cs) || cs[i].Open <= 0 {
			return 0
		}
		if direction > 0 {
			return cs[i].Open * (1 + cfg.SlipBps/10000)
		}
		return cs[i].Open * (1 - cfg.SlipBps/10000)
	}
	closePrice := func(i, direction int) float64 {
		if i < 0 || i >= len(cs) || cs[i].Open <= 0 {
			return 0
		}
		if direction > 0 {
			return cs[i].Open * (1 - cfg.SlipBps/10000)
		}
		return cs[i].Open * (1 + cfg.SlipBps/10000)
	}

	for i := minWarm; i < len(cs)-1; i++ {
		if !ind[i].valid || cs[i+1].Open <= 0 {
			continue
		}
		b := ind[i]
		c := cs[i]
		tClosedThisBar := false

		// A T reaction starts a local phase. Keep its direction until price
		// loses ST support/resistance or the phase ages out; this prevents a
		// single noisy candle from flipping the strategy long/short.
		if phaseOpen && !runnerOpen {
			phaseHold := i - phaseEntryIdx
			phaseStop := (phaseDir > 0 && c.Close < b.st) || (phaseDir < 0 && c.Close > b.st)
			if phaseStop || phaseHold >= cfg.MaxHold {
				phaseOpen = false
				phaseDir = 0
			}
		}

		if tOpen {
			hold := i - tEntryIdx
			stop := (tDir > 0 && c.Close < b.lower) || (tDir < 0 && c.Close > b.upper)
			target := (tDir > 0 && c.Close >= b.mid) || (tDir < 0 && c.Close <= b.mid)
			if stop || target || hold >= tHold {
				exit := closePrice(i+1, tDir)
				if exit > 0 {
					reason := "timeout"
					if stop {
						reason = "outer_stop"
					} else if target {
						reason = "bb_mid"
					}
					r.trades = append(r.trades, makeLegTrade(sym, "t", tDir, tEntryTime, cs[i+1].OpenTime, tEntry, exit, hold, reason, tAlloc*netReturn(tDir, tEntry, exit, cfg)))
				}
				tOpen = false
				tClosedThisBar = true
				lastTExitIdx = i
			}
		}

		if runnerOpen {
			hold := i - runnerEntryIdx
			stop := (runnerDir > 0 && c.Close < b.st) || (runnerDir < 0 && c.Close > b.st)
			if stop || hold >= cfg.MaxHold {
				exit := closePrice(i+1, runnerDir)
				if exit > 0 {
					reason := "stop_st"
					if !stop {
						reason = "timeout"
					}
					r.trades = append(r.trades, makeLegTrade(sym, "runner", runnerDir, runnerEntryTime, cs[i+1].OpenTime, runnerEntry, exit, hold, reason, runnerAlloc*netReturn(runnerDir, runnerEntry, exit, cfg)))
				}
				runnerOpen = false
				runnerAlloc = 0
				phaseOpen = false
				phaseDir = 0
			}
		}

		breakoutDir := stateBreakoutSignal(c, ind, i, cfg)
		if !runnerOpen && breakoutDir != 0 && (!phaseOpen || phaseDir == breakoutDir) && (!tOpen || tDir == breakoutDir) {
			entry := openPrice(i+1, breakoutDir)
			if entry > 0 {
				runnerOpen = true
				runnerDir = breakoutDir
				runnerEntryIdx = i + 1
				runnerEntryTime = cs[i+1].OpenTime
				runnerEntry = entry
				if tOpen {
					runnerAlloc = 1 - tWeight
				} else {
					runnerAlloc = 1
				}
			}
		}

		if tClosedThisBar || (runnerOpen && runnerAlloc >= 0.999) {
			continue
		}
		signalDir := stateReactionSignal(c, ind, i)
		if !tSideAllowed(signalDir, cfg.TSide) {
			signalDir = 0
		}
		if signalDir == 0 || i-lastTExitIdx < tGap {
			continue
		}
		if runnerOpen && signalDir != runnerDir {
			continue
		}
		if tOpen {
			continue
		}
		if phaseOpen && signalDir != phaseDir {
			continue
		}
		entry := openPrice(i+1, signalDir)
		if entry <= 0 {
			continue
		}
		tOpen = true
		tDir = signalDir
		if !phaseOpen {
			phaseOpen = true
			phaseDir = signalDir
			phaseEntryIdx = i + 1
		}
		tEntryIdx = i + 1
		tEntryTime = cs[i+1].OpenTime
		tEntry = entry
		if runnerOpen {
			tAlloc = tWeight
		} else {
			tAlloc = 1
		}
	}

	last := len(cs) - 1
	lastTime := cs[last].CloseTime
	if lastTime == 0 {
		lastTime = cs[last].OpenTime
	}
	if tOpen {
		exit := cs[last].Close
		if tDir > 0 {
			exit *= 1 - cfg.SlipBps/10000
		} else {
			exit *= 1 + cfg.SlipBps/10000
		}
		r.trades = append(r.trades, makeLegTrade(sym, "t", tDir, tEntryTime, lastTime, tEntry, exit, last-tEntryIdx, "data_end", tAlloc*netReturn(tDir, tEntry, exit, cfg)))
	}
	if runnerOpen {
		exit := cs[last].Close
		if runnerDir > 0 {
			exit *= 1 - cfg.SlipBps/10000
		} else {
			exit *= 1 + cfg.SlipBps/10000
		}
		r.trades = append(r.trades, makeLegTrade(sym, "runner", runnerDir, runnerEntryTime, lastTime, runnerEntry, exit, last-runnerEntryIdx, "data_end", runnerAlloc*netReturn(runnerDir, runnerEntry, exit, cfg)))
	}
	r.maxDD = symbolMaxDD(r.trades)
	return r
}

type phaseT2Position struct {
	dir        int
	entryIdx   int
	entryTime  int64
	entry      float64
	stop       float64
	target     float64
	allocation float64
	state      string
	armed      bool
	residual   float64
	partialNet float64
}

// runPhaseT2Symbol trades only closed-bar ST/Bollinger reactions. The initial
// stop is fixed at entry; after the middle-band partial, the residual stop and
// outer target are updated from the next closed bars only.
func runPhaseT2Symbol(sym string, cs []types.Candle, cfg config) symbolResult {
	ind := buildIndicators(cs, cfg)
	r := symbolResult{sym: sym, bars: len(cs)}
	minWarm := cfg.BBPeriod
	if cfg.STPeriod > minWarm {
		minWarm = cfg.STPeriod
	}
	if minWarm < 3 {
		minWarm = 3
	}
	gap := cfg.T2Gap
	if gap < 0 {
		gap = 0
	}
	maxHold := cfg.T2Hold
	if maxHold < 1 {
		maxHold = 12
	}
	lastExitIdx := -gap
	var pos phaseT2Position

	openPrice := func(i, dir int) float64 {
		if i < 0 || i >= len(cs) || cs[i].Open <= 0 {
			return 0
		}
		if dir > 0 {
			return cs[i].Open * (1 + cfg.SlipBps/10000)
		}
		return cs[i].Open * (1 - cfg.SlipBps/10000)
	}
	closePrice := func(i, dir int) float64 {
		if i < 0 || i >= len(cs) || cs[i].Open <= 0 {
			return 0
		}
		if dir > 0 {
			return cs[i].Open * (1 - cfg.SlipBps/10000)
		}
		return cs[i].Open * (1 + cfg.SlipBps/10000)
	}
	appendTrade := func(exitIdx int, exit, net float64, reason string) {
		if exitIdx < 0 || exitIdx >= len(cs) || pos.dir == 0 || exit <= 0 {
			return
		}
		r.trades = append(r.trades, trade{
			sym: sym, dir: pos.dir, entryTime: pos.entryTime, exitTime: cs[exitIdx].OpenTime,
			entry: pos.entry, exit: exit, net: net, hold: exitIdx - pos.entryIdx,
			reason: reason, tCycles: 1, kind: "t", family: "phase_t2", exitModel: "fixed", state: pos.state,
		})
	}

	for i := minWarm; i < len(cs)-1; i++ {
		if !ind[i].valid || !ind[i-1].valid || cs[i+1].Open <= 0 {
			continue
		}
		c := cs[i]
		if pos.dir != 0 {
			reason, trigger, partial := phaseT2ManagedExit(pos, c, ind[i], i-pos.entryIdx, maxHold, cfg)
			if trigger {
				exit := closePrice(i+1, pos.dir)
				if exit > 0 {
					if partial && !pos.armed && pos.residual > 0.5 {
						partialReturn := 0.5 * netReturn(pos.dir, pos.entry, exit, cfg)
						appendTrade(i+1, exit, pos.allocation*partialReturn, reason)
						pos.partialNet += partialReturn
						pos.residual = 0.5
						pos.armed = true
						continue
					}
					net := pos.partialNet + pos.residual*netReturn(pos.dir, pos.entry, exit, cfg)
					appendTrade(i+1, exit, pos.allocation*net, reason)
				}
				pos = phaseT2Position{}
				lastExitIdx = i
				continue
			}
			continue
		}
		if i-lastExitIdx < gap {
			continue
		}
		dir, state := phaseT2Signal(c, cs[i-1], ind, i, cfg)
		if dir == 0 {
			continue
		}
		entry := openPrice(i+1, dir)
		if entry <= 0 {
			continue
		}
		stop, target := phaseT2Levels(dir, c, ind[i], state, entry, cfg)
		if stop <= 0 || target <= 0 || (dir > 0 && target <= entry) || (dir < 0 && target >= entry) {
			continue
		}
		distance := math.Abs(entry - stop)
		allocation := phaseT2Allocation(entry, distance, cfg)
		if allocation <= 0 {
			continue
		}
		pos = phaseT2Position{
			dir: dir, entryIdx: i + 1, entryTime: cs[i+1].OpenTime,
			entry: entry, stop: stop, target: target, allocation: allocation,
			state: matrixStateLabel(state), residual: 1,
		}
	}

	last := len(cs) - 1
	if pos.dir != 0 && pos.entry > 0 {
		exit := cs[last].Close
		if pos.dir > 0 {
			exit *= 1 - cfg.SlipBps/10000
		} else {
			exit *= 1 + cfg.SlipBps/10000
		}
		net := pos.partialNet + pos.residual*netReturn(pos.dir, pos.entry, exit, cfg)
		appendTrade(last, exit, pos.allocation*net, "data_end")
	}
	r.maxDD = symbolMaxDD(r.trades)
	return r
}

// phaseT2ManagedExit keeps the initial risk fixed and only tightens the stop
// after the first half is realized. All levels come from the current closed
// bar, so the next-open fill cannot use future candle information.
func phaseT2ManagedExit(pos phaseT2Position, c types.Candle, b indicatorBar, hold, maxHold int, cfg config) (string, bool, bool) {
	model := strings.ToLower(strings.TrimSpace(cfg.T2Exit))
	if model != "hybrid" {
		reason, trigger := phaseT2Exit(pos.dir, c, pos.stop, pos.target, hold, maxHold, cfg.T2StopOnClose)
		return reason, trigger, false
	}
	stop := pos.stop
	target := pos.target
	partial := false
	if pos.armed {
		if pos.dir > 0 {
			stop = math.Max(pos.entry, b.st)
			target = b.upper
		} else {
			stop = math.Min(pos.entry, b.st)
			target = b.lower
		}
	} else {
		partial = true
	}
	if pos.dir > 0 {
		if (cfg.T2StopOnClose && c.Close <= stop) || (!cfg.T2StopOnClose && c.Low <= stop) {
			return "sl", true, false
		}
		if target > pos.entry && c.High >= target {
			if partial {
				return "tp_mid_partial", true, true
			}
			return "tp_outer", true, false
		}
	} else {
		if (cfg.T2StopOnClose && c.Close >= stop) || (!cfg.T2StopOnClose && c.High >= stop) {
			return "sl", true, false
		}
		if target > 0 && target < pos.entry && c.Low <= target {
			if partial {
				return "tp_mid_partial", true, true
			}
			return "tp_outer", true, false
		}
	}
	if maxHold > 0 && hold >= maxHold {
		return "timeout", true, false
	}
	return "", false, false
}

func phaseT2Signal(c, prevC types.Candle, ind []indicatorBar, i int, cfg config) (int, matrixState) {
	if i < 1 || !ind[i].valid || !ind[i-1].valid || bandWidth(ind[i]) <= 0 {
		return 0, matrixState{}
	}
	b := ind[i]
	prev := ind[i-1]
	state := matrixStateAt(ind, i, cfg)
	if state.support <= 0 || state.resistance <= 0 {
		return 0, state
	}
	if family := strings.ToLower(strings.TrimSpace(cfg.T2Family)); family == "selected" || family == "paired" || family == "state" || family == "hybrid" {
		matrixFamily := family
		if family == "hybrid" {
			matrixFamily = "state"
		}
		if family == "paired" {
			matrixFamily = "state_paired"
		}
		dir, selectedState := matrixSignal(c, prevC, ind, i, matrixFamily, cfg)
		if strings.EqualFold(cfg.TSide, "long") && dir < 0 {
			dir = 0
		}
		if strings.EqualFold(cfg.TSide, "short") && dir > 0 {
			dir = 0
		}
		if dir != 0 || family != "hybrid" {
			return dir, selectedState
		}
	}

	// Longs use the lower band as support while ST remains inside Bollinger.
	long := b.dir == 1 && b.st >= b.lower && tRegimeAllowed(ind, i, 1, cfg.TRegime, cfg.TPhaseLB, cfg.TPhaseMin) &&
		c.Low <= b.lower && c.Close > b.lower && prevC.Low > prev.lower

	// Shorts require a bearish upper-band rejection with a meaningful upper
	// wick. ST direction is used as regime context, never as a signal alone.
	rng := c.High - c.Low
	strongUpper := rng > 0 && c.Close <= c.Open && (c.High-c.Close)/rng >= 0.60
	shortOrder := state.order == "lower<mid<st<upper" || state.order == "lower<st<mid<upper" || state.order == "lower<mid<upper<st"
	_, phaseDown := phaseBias(ind, i, cfg.TPhaseLB, cfg.TPhaseMin)
	short := b.dir == -1 && shortOrder && (state.zone == "mid_upper" || state.zone == "above_upper") &&
		phaseDown && c.High >= b.upper && c.Close < b.upper && c.Close > b.mid && strongUpper &&
		prevC.High < prev.upper
	if strings.EqualFold(cfg.TSide, "long") {
		short = false
	}
	if strings.EqualFold(cfg.TSide, "short") {
		long = false
	}
	if long == short {
		return 0, state
	}
	if long {
		return 1, state
	}
	return -1, state
}

func phaseT2Levels(dir int, c types.Candle, b indicatorBar, state matrixState, entry float64, cfg config) (float64, float64) {
	atr := b.atr
	if atr <= 0 {
		atr = bandWidth(b) / 4
	}
	if atr <= 0 {
		atr = entry * 0.01
	}
	stopATR := cfg.T2StopATR
	if stopATR <= 0 {
		stopATR = 1.5
	}
	buffer := 0.25 * atr
	if dir > 0 {
		stop := state.support - buffer
		configured := entry - stopATR*atr
		if stop <= 0 || stop >= entry {
			stop = configured
		} else if configured < stop {
			stop = configured
		}
		target := b.mid
		if strings.EqualFold(cfg.T2Exit, "outer") {
			target = b.upper
		}
		if cfg.T2TargetATR > 0 {
			target = entry + cfg.T2TargetATR*atr
		}
		return stop, target
	}
	stop := state.resistance + buffer
	configured := entry + stopATR*atr
	if stop <= entry || configured > stop {
		stop = configured
	}
	target := b.mid
	if strings.EqualFold(cfg.T2Exit, "outer") {
		target = b.lower
	}
	if cfg.T2TargetATR > 0 {
		target = entry - cfg.T2TargetATR*atr
	}
	return stop, target
}

func phaseT2Allocation(entry, distance float64, cfg config) float64 {
	if entry <= 0 || distance <= 0 {
		return 0
	}
	riskPct := cfg.T2RiskPct
	if riskPct <= 0 {
		riskPct = 0.005
	}
	maxAlloc := cfg.T2MaxAlloc
	if maxAlloc <= 0 || maxAlloc > 1 {
		maxAlloc = 0.25
	}
	allocation := riskPct / (distance / entry)
	if allocation > maxAlloc {
		allocation = maxAlloc
	}
	return allocation
}

func phaseT2Exit(dir int, c types.Candle, stop, target float64, hold, maxHold int, stopOnClose bool) (string, bool) {
	if dir > 0 {
		if (stopOnClose && c.Close <= stop) || (!stopOnClose && c.Low <= stop) {
			return "sl", true
		}
		if c.High >= target {
			return "tp_mid", true
		}
	} else {
		if (stopOnClose && c.Close >= stop) || (!stopOnClose && c.High >= stop) {
			return "sl", true
		}
		if c.Low <= target {
			return "tp_mid", true
		}
	}
	if maxHold > 0 && hold >= maxHold {
		return "timeout", true
	}
	return "", false
}

func phaseT2TargetLabel(cfg config) string {
	if cfg.T2TargetATR > 0 {
		return fmt.Sprintf("%.2f ATR", cfg.T2TargetATR)
	}
	if strings.EqualFold(cfg.T2Exit, "hybrid") {
		return "mid partial -> outer runner"
	}
	if strings.EqualFold(cfg.T2Exit, "outer") {
		return "Bollinger outer"
	}
	return "Bollinger mid"
}

func stateReactionSignal(c types.Candle, ind []indicatorBar, i int) int {
	if i < 4 || !ind[i].valid || bandWidth(ind[i]) <= 0 {
		return 0
	}
	b := ind[i]
	// The primary T trigger is a price rejection/reclaim at ST or the upper
	// Bollinger resistance. Both the price interaction and ST/Boll location are
	// required; ST direction alone cannot produce a signal.
	if signal := highFrequencySignal(c, ind, i); signal != 0 {
		if signal > 0 {
			return 0
		}
		return signal
	}
	// The remaining state observations are retained for research below, but
	// they are intentionally not promoted to live T entries until they pass
	// the same costed validation as the primary rejection/reclaim trigger.
	return 0
	upBias, downBias, rangeBias := stateBias(ind, i)
	width := bandWidth(b)
	flat := true
	for j := i - 2; j <= i; j++ {
		if math.Abs(ind[j].st-ind[j-1].st) > 0.25*width {
			flat = false
			break
		}
	}
	// Within a confirmed Bollinger trend, use the ST line as a moving support
	// or resistance only when price is still on the corresponding side of the
	// Bollinger midline. This is a pullback T, not an ST-only direction signal.
	if b.dir == 1 && c.Low <= b.st && c.Close > b.st && c.Close < b.upper && c.Close >= c.Open {
		return 1
	}
	if b.dir == -1 && ((c.High >= b.st && c.Close < b.st) || (c.High >= b.upper && c.Close < b.upper)) && c.Close > b.lower && c.Close <= c.Open {
		return -1
	}
	if flat && b.midUpAge >= 2 && b.midUpAge <= 6 && c.Close < b.mid && !downBias && b.dir == 1 {
		return 1
	}
	if flat && b.midDownAge >= 2 && b.midDownAge <= 6 && c.Close > b.mid && !upBias && b.dir == -1 {
		return -1
	}

	lowerTouch := c.Low <= b.lower && c.Close > b.lower
	upperTouch := c.High >= b.upper && c.Close < b.upper
	if lowerTouch && b.dir == 1 && (upBias || (rangeBias && b.lowerOuterAge > 24)) && !upperTouch {
		return 1
	}
	if upperTouch && b.dir == -1 && (downBias || (rangeBias && b.upperOuterAge > 24)) && !lowerTouch {
		return -1
	}
	return 0
}

func stateBias(ind []indicatorBar, i int) (up, down, flat bool) {
	const lookback = 6
	if i < lookback || !ind[i].valid || !ind[i-lookback].valid {
		return false, false, false
	}
	width := bandWidth(ind[i])
	if width <= 0 {
		return false, false, false
	}
	above, below := 0, 0
	for j := i - 5; j <= i; j++ {
		if ind[j].close >= ind[j].mid {
			above++
		}
		if ind[j].close <= ind[j].mid {
			below++
		}
	}
	up = ind[i].mid > ind[i-lookback].mid && above >= 4
	down = ind[i].mid < ind[i-lookback].mid && below >= 4
	flat = math.Abs(ind[i].mid-ind[i-lookback].mid)/width <= 0.20
	return up, down, flat
}

func stateBreakoutSignal(c types.Candle, ind []indicatorBar, i int, cfg config) int {
	if i < 3 || !ind[i].valid || !ind[i-1].valid || !ind[i-3].valid {
		return 0
	}
	if bandWidth(ind[i]) < bandWidth(ind[i-3]) {
		return 0
	}
	// A runner starts only while the recent Bollinger compression is still
	// active; an old breakout is treated as a late chase and left to T logic.
	if ind[i].squeezeAge > cfg.SqueezeW {
		return 0
	}
	if c.Close > ind[i].upper && ind[i-1].close <= ind[i-1].upper && ind[i].mid > ind[i-3].mid && ind[i].dir == 1 {
		return 1
	}
	if c.Close < ind[i].lower && ind[i-1].close >= ind[i-1].lower && ind[i].mid < ind[i-3].mid && ind[i].dir == -1 {
		return -1
	}
	return 0
}

func highFrequencySignal(c types.Candle, ind []indicatorBar, i int) int {
	if i < 1 || !ind[i].valid {
		return 0
	}
	b := ind[i]
	if bandWidth(b) <= 0 {
		return 0
	}
	// Keep the high-frequency ST reclaim and upper-band rejection signals. The
	// trend qualification is applied only to the runner so T frequency is not
	// sacrificed for a runner filter.
	long := b.dir == 1 && c.Low <= b.st && c.Close > b.st
	short := b.dir == -1 && ((c.High >= b.st && c.Close < b.st) || (c.High >= b.upper && c.Close < b.upper))
	if long && !short {
		return 1
	}
	if short && !long {
		return -1
	}
	return 0
}

func trendBreakoutSignal(c types.Candle, ind []indicatorBar, i, dir int, cfg config) bool {
	if i < 3 || !ind[i].valid || !ind[i-3].valid {
		return false
	}
	if !ind[i-1].valid {
		return false
	}
	widthNow := bandWidth(ind[i])
	widthPrev := bandWidth(ind[i-3])
	if widthNow <= 0 || widthPrev <= 0 || widthNow < widthPrev {
		return false
	}
	if cfg.EntryMode == "state" && ind[i].squeezeAge > cfg.SqueezeW && ind[i].lowerOuterAge > cfg.CrossW && ind[i].upperOuterAge > cfg.CrossW {
		return false
	}
	if dir > 0 {
		return ind[i].dir == 1 && ind[i].mid > ind[i-3].mid && c.Close > ind[i].upper && ind[i-1].close <= ind[i-1].upper
	}
	return ind[i].dir == -1 && ind[i].mid < ind[i-3].mid && c.Close < ind[i].lower && ind[i-1].close >= ind[i-1].lower
}

func makeLegTrade(sym, kind string, dir int, entryTime, exitTime int64, entry, exit float64, hold int, reason string, net float64) trade {
	return trade{sym: sym, kind: kind, dir: dir, entryTime: entryTime, exitTime: exitTime, entry: entry, exit: exit, net: net, hold: hold, reason: reason, tCycles: 1}
}

func netReturn(dir int, entry, exit float64, cfg config) float64 {
	if entry <= 0 || exit <= 0 {
		return -1
	}
	gross := float64(dir) * (exit/entry - 1)
	fees := cfg.FeeBps / 10000 * (entry + exit) / entry
	net := gross - fees
	if net < -1 {
		return -1
	}
	return net
}

func compoundLeg(profit, allocation, legNet float64) float64 {
	return (allocation+profit)*(1+legNet) - allocation
}

func finishHybridTrade(sym string, p hybridPosition) trade {
	net := p.tNet + p.rNet
	hold := p.tHold
	if p.rHold > hold {
		hold = p.rHold
	}
	if p.exitTime == 0 {
		p.exitTime = p.entryTime
	}
	exit := p.entry
	if p.dir != 0 {
		exit = p.entry * (1 + net/float64(p.dir))
	}
	return trade{
		sym:       sym,
		kind:      "composite",
		dir:       p.dir,
		entryTime: p.entryTime,
		exitTime:  p.exitTime,
		entry:     p.entry,
		exit:      exit,
		net:       net,
		hold:      hold,
		reason:    "t=" + p.tReason + ";runner=" + p.rReason,
		tCycles:   p.tCycles,
	}
}

func appendHybridReason(existing, next string) string {
	if existing == "" || existing == "open" {
		return next
	}
	return existing + "," + next
}

func hybridTReasonLong(c types.Candle, b indicatorBar) string {
	if c.Close >= b.mid {
		return "bb_mid"
	}
	if c.Close < b.lower {
		return "stop_bb_lower"
	}
	return "stop_ema"
}

func hybridTReasonShort(c types.Candle, b indicatorBar) string {
	if c.Close <= b.mid {
		return "bb_mid"
	}
	if c.Close > b.upper {
		return "stop_bb_upper"
	}
	return "stop_ema"
}

func hybridRunnerReasonLong(c types.Candle, b indicatorBar, hold int, cfg config) string {
	if c.Close < b.st {
		return "stop_st"
	}
	if cfg.RunnerExit == "st_ema" && c.Close < b.ema {
		return "stop_ema"
	}
	if hold >= cfg.MaxHold {
		return "timeout"
	}
	return "exit"
}

func hybridRunnerReasonShort(c types.Candle, b indicatorBar, hold int, cfg config) string {
	if c.Close > b.st {
		return "stop_st"
	}
	if cfg.RunnerExit == "st_ema" && c.Close > b.ema {
		return "stop_ema"
	}
	if hold >= cfg.MaxHold {
		return "timeout"
	}
	return "exit"
}

func longSignal(c types.Candle, b, prev indicatorBar, cfg config) bool {
	slopeOK := !cfg.EMASlope || b.ema > prev.ema
	if cfg.EntryMode == "cross_t" || cfg.EntryMode == "hybrid" {
		return b.dir == 1 && b.crossLongAge >= 0 && b.crossLongAge <= cfg.CrossW && b.st < b.lower && slopeOK && c.Close > b.ema && c.Low <= b.lower && c.Close > b.lower && c.Close >= c.Open
	}
	if cfg.EntryMode == "stboll" || cfg.EntryMode == "stboll_t" {
		return b.dir == 1 && b.crossLongAge >= 0 && b.crossLongAge <= cfg.CrossW && b.st < b.lower && b.squeezeAge <= cfg.SqueezeW && c.Low <= b.lower && c.Close > b.lower && c.Close >= c.Open
	}
	if cfg.EntryMode == "breakout" {
		return b.dir == 1 && slopeOK && b.close > b.ema && b.close > b.upper && prev.close > prev.upper && b.squeezeAge <= cfg.SqueezeW
	}
	return b.dir == 1 && slopeOK && c.Close > b.ema && c.Low <= b.lower && c.Close > b.lower && c.Close >= c.Open
}

func shortSignal(c types.Candle, b, prev indicatorBar, cfg config) bool {
	slopeOK := !cfg.EMASlope || b.ema < prev.ema
	if cfg.EntryMode == "cross_t" || cfg.EntryMode == "hybrid" {
		return b.dir == -1 && b.crossShortAge >= 0 && b.crossShortAge <= cfg.CrossW && b.st > b.upper && slopeOK && c.Close < b.ema && c.High >= b.upper && c.Close < b.upper && c.Close <= c.Open
	}
	if cfg.EntryMode == "stboll" || cfg.EntryMode == "stboll_t" {
		return b.dir == -1 && b.crossShortAge >= 0 && b.crossShortAge <= cfg.CrossW && b.st > b.upper && b.squeezeAge <= cfg.SqueezeW && c.High >= b.upper && c.Close < b.upper && c.Close <= c.Open
	}
	if cfg.EntryMode == "breakout" {
		return b.dir == -1 && slopeOK && b.close < b.ema && b.close < b.lower && prev.close < prev.lower && b.squeezeAge <= cfg.SqueezeW
	}
	return b.dir == -1 && slopeOK && c.Close < b.ema && c.High >= b.upper && c.Close < b.upper && c.Close <= c.Open
}

func exitLong(c types.Candle, b indicatorBar, hold int, cfg config) bool {
	if cfg.ExitMode == "mid_st" && c.Close >= b.mid {
		return true
	}
	if cfg.ExitMode == "opposite" && c.Close >= b.upper {
		return true
	}
	if cfg.ExitMode == "trend_st" {
		return c.Close < b.st || hold >= cfg.MaxHold
	}
	if cfg.ExitMode == "trend_ema" {
		return c.Close < b.ema || hold >= cfg.MaxHold
	}
	return c.Close < b.st || c.Close < b.ema || hold >= cfg.MaxHold
}

func exitShort(c types.Candle, b indicatorBar, hold int, cfg config) bool {
	if cfg.ExitMode == "mid_st" && c.Close <= b.mid {
		return true
	}
	if cfg.ExitMode == "opposite" && c.Close <= b.lower {
		return true
	}
	if cfg.ExitMode == "trend_st" {
		return c.Close > b.st || hold >= cfg.MaxHold
	}
	if cfg.ExitMode == "trend_ema" {
		return c.Close > b.ema || hold >= cfg.MaxHold
	}
	return c.Close > b.st || c.Close > b.ema || hold >= cfg.MaxHold
}

func exitReasonLong(c types.Candle, b indicatorBar, hold int, cfg config) string {
	if cfg.ExitMode == "mid_st" && c.Close >= b.mid {
		return "bb_mid"
	}
	if cfg.ExitMode == "opposite" && c.Close >= b.upper {
		return "bb_upper"
	}
	if c.Close < b.st {
		return "st_break"
	}
	if cfg.ExitMode == "trend_st" {
		if hold >= cfg.MaxHold {
			return "timeout"
		}
		return "exit"
	}
	if c.Close < b.ema {
		return "ema_break"
	}
	if hold >= cfg.MaxHold {
		return "timeout"
	}
	return "exit"
}

func exitReasonShort(c types.Candle, b indicatorBar, hold int, cfg config) string {
	if cfg.ExitMode == "mid_st" && c.Close <= b.mid {
		return "bb_mid"
	}
	if cfg.ExitMode == "opposite" && c.Close <= b.lower {
		return "bb_lower"
	}
	if c.Close > b.st {
		return "st_break"
	}
	if cfg.ExitMode == "trend_st" {
		if hold >= cfg.MaxHold {
			return "timeout"
		}
		return "exit"
	}
	if c.Close > b.ema {
		return "ema_break"
	}
	if hold >= cfg.MaxHold {
		return "timeout"
	}
	return "exit"
}

func makeTrade(sym string, dir int, entryTime, exitTime int64, entry, exit float64, hold int, reason string, cfg config) trade {
	net := netReturn(dir, entry, exit, cfg)
	return trade{sym: sym, kind: "composite", dir: dir, entryTime: entryTime, exitTime: exitTime, entry: entry, exit: exit, net: net, hold: hold, reason: reason, tCycles: 1}
}

func buildIndicators(cs []types.Candle, cfg config) []indicatorBar {
	closes := make([]float64, len(cs))
	for i := range cs {
		closes[i] = cs[i].Close
	}
	ema := emaSeries(closes, cfg.EMAPeriod)
	mid, upper, lower := bollSeries(closes, cfg.BBPeriod, cfg.BBK)
	stDir, st := supertrend(cs, cfg.STPeriod, cfg.STFactor)
	atr := atrSeries(cs, cfg.STPeriod)
	width := make([]float64, len(cs))
	for i := range cs {
		if mid[i] > 0 && upper[i] > lower[i] {
			width[i] = (upper[i] - lower[i]) / mid[i]
		}
	}
	squeezeAge := make([]int, len(cs))
	for i := range squeezeAge {
		squeezeAge[i] = len(cs) + 1
	}
	if cfg.SqueezeLB <= 0 {
		cfg.SqueezeLB = 125
	}
	for i := range cs {
		if i < cfg.SqueezeLB-1 || width[i] <= 0 {
			continue
		}
		minWidth := width[i]
		for j := i - cfg.SqueezeLB + 1; j <= i; j++ {
			if width[j] > 0 && width[j] < minWidth {
				minWidth = width[j]
			}
		}
		if width[i] <= minWidth*(1+1e-12) {
			squeezeAge[i] = 0
		} else if i > 0 && squeezeAge[i-1] < len(cs) {
			squeezeAge[i] = squeezeAge[i-1] + 1
		}
	}
	crossLongAge := make([]int, len(cs))
	crossShortAge := make([]int, len(cs))
	lowerOuterAge := make([]int, len(cs))
	upperOuterAge := make([]int, len(cs))
	midUpAge := make([]int, len(cs))
	midDownAge := make([]int, len(cs))
	for i := range cs {
		crossLongAge[i] = len(cs) + 1
		crossShortAge[i] = len(cs) + 1
		lowerOuterAge[i] = len(cs) + 1
		upperOuterAge[i] = len(cs) + 1
		midUpAge[i] = len(cs) + 1
		midDownAge[i] = len(cs) + 1
	}
	for i := 1; i < len(cs); i++ {
		if stDir[i] == 1 && st[i] > st[i-1] && st[i-1] >= lower[i-1] && st[i] < lower[i] {
			crossLongAge[i] = 0
		} else if crossLongAge[i-1] < len(cs) {
			crossLongAge[i] = crossLongAge[i-1] + 1
		}
		if stDir[i] == -1 && st[i] < st[i-1] && st[i-1] <= upper[i-1] && st[i] > upper[i] {
			crossShortAge[i] = 0
		} else if crossShortAge[i-1] < len(cs) {
			crossShortAge[i] = crossShortAge[i-1] + 1
		}
		if st[i-1] >= lower[i-1] && st[i] < lower[i] {
			lowerOuterAge[i] = 0
		} else if lowerOuterAge[i-1] < len(cs) {
			lowerOuterAge[i] = lowerOuterAge[i-1] + 1
		}
		if st[i-1] <= upper[i-1] && st[i] > upper[i] {
			upperOuterAge[i] = 0
		} else if upperOuterAge[i-1] < len(cs) {
			upperOuterAge[i] = upperOuterAge[i-1] + 1
		}
		if st[i-1] <= mid[i-1] && st[i] > mid[i] {
			midUpAge[i] = 0
		} else if midUpAge[i-1] < len(cs) {
			midUpAge[i] = midUpAge[i-1] + 1
		}
		if st[i-1] >= mid[i-1] && st[i] < mid[i] {
			midDownAge[i] = 0
		} else if midDownAge[i-1] < len(cs) {
			midDownAge[i] = midDownAge[i-1] + 1
		}
	}
	out := make([]indicatorBar, len(cs))
	for i := range cs {
		emaReady := cfg.EntryMode == "stboll" || cfg.EntryMode == "hfregime" || cfg.EntryMode == "state" || cfg.EntryMode == "phase_ttrend" || cfg.EntryMode == "t_matrix" || ema[i] > 0
		if emaReady && mid[i] > 0 && upper[i] > lower[i] && st[i] > 0 && stDir[i] != 0 {
			out[i] = indicatorBar{
				close: cs[i].Close, ema: ema[i], mid: mid[i], upper: upper[i], lower: lower[i], st: st[i], atr: atr[i], dir: stDir[i],
				squeezeAge: squeezeAge[i], crossLongAge: crossLongAge[i], crossShortAge: crossShortAge[i],
				lowerOuterAge: lowerOuterAge[i], upperOuterAge: upperOuterAge[i], midUpAge: midUpAge[i], midDownAge: midDownAge[i], valid: true,
			}
		}
	}
	return out
}

func buildMarketRegime(cs []types.Candle, cfg config, lookback int) map[int64]int {
	regime := make(map[int64]int)
	if lookback < 20 {
		lookback = 20
	}
	ind := buildIndicators(cs, cfg)
	for i := lookback + 6; i < len(cs); i++ {
		if !ind[i].valid || !ind[i-lookback].valid {
			continue
		}
		above, below := 0, 0
		for j := i - 6; j < i; j++ {
			if ind[j].close >= ind[j].mid {
				above++
			}
			if ind[j].close <= ind[j].mid {
				below++
			}
		}
		width := bandWidth(ind[i])
		if ind[i].mid > ind[i-lookback].mid && above >= 4 {
			regime[cs[i].OpenTime] = 1
		} else if ind[i].mid < ind[i-lookback].mid && below >= 4 {
			regime[cs[i].OpenTime] = -1
		} else if width > 0 && math.Abs(ind[i].mid-ind[i-lookback].mid)/width <= 0.20 {
			regime[cs[i].OpenTime] = 2
		} else {
			regime[cs[i].OpenTime] = 0
		}
	}
	return regime
}

func emaSeries(v []float64, n int) []float64 {
	out := make([]float64, len(v))
	if n <= 0 || len(v) < n {
		return out
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += v[i]
	}
	out[n-1] = sum / float64(n)
	a := 2 / float64(n+1)
	for i := n; i < len(v); i++ {
		out[i] = a*v[i] + (1-a)*out[i-1]
	}
	return out
}

func bollSeries(v []float64, n int, k float64) (mid, upper, lower []float64) {
	mid = make([]float64, len(v))
	upper = make([]float64, len(v))
	lower = make([]float64, len(v))
	if n <= 0 {
		return
	}
	for i := n - 1; i < len(v); i++ {
		sum := 0.0
		for j := i - n + 1; j <= i; j++ {
			sum += v[j]
		}
		m := sum / float64(n)
		var ss float64
		for j := i - n + 1; j <= i; j++ {
			d := v[j] - m
			ss += d * d
		}
		sd := math.Sqrt(ss / float64(n))
		mid[i], upper[i], lower[i] = m, m+k*sd, m-k*sd
	}
	return
}

func supertrend(cs []types.Candle, n int, factor float64) ([]int, []float64) {
	dir := make([]int, len(cs))
	st := make([]float64, len(cs))
	if n <= 0 || len(cs) <= n {
		return dir, st
	}
	atr := make([]float64, len(cs))
	tr := make([]float64, len(cs))
	for i := range cs {
		tr[i] = cs[i].High - cs[i].Low
		if i > 0 {
			tr[i] = math.Max(tr[i], math.Abs(cs[i].High-cs[i-1].Close))
			tr[i] = math.Max(tr[i], math.Abs(cs[i].Low-cs[i-1].Close))
		}
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += tr[i]
	}
	atr[n-1] = sum / float64(n)
	for i := n; i < len(cs); i++ {
		atr[i] = (atr[i-1]*float64(n-1) + tr[i]) / float64(n)
	}
	var finalUp, finalLo float64
	for i := n - 1; i < len(cs); i++ {
		mid := (cs[i].High + cs[i].Low) / 2
		basicUp, basicLo := mid+factor*atr[i], mid-factor*atr[i]
		if i == n-1 {
			finalUp, finalLo = basicUp, basicLo
			dir[i] = 1
		} else {
			prevClose := cs[i-1].Close
			prevUp, prevLo := finalUp, finalLo
			if basicUp < prevUp || prevClose > prevUp {
				finalUp = basicUp
			} else {
				finalUp = prevUp
			}
			if basicLo > prevLo || prevClose < prevLo {
				finalLo = basicLo
			} else {
				finalLo = prevLo
			}
			dir[i] = dir[i-1]
			if dir[i-1] == -1 && cs[i].Close > prevUp {
				dir[i] = 1
			} else if dir[i-1] == 1 && cs[i].Close < prevLo {
				dir[i] = -1
			}
		}
		if dir[i] == 1 {
			st[i] = finalLo
		} else {
			st[i] = finalUp
		}
	}
	return dir, st
}

func atrSeries(cs []types.Candle, n int) []float64 {
	out := make([]float64, len(cs))
	if n <= 0 || len(cs) < n {
		return out
	}
	tr := make([]float64, len(cs))
	for i := range cs {
		tr[i] = cs[i].High - cs[i].Low
		if i > 0 {
			tr[i] = math.Max(tr[i], math.Abs(cs[i].High-cs[i-1].Close))
			tr[i] = math.Max(tr[i], math.Abs(cs[i].Low-cs[i-1].Close))
		}
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += tr[i]
	}
	out[n-1] = sum / float64(n)
	for i := n; i < len(cs); i++ {
		out[i] = (out[i-1]*float64(n-1) + tr[i]) / float64(n)
	}
	return out
}

type tMatrixVariant struct {
	entryFamily string
	exitModel   string
}

type tMatrixSeries struct {
	sym string
	cs  []types.Candle
	ind []indicatorBar
}

type matrixLevel struct {
	name  string
	value float64
}

type matrixState struct {
	order          string
	zone           string
	supportName    string
	resistanceName string
	support        float64
	resistance     float64
	stSlope        string
	midSlope       string
	widthSlope     string
}

type matrixExit struct {
	trigger bool
	partial bool
	stop    bool
	reason  string
}

type matrixStats struct {
	trades  int
	wins    int
	longN   int
	longW   int
	shortN  int
	shortW  int
	sum     float64
	winSum  float64
	lossSum float64
	maxDD   float64
	avgHold float64
}

func runTMatrix(files []string, dataDir, interval string, start, end int64, minBars int, cfg config, symbolsFilter string, includeBTC bool) {
	selected := parseSymbolFilter(symbolsFilter)
	series := make([]tMatrixSeries, 0, len(files))
	for _, f := range files {
		sym := strings.TrimSuffix(filepath.Base(f), "_"+interval+".json")
		if strings.EqualFold(sym, "BTCUSDT") && !includeBTC {
			continue
		}
		if len(selected) > 0 && !selected[strings.ToUpper(sym)] {
			continue
		}
		cs, err := loadCandles(f)
		if err != nil {
			continue
		}
		cs = filterTime(cs, start, end)
		if len(cs) < minBars {
			continue
		}
		series = append(series, tMatrixSeries{sym: sym, cs: cs, ind: buildIndicators(cs, cfg)})
	}
	if len(series) == 0 {
		fatal("no usable symbols for t_matrix")
	}

	variants := make([]tMatrixVariant, 0, 24)
	for _, entryFamily := range []string{"touch", "confirm", "state", "state_robust", "state_paired", "selected"} {
		for _, exitModel := range []string{"mid", "outer", "atr", "hybrid"} {
			variants = append(variants, tMatrixVariant{entryFamily: entryFamily, exitModel: exitModel})
		}
	}

	var writer *csv.Writer
	var reportFile *os.File
	if cfg.MatrixReport != "" {
		var err error
		reportFile, err = os.Create(cfg.MatrixReport)
		if err != nil {
			fatal("create matrix report %s: %v", cfg.MatrixReport, err)
		}
		defer reportFile.Close()
		writer = csv.NewWriter(reportFile)
		_ = writer.Write([]string{"variant", "scope", "symbol", "state", "side", "trades", "wins", "win_pct", "expectancy", "pf", "avg_win", "avg_loss", "net", "max_dd", "avg_hold"})
	}

	fmt.Println("============================================================")
	fmt.Printf("T matrix research: data=%s | interval=%s | symbols=%d | entry families=touch,confirm,state,state_robust,state_paired,selected | exit models=mid,outer,atr,hybrid\n", dataDir, interval, len(series))
	fmt.Printf("Costs: fee=%.1fbps/side | slippage=%.1fbps/side | max_hold=%d | state_distance=%.2f Boll width | ATR stop=%.2f | ATR target=%.2f | reverse=%t | 1.0 allocation | next-open fills\n", cfg.FeeBps, cfg.SlipBps, cfg.MaxHold, cfg.MatrixTouchFrac, cfg.MatrixATRStop, cfg.MatrixATRTake, cfg.MatrixReverse)

	for _, variant := range variants {
		allTrades := make([]trade, 0)
		perSymbol := make([]symbolResult, 0, len(series))
		for _, data := range series {
			r := runTMatrixSymbol(data.sym, data.cs, data.ind, cfg, variant)
			allTrades = append(allTrades, r.trades...)
			perSymbol = append(perSymbol, r)
			if writer != nil {
				writeMatrixRows(writer, variant, r.sym, r.trades)
			}
		}
		printMatrixVariant(variant, allTrades, perSymbol)
		if writer != nil {
			writeMatrixRows(writer, variant, "__ALL__", allTrades)
		}
	}
	if writer != nil {
		writer.Flush()
		if err := writer.Error(); err != nil {
			fatal("write matrix report %s: %v", cfg.MatrixReport, err)
		}
		fmt.Printf("Matrix report: %s\n", cfg.MatrixReport)
	}
}

func runTMatrixSymbol(sym string, cs []types.Candle, ind []indicatorBar, cfg config, variant tMatrixVariant) symbolResult {
	r := symbolResult{sym: sym, bars: len(cs)}
	minWarm := cfg.BBPeriod
	if cfg.STPeriod > minWarm {
		minWarm = cfg.STPeriod
	}
	if minWarm < 3 {
		minWarm = 3
	}
	maxHold := cfg.MaxHold
	if maxHold < 1 {
		maxHold = 18
	}
	posDir := 0
	entryIdx := 0
	entryTime := int64(0)
	entryPrice := 0.0
	entryState := ""
	partialNet := 0.0
	residual := 1.0
	hybridArmed := false

	openPrice := func(i, dir int) float64 {
		if i < 0 || i >= len(cs) || cs[i].Open <= 0 {
			return 0
		}
		if dir > 0 {
			return cs[i].Open * (1 + cfg.SlipBps/10000)
		}
		return cs[i].Open * (1 - cfg.SlipBps/10000)
	}
	closeNet := func(dir int, exit float64) float64 {
		return partialNet + residual*netReturn(dir, entryPrice, exit, cfg)
	}
	appendTrade := func(exitIdx int, reason string, net float64) {
		if exitIdx < 0 || exitIdx >= len(cs) || net == 0 && entryPrice <= 0 {
			return
		}
		r.trades = append(r.trades, trade{
			sym: sym, dir: posDir, entryTime: entryTime, exitTime: cs[exitIdx].OpenTime,
			entry: entryPrice, exit: cs[exitIdx].Open, net: net, hold: exitIdx - entryIdx,
			reason: reason, tCycles: 1, kind: "t", family: variant.entryFamily, exitModel: variant.exitModel, state: entryState,
		})
	}

	for i := minWarm; i < len(cs)-1; i++ {
		if !ind[i].valid || cs[i+1].Open <= 0 {
			continue
		}
		b := ind[i]
		c := cs[i]
		closed := false

		if posDir != 0 {
			ev := matrixExitForBar(posDir, c, b, entryPrice, variant.exitModel, hybridArmed, entryIdx, i, maxHold, cfg)
			if ev.trigger {
				exit := openPrice(i+1, posDir)
				if exit > 0 {
					if ev.partial {
						partialNet += 0.5 * netReturn(posDir, entryPrice, exit, cfg)
						residual = 0.5
						hybridArmed = true
					} else {
						appendTrade(i+1, ev.reason, closeNet(posDir, exit))
						posDir = 0
						closed = true
					}
				}
			}
		}

		signalDir, state := matrixSignal(c, cs[i-1], ind, i, variant.entryFamily, cfg)
		if cfg.MatrixReverse && posDir != 0 && signalDir != 0 && signalDir != posDir && !closed {
			exit := openPrice(i+1, posDir)
			if exit > 0 {
				appendTrade(i+1, "reverse", closeNet(posDir, exit))
				posDir = 0
				closed = true
			}
		}
		if posDir == 0 && signalDir != 0 {
			entry := openPrice(i+1, signalDir)
			if entry > 0 {
				posDir = signalDir
				entryIdx = i + 1
				entryTime = cs[i+1].OpenTime
				entryPrice = entry
				entryState = matrixStateLabel(state)
				partialNet = 0
				residual = 1
				hybridArmed = false
			}
		}
	}

	last := len(cs) - 1
	if posDir != 0 && entryPrice > 0 {
		exit := cs[last].Close
		if posDir > 0 {
			exit *= 1 - cfg.SlipBps/10000
		} else {
			exit *= 1 + cfg.SlipBps/10000
		}
		net := closeNet(posDir, exit)
		r.trades = append(r.trades, trade{
			sym: sym, dir: posDir, entryTime: entryTime, exitTime: cs[last].CloseTime,
			entry: entryPrice, exit: exit, net: net, hold: last - entryIdx,
			reason: "data_end", tCycles: 1, kind: "t", family: variant.entryFamily, exitModel: variant.exitModel, state: entryState,
		})
	}
	r.maxDD = symbolMaxDD(r.trades)
	return r
}

func matrixSignal(c, prevC types.Candle, ind []indicatorBar, i int, family string, cfg config) (int, matrixState) {
	if i < 1 || !ind[i].valid {
		return 0, matrixState{}
	}
	state := matrixStateAt(ind, i, cfg)
	prevState := matrixStateAt(ind, i-1, cfg)
	width := bandWidth(ind[i])
	prevWidth := bandWidth(ind[i-1])
	if width <= 0 || state.support <= 0 || state.resistance <= 0 {
		return 0, state
	}
	frac := cfg.MatrixTouchFrac
	if frac <= 0 {
		frac = 0.35
	}
	long, short := false, false
	switch strings.ToLower(family) {
	case "touch":
		long = c.Close <= ind[i].mid && c.Low <= state.support && prevC.Low > prevState.support
		short = c.Close >= ind[i].mid && c.High >= state.resistance && prevC.High < prevState.resistance
	case "confirm":
		long = c.Close <= ind[i].mid && c.Low <= state.support && c.Close > state.support && prevC.Close > prevState.support
		short = c.Close >= ind[i].mid && c.High >= state.resistance && c.Close < state.resistance && prevC.Close < prevState.resistance
	case "state":
		nearSupport := math.Abs(c.Close-state.support) <= frac*width
		nearResistance := math.Abs(c.Close-state.resistance) <= frac*width
		wasNearSupport := prevWidth > 0 && math.Abs(prevC.Close-prevState.support) <= frac*prevWidth
		wasNearResistance := prevWidth > 0 && math.Abs(prevC.Close-prevState.resistance) <= frac*prevWidth
		long = c.Close <= ind[i].mid && nearSupport && !wasNearSupport
		short = c.Close >= ind[i].mid && nearResistance && !wasNearResistance
	case "state_robust":
		if matrixRobustLong(c, state, ind[i]) {
			if state.zone == "below_lower" {
				long = c.Low <= state.support && prevC.Low > prevState.support
			} else {
				long = c.Low <= state.support && c.Close > state.support && prevC.Low > prevState.support
			}
		}
		short = matrixRobustShort(c, state, ind[i]) && c.High >= state.resistance && c.Close < state.resistance && prevC.High < prevState.resistance
	case "state_paired":
		long = matrixPairedLong(state) && c.Close < ind[i].lower && c.Low <= state.support && prevC.Low > prevState.support
		short = matrixPairedShort(state) && c.High >= state.resistance && c.Close < state.resistance && prevC.High < prevState.resistance
	case "selected":
		long = matrixSelectedLong(state) && c.Close < ind[i].lower && c.Low <= state.support && prevC.Low > prevState.support
		short = matrixSelectedShort(state) && c.High >= state.resistance && c.Close < state.resistance && prevC.Close < prevState.resistance
	default:
		return 0, state
	}
	if long == short {
		return 0, state
	}
	if long {
		return 1, state
	}
	return -1, state
}

func matrixRobustLong(c types.Candle, state matrixState, b indicatorBar) bool {
	if c.Close > b.mid || state.supportName != "lower" || state.zone != "lower_mid" && state.zone != "below_lower" {
		return false
	}
	if state.order != "lower<st<mid<upper" && state.order != "lower<mid<st<upper" {
		return false
	}
	return state.widthSlope == "up" && (state.stSlope == "flat" || state.stSlope == "down") && (state.midSlope == "flat" || state.midSlope == "down")
}

func matrixRobustShort(c types.Candle, state matrixState, b indicatorBar) bool {
	if c.Close < b.mid || (state.resistanceName != "upper" && state.resistanceName != "st") || state.zone != "mid_upper" && state.zone != "above_upper" {
		return false
	}
	if state.order != "lower<mid<st<upper" && state.order != "st<lower<mid<upper" && state.order != "lower<st<mid<upper" {
		return false
	}
	return (state.stSlope == "up" || state.stSlope == "flat") && (state.midSlope == "down" || state.midSlope == "flat") && (state.widthSlope == "down" || state.widthSlope == "flat")
}

func matrixPairedLong(state matrixState) bool {
	return state.order == "lower<st<mid<upper" && state.zone == "below_lower" && state.supportName == "lower" && state.stSlope == "down" && state.midSlope == "down" && state.widthSlope == "up"
}

func matrixPairedShort(state matrixState) bool {
	return state.order == "lower<mid<st<upper" && state.zone == "mid_upper" && state.resistanceName == "st" && state.stSlope == "flat" && state.midSlope == "down" && state.widthSlope == "down"
}

// selected is the train/validation structural rule: the long leg is the
// lower-band rejection pattern; the short leg uses the ST or upper line as
// resistance while ST is flat/rising and the middle band is flat/falling.
func matrixSelectedLong(state matrixState) bool {
	return state.order == "lower<st<mid<upper" && state.zone == "below_lower" && state.supportName == "lower" && state.stSlope == "down" && state.midSlope == "down" && state.widthSlope == "up"
}

func matrixSelectedShort(state matrixState) bool {
	if state.zone != "mid_upper" {
		return false
	}
	// These are the four short-side line arrangements that stayed positive
	// in both the training and validation splits before the test was read.
	switch {
	case state.order == "lower<mid<st<upper" && state.resistanceName == "st" && state.stSlope == "flat" && state.midSlope == "down" && state.widthSlope == "down":
		return true
	case state.order == "lower<st<mid<upper" && state.resistanceName == "upper" && state.stSlope == "up" && state.midSlope == "flat" && state.widthSlope == "flat":
		return true
	case state.order == "lower<mid<st<upper" && state.resistanceName == "upper" && state.stSlope == "flat" && state.midSlope == "flat" && state.widthSlope == "down":
		return true
	default:
		return false
	}
}

func matrixStateAt(ind []indicatorBar, i int, cfg config) matrixState {
	b := ind[i]
	levels := []matrixLevel{{"lower", b.lower}, {"mid", b.mid}, {"upper", b.upper}, {"st", b.st}}
	sort.SliceStable(levels, func(a, z int) bool { return levels[a].value < levels[z].value })
	orderParts := make([]string, 0, len(levels))
	for _, l := range levels {
		orderParts = append(orderParts, l.name)
	}
	state := matrixState{order: strings.Join(orderParts, "<")}
	for _, l := range levels {
		if l.value <= b.close {
			if l.value >= state.support {
				state.supportName, state.support = l.name, l.value
			}
		}
		if l.value > b.close && (state.resistance == 0 || l.value < state.resistance) {
			state.resistanceName, state.resistance = l.name, l.value
		}
	}
	if state.support == 0 {
		state.supportName, state.support = levels[0].name, levels[0].value
	}
	if state.resistance == 0 {
		state.resistanceName, state.resistance = levels[len(levels)-1].name, levels[len(levels)-1].value
	}
	if b.close < b.lower {
		state.zone = "below_lower"
	} else if b.close <= b.mid {
		state.zone = "lower_mid"
	} else if b.close <= b.upper {
		state.zone = "mid_upper"
	} else {
		state.zone = "above_upper"
	}
	width := bandWidth(b)
	prev := b
	if i > 0 && ind[i-1].valid {
		prev = ind[i-1]
	}
	state.stSlope = matrixSlopeBucket(b.st-prev.st, width)
	state.midSlope = matrixSlopeBucket(b.mid-prev.mid, width)
	state.widthSlope = matrixSlopeBucket(bandWidth(b)-bandWidth(prev), width)
	return state
}

func matrixSlopeBucket(delta, width float64) string {
	threshold := width * 0.02
	if threshold <= 0 {
		threshold = 1e-12
	}
	if delta > threshold {
		return "up"
	}
	if delta < -threshold {
		return "down"
	}
	return "flat"
}

func matrixStateLabel(s matrixState) string {
	return fmt.Sprintf("order=%s;zone=%s;support=%s;resistance=%s;st=%s;mid=%s;width=%s", s.order, s.zone, s.supportName, s.resistanceName, s.stSlope, s.midSlope, s.widthSlope)
}

func matrixExitForBar(dir int, c types.Candle, b indicatorBar, entry float64, model string, hybridArmed bool, entryIdx, i, maxHold int, cfg config) matrixExit {
	atr := b.atr
	if atr <= 0 {
		atr = bandWidth(b) / 4
	}
	if atr <= 0 {
		atr = entry * 0.01
	}
	buffer := 0.25 * atr
	stop, target := 0.0, 0.0
	if dir > 0 {
		stop = b.lower - buffer
		switch model {
		case "mid":
			target = b.mid
		case "outer":
			target = b.upper
		case "atr":
			stop = entry - cfg.MatrixATRStop*atr
			target = entry + cfg.MatrixATRTake*atr
		case "hybrid":
			if hybridArmed {
				stop = math.Max(entry, b.st)
				target = b.upper
			} else {
				target = b.mid
			}
		}
		if stop > 0 && c.Low <= stop {
			return matrixExit{trigger: true, stop: true, reason: "sl"}
		}
		if model == "hybrid" && !hybridArmed && target > entry && c.High >= target {
			return matrixExit{trigger: true, partial: true, reason: "tp_mid_partial"}
		}
		if target > entry && c.High >= target {
			return matrixExit{trigger: true, reason: "tp"}
		}
	} else {
		stop = b.upper + buffer
		switch model {
		case "mid":
			target = b.mid
		case "outer":
			target = b.lower
		case "atr":
			stop = entry + cfg.MatrixATRStop*atr
			target = entry - cfg.MatrixATRTake*atr
		case "hybrid":
			if hybridArmed {
				stop = math.Min(entry, b.st)
				target = b.lower
			} else {
				target = b.mid
			}
		}
		if stop > 0 && c.High >= stop {
			return matrixExit{trigger: true, stop: true, reason: "sl"}
		}
		if model == "hybrid" && !hybridArmed && target < entry && c.Low <= target {
			return matrixExit{trigger: true, partial: true, reason: "tp_mid_partial"}
		}
		if target > 0 && target < entry && c.Low <= target {
			return matrixExit{trigger: true, reason: "tp"}
		}
	}
	if maxHold > 0 && i-entryIdx >= maxHold {
		return matrixExit{trigger: true, reason: "timeout"}
	}
	return matrixExit{}
}

func writeMatrixRows(w *csv.Writer, variant tMatrixVariant, symbol string, trades []trade) {
	groups := map[string][]trade{"all": trades}
	for _, t := range trades {
		side := "short"
		if t.dir > 0 {
			side = "long"
		}
		groups["state="+t.state+";side="+side] = append(groups["state="+t.state+";side="+side], t)
		if t.dir > 0 {
			groups["long"] = append(groups["long"], t)
		} else {
			groups["short"] = append(groups["short"], t)
		}
	}
	for scope, ts := range groups {
		st := matrixStatsFor(ts)
		_ = w.Write([]string{
			variant.entryFamily + "/" + variant.exitModel, scope, symbol, scope,
			"", fmt.Sprint(st.trades), fmt.Sprint(st.wins), fmt.Sprintf("%.6f", pct(st.wins, st.trades)),
			fmt.Sprintf("%.8f", st.expectancy()), fmt.Sprintf("%.6f", st.pf()), fmt.Sprintf("%.8f", st.avgWin()),
			fmt.Sprintf("%.8f", st.avgLoss()), fmt.Sprintf("%.8f", st.sum), fmt.Sprintf("%.8f", st.maxDD), fmt.Sprintf("%.4f", st.avgHold),
		})
	}
}

func matrixStatsFor(trades []trade) matrixStats {
	st := matrixStats{}
	for _, t := range trades {
		st.trades++
		st.sum += t.net
		st.avgHold += float64(t.hold)
		if t.dir > 0 {
			st.longN++
		} else {
			st.shortN++
		}
		if t.net > 0 {
			st.wins++
			st.winSum += t.net
			if t.dir > 0 {
				st.longW++
			} else {
				st.shortW++
			}
		} else if t.net < 0 {
			st.lossSum -= t.net
		}
	}
	if st.trades > 0 {
		st.avgHold /= float64(st.trades)
	}
	st.maxDD = symbolMaxDD(trades)
	return st
}

func (s matrixStats) expectancy() float64 {
	if s.trades == 0 {
		return 0
	}
	return s.sum / float64(s.trades)
}

func (s matrixStats) pf() float64 {
	if s.lossSum == 0 {
		return 0
	}
	return s.winSum / s.lossSum
}

func (s matrixStats) avgWin() float64 {
	if s.wins == 0 {
		return 0
	}
	return s.winSum / float64(s.wins)
}

func (s matrixStats) avgLoss() float64 {
	losses := s.trades - s.wins
	if losses <= 0 {
		return 0
	}
	return s.lossSum / float64(losses)
}

func printMatrixVariant(variant tMatrixVariant, trades []trade, perSymbol []symbolResult) {
	all := matrixStatsFor(trades)
	long := make([]trade, 0)
	short := make([]trade, 0)
	byState := make(map[string][]trade)
	for _, t := range trades {
		if t.dir > 0 {
			long = append(long, t)
		} else {
			short = append(short, t)
		}
		byState[t.state] = append(byState[t.state], t)
	}
	positive := 0
	for _, r := range perSymbol {
		if matrixStatsFor(r.trades).sum > 0 {
			positive++
		}
	}
	fmt.Println("------------------------------------------------------------")
	fmt.Printf("Variant: entry=%s | exit=%s\n", variant.entryFamily, variant.exitModel)
	fmt.Printf("All: trades=%d | win=%.2f%% | expectancy=%+.4f%% | PF=%.2f | avg win/loss=%.4f%%/%.4f%% | max DD=%.2f%% | positive symbols=%d/%d\n", all.trades, pct(all.wins, all.trades), all.expectancy()*100, all.pf(), all.avgWin()*100, all.avgLoss()*100, all.maxDD*100, positive, len(perSymbol))
	for name, ts := range map[string][]trade{"long": long, "short": short} {
		st := matrixStatsFor(ts)
		fmt.Printf("  %s: trades=%d | win=%.2f%% | expectancy=%+.4f%% | PF=%.2f | avg win/loss=%.4f%%/%.4f%% | max DD=%.2f%%\n", name, st.trades, pct(st.wins, st.trades), st.expectancy()*100, st.pf(), st.avgWin()*100, st.avgLoss()*100, st.maxDD*100)
	}
	keys := make([]string, 0, len(byState))
	for k := range byState {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Printf("  states=%d\n", len(keys))
	for _, key := range keys {
		st := matrixStatsFor(byState[key])
		fmt.Printf("    %s | n=%d | win=%.2f%% | exp=%+.4f%% | PF=%.2f\n", key, st.trades, pct(st.wins, st.trades), st.expectancy()*100, st.pf())
	}
}

func printSummary(s summary, cfg config, dataDir, interval string) {
	sort.Slice(s.trades, func(i, j int) bool { return s.trades[i].exitTime < s.trades[j].exitTime })
	wins, losses := 0, 0
	var sum, winSum, lossSum, longSum, shortSum float64
	longN, shortN, longW, shortW := 0, 0, 0, 0
	tCycles := 0
	tLegs, runnerLegs, compositeEntries := 0, 0, 0
	holds := make([]float64, 0, len(s.trades))
	for _, t := range s.trades {
		switch t.kind {
		case "t":
			tLegs++
		case "runner":
			runnerLegs++
		default:
			compositeEntries++
		}
		if t.tCycles > 0 {
			tCycles += t.tCycles
		} else {
			tCycles++
		}
		sum += t.net
		holds = append(holds, float64(t.hold))
		if t.dir > 0 {
			longN++
			longSum += t.net
			if t.net > 0 {
				longW++
			}
		} else {
			shortN++
			shortSum += t.net
			if t.net > 0 {
				shortW++
			}
		}
		if t.net > 0 {
			wins++
			winSum += t.net
		} else if t.net < 0 {
			losses++
			lossSum += -t.net
		}
	}
	avg := 0.0
	if len(s.trades) > 0 {
		avg = sum / float64(len(s.trades))
	}
	pf := 0.0
	if lossSum > 0 {
		pf = winSum / lossSum
	}
	maxDD := 0.0
	positiveSymbols := 0
	medianFinal := make([]float64, 0, len(s.perSymbol))
	for _, r := range s.perSymbol {
		if r.maxDD > maxDD {
			maxDD = r.maxDD
		}
		if len(r.trades) > 0 {
			eq := 1.0
			for _, t := range r.trades {
				eq *= 1 + t.net
			}
			medianFinal = append(medianFinal, eq-1)
			if eq > 1 {
				positiveSymbols++
			}
		}
	}
	sort.Float64s(medianFinal)
	median := 0.0
	if len(medianFinal) > 0 {
		median = medianFinal[len(medianFinal)/2]
	}
	fmt.Println("============================================================")
	fmt.Printf("Strategy: %s\n", s.name)
	fmt.Printf("Data: %s | interval=%s | symbols=%d | bars=%d\n", dataDir, interval, s.symbols, s.bars)
	if cfg.EntryMode == "phase_ttrend" {
		phaseSource := "local phase"
		if cfg.MarketRegime != nil {
			phaseSource = "BTC phase"
		}
		fmt.Printf("Rules: %s filter LB=%d; T=%s phase pullback, target=%.0f%% of signal-to-mid move, min_move=%.2f%%, stop_line=%s, hard_stop=%.1f%%; trend=two-bar outer move then %s retest/reclaim + ST confirmation, runner_exit=%s, hard_stop=%.1f%%, trail=%.1f%%; T weight=%.2f; gap=%d bars; T max_hold=%d; closed-bar signal -> next open; fee=%.1fbps/side; slippage=%.1fbps/side; max_hold=%d; EMA=false\n", phaseSource, cfg.PhaseLB, cfg.TSide, cfg.TTargetFrac*100, cfg.TMinMovePct, cfg.TStopMode, cfg.TStopPct, cfg.RunnerSetup, cfg.RunnerExit, cfg.RunnerStopPct, cfg.RunnerTrailPct, cfg.TWeight, cfg.TGap, cfg.THold, cfg.FeeBps, cfg.SlipBps, cfg.MaxHold)
	} else if cfg.EntryMode == "hfregime" {
		fmt.Printf("Rules: T=ST reclaim or Boll upper rejection; runner=mid-slope + width-expansion Boll breakout; T target=mid; virtual ST gate; T gap=%d bars; T max_hold=%d; closed-bar signal -> next open; fee=%.1fbps/side; slippage=%.1fbps/side; max_hold=%d; EMA=false\n", cfg.TGap, cfg.THold, cfg.FeeBps, cfg.SlipBps, cfg.MaxHold)
	} else if cfg.EntryMode == "state" {
		fmt.Printf("Rules: phase-locked ST/Boll rejection T, side=%s, regime=%s, phase_lb=%d, phase_min=%d; T allocation=%.2f, runner weight=%.2f; runner=fresh-compression + mid-slope + width-expansion Boll breakout; T target=mid; T stop=%s; runner_exit=%s, runner_hard_stop=%.1f%%, runner_trail=%.1f%%; T gap=%d bars; T max_hold=%d; closed-bar signal -> next open; fee=%.1fbps/side; slippage=%.1fbps/side; max_hold=%d; EMA=false\n", cfg.TSide, cfg.TRegime, cfg.TPhaseLB, cfg.TPhaseMin, cfg.TAlloc, cfg.TWeight, cfg.TStopMode, cfg.RunnerExit, cfg.RunnerStopPct, cfg.RunnerTrailPct, cfg.TGap, cfg.THold, cfg.FeeBps, cfg.SlipBps, cfg.MaxHold)
	} else if cfg.EntryMode == "phase_t2" {
		stopMode := "intrabar"
		if cfg.T2StopOnClose {
			stopMode = "close"
		}
		fmt.Printf("Rules: T2 family=%s (ST/Boll support-resistance state); minimum stop=%.2f ATR plus structural buffer (%s); target=%s; risk=%.3f%%; max allocation=%.2f; gap=%d bars; max_hold=%d; closed-bar signal -> next-open fill; fee=%.1fbps/side; slippage=%.1fbps/side; EMA=false\n", cfg.T2Family, cfg.T2StopATR, stopMode, phaseT2TargetLabel(cfg), cfg.T2RiskPct*100, cfg.T2MaxAlloc, cfg.T2Gap, cfg.T2Hold, cfg.FeeBps, cfg.SlipBps)
	} else {
		fmt.Printf("Rules: entry=%s; closed-bar signal -> next open; squeeze_lb=%d window=%d; cross_window=%d; exit=%s; runner_exit=%s; t_weight=%.2f; ema_filter=%t; fee=%.1fbps/side; slippage=%.1fbps/side; max_hold=%d\n", cfg.EntryMode, cfg.SqueezeLB, cfg.SqueezeW, cfg.CrossW, cfg.ExitMode, cfg.RunnerExit, cfg.TWeight, cfg.EntryMode != "stboll", cfg.FeeBps, cfg.SlipBps, cfg.MaxHold)
	}
	if cfg.EntryMode == "hfregime" || cfg.EntryMode == "state" || cfg.EntryMode == "phase_ttrend" || cfg.EntryMode == "phase_t2" {
		fmt.Printf("Trades: %d actual legs | T=%d runner=%d | long=%d short=%d | win=%.1f%% (%d/%d)\n", len(s.trades), tLegs, runnerLegs, longN, shortN, pct(wins, len(s.trades)), wins, len(s.trades))
	} else {
		fmt.Printf("Trades: %d composite entries | T cycles=%d | long=%d short=%d | win=%.1f%% (%d/%d)\n", compositeEntries, tCycles, longN, shortN, pct(wins, len(s.trades)), wins, len(s.trades))
	}
	fmt.Printf("Net expectancy: %+.4f%%/trade | median hold: %.1f bars | PF: %.2f\n", avg*100, medianFloat(holds), pf)
	fmt.Printf("Avg win/loss: %.4f%% / %.4f%% | long win=%.1f%% | short win=%.1f%%\n", pctFloat(winSum, wins)*100, pctFloat(lossSum, losses)*100, pct(longW, longN), pct(shortW, shortN))
	fmt.Printf("Side expectancy: long=%+.4f%% (%d) | short=%+.4f%% (%d)\n", pctFloat(longSum, longN)*100, longN, pctFloat(shortSum, shortN)*100, shortN)
	if cfg.EntryMode == "hfregime" || cfg.EntryMode == "state" || cfg.EntryMode == "phase_ttrend" || cfg.EntryMode == "phase_t2" {
		printLegStats("T", s.trades, "t")
		printLegStats("runner", s.trades, "runner")
	}
	fmt.Printf("Max symbol compounded DD: %.2f%% | positive symbols: %d/%d | median symbol return: %+.2f%%\n", maxDD*100, positiveSymbols, len(s.perSymbol), median*100)
	fmt.Printf("Portfolio compounded DD: %.2f%%\n", symbolMaxDD(s.trades)*100)
	months := make([]string, 0, len(s.monthlyNet))
	goodMonths := 0
	for m := range s.monthlyNet {
		months = append(months, m)
	}
	sort.Strings(months)
	for _, m := range months {
		if s.monthlyNet[m] > 0 {
			goodMonths++
		}
	}
	fmt.Printf("Positive months: %d/%d | monthly net sum: %+.2f%%\n", goodMonths, len(months), sumMap(s.monthlyNet)*100)
	fmt.Println("Exit reasons:", reasonCounts(s.trades))
}

func printLegStats(label string, trades []trade, kind string) {
	count, wins := 0, 0
	var sum, winSum, lossSum float64
	holds := make([]float64, 0)
	bySymbol := make(map[string][]trade)
	for _, t := range trades {
		if t.kind != kind {
			continue
		}
		count++
		sum += t.net
		holds = append(holds, float64(t.hold))
		bySymbol[t.sym] = append(bySymbol[t.sym], t)
		if t.net > 0 {
			wins++
			winSum += t.net
		} else if t.net < 0 {
			lossSum -= t.net
		}
	}
	pf := 0.0
	if lossSum > 0 {
		pf = winSum / lossSum
	}
	avg := 0.0
	if count > 0 {
		avg = sum / float64(count)
	}
	maxDD := 0.0
	for _, ts := range bySymbol {
		if dd := symbolMaxDD(ts); dd > maxDD {
			maxDD = dd
		}
	}
	avgWin, avgLoss := 0.0, 0.0
	if wins > 0 {
		avgWin = winSum / float64(wins)
	}
	losses := count - wins
	if losses > 0 {
		avgLoss = lossSum / float64(losses)
	}
	fmt.Printf("  %s legs: %d | win=%.1f%% | expectancy=%+.4f%% | PF=%.2f | avg win/loss=%.4f%%/%.4f%% | median hold=%.1f bars | max symbol DD=%.2f%%\n", label, count, pct(wins, count), avg*100, pf, avgWin*100, avgLoss*100, medianFloat(holds), maxDD*100)
}

func reasonCounts(ts []trade) map[string]int {
	out := make(map[string]int)
	for _, t := range ts {
		out[t.reason]++
	}
	return out
}

func symbolMaxDD(ts []trade) float64 {
	ordered := append([]trade(nil), ts...)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].exitTime == ordered[j].exitTime {
			return ordered[i].entryTime < ordered[j].entryTime
		}
		return ordered[i].exitTime < ordered[j].exitTime
	})
	eq, peak, dd := 1.0, 1.0, 0.0
	for _, t := range ordered {
		eq *= 1 + t.net
		if eq > peak {
			peak = eq
		}
		if eq <= 0 {
			return 1
		}
		if x := (peak - eq) / peak; x > dd {
			dd = x
		}
	}
	return dd
}

func loadCandles(path string) ([]types.Candle, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var arr []types.Candle
	if err := json.Unmarshal(b, &arr); err == nil {
		sort.Slice(arr, func(i, j int) bool { return arr[i].OpenTime < arr[j].OpenTime })
		return arr, nil
	}
	var w struct {
		Candles []types.Candle `json:"candles"`
	}
	if err := json.Unmarshal(b, &w); err != nil {
		return nil, err
	}
	sort.Slice(w.Candles, func(i, j int) bool { return w.Candles[i].OpenTime < w.Candles[j].OpenTime })
	return w.Candles, nil
}

func filterTime(cs []types.Candle, start, end int64) []types.Candle {
	if start == 0 && end == 0 {
		return cs
	}
	out := make([]types.Candle, 0, len(cs))
	for _, c := range cs {
		if start > 0 && c.OpenTime < start {
			continue
		}
		if end > 0 && c.OpenTime > end {
			continue
		}
		out = append(out, c)
	}
	return out
}

func parseDate(s string) int64 {
	t, err := time.ParseInLocation("2006-01-02", s, time.UTC)
	if err != nil {
		fatal("bad date %s: %v", s, err)
	}
	return t.UnixMilli()
}

func pct(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d) * 100
}

func pctFloat(sum float64, n int) float64 {
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

func medianFloat(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	sort.Float64s(v)
	return v[len(v)/2]
}

func sumMap(m map[string]float64) float64 {
	var out float64
	for _, v := range m {
		out += v
	}
	return out
}

type hfResearchEvent struct {
	dir     int
	returns [3]float64
}

func parseSymbolFilter(raw string) map[string]bool {
	out := make(map[string]bool)
	for _, token := range strings.Split(raw, ",") {
		token = strings.ToUpper(strings.TrimSpace(token))
		if token == "" {
			continue
		}
		if !strings.HasSuffix(token, "USDT") {
			token += "USDT"
		}
		out[token] = true
	}
	return out
}

func analysisSymbolAllowed(sym string, selected map[string]bool, includeBTC bool) bool {
	if strings.EqualFold(sym, "BTCUSDT") && !includeBTC {
		return false
	}
	if len(selected) > 0 && !selected[strings.ToUpper(sym)] {
		return false
	}
	return true
}

func analyzeHighFrequencyFilters(files []string, dataDir, interval string, start, end int64, minBars int, cfg config) {
	type bucket struct {
		name string
		fn   func(types.Candle, []indicatorBar, int) bool
		dir  int
	}
	buckets := []bucket{
		{name: "long_lower_reclaim", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return ind[i].dir == 1 && c.Low <= ind[i].lower && c.Close > ind[i].lower
		}},
		{name: "long_st_inside", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return ind[i].dir == 1 && ind[i].st >= ind[i].lower && c.Low <= ind[i].lower && c.Close > ind[i].lower
		}},
		{name: "long_st_outside", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return ind[i].dir == 1 && ind[i].st < ind[i].lower && c.Low <= ind[i].lower && c.Close > ind[i].lower
		}},
		{name: "long_mid_rising", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return i >= 3 && ind[i].dir == 1 && ind[i].mid > ind[i-3].mid && c.Low <= ind[i].lower && c.Close > ind[i].lower
		}},
		{name: "long_width_expanding", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return i >= 3 && ind[i].dir == 1 && bandWidth(ind[i]) >= bandWidth(ind[i-3]) && c.Low <= ind[i].lower && c.Close > ind[i].lower
		}},
		{name: "long_inside_strong", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			rng := c.High - c.Low
			return rng > 0 && ind[i].dir == 1 && ind[i].st >= ind[i].lower && c.Low <= ind[i].lower && c.Close > ind[i].lower && c.Close >= c.Open && (c.Close-c.Low)/rng >= 0.60
		}},
		{name: "short_upper_reject", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return ind[i].dir == -1 && c.High >= ind[i].upper && c.Close < ind[i].upper
		}},
		{name: "short_upper_strong", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			rng := c.High - c.Low
			return rng > 0 && ind[i].dir == -1 && c.High >= ind[i].upper && c.Close < ind[i].upper && c.Close <= c.Open && (c.High-c.Close)/rng >= 0.60
		}},
		{name: "short_upper_below_mid", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return ind[i].dir == -1 && c.High >= ind[i].upper && c.Close < ind[i].upper && c.Close > ind[i].mid
		}},
		{name: "short_st_inside", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return ind[i].dir == -1 && ind[i].st <= ind[i].upper && c.High >= ind[i].upper && c.Close < ind[i].upper
		}},
		{name: "short_st_outside", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return ind[i].dir == -1 && ind[i].st > ind[i].upper && c.High >= ind[i].upper && c.Close < ind[i].upper
		}},
		{name: "short_mid_falling", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return i >= 3 && ind[i].dir == -1 && ind[i].mid < ind[i-3].mid && c.High >= ind[i].upper && c.Close < ind[i].upper
		}},
		{name: "short_width_expanding", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return i >= 3 && ind[i].dir == -1 && bandWidth(ind[i]) >= bandWidth(ind[i-3]) && c.High >= ind[i].upper && c.Close < ind[i].upper
		}},
	}
	byName := make(map[string][]hfResearchEvent)
	symbols, bars := 0, 0
	for _, f := range files {
		sym := strings.TrimSuffix(filepath.Base(f), "_"+interval+".json")
		if strings.EqualFold(sym, "BTCUSDT") {
			continue
		}
		cs, err := loadCandles(f)
		if err != nil || len(cs) < minBars {
			continue
		}
		cs = filterTime(cs, start, end)
		if len(cs) < minBars {
			continue
		}
		ind := buildIndicators(cs, cfg)
		symbols++
		bars += len(cs)
		for i := 3; i+24 < len(cs)-1; i++ {
			if !ind[i].valid || bandWidth(ind[i]) <= 0 || cs[i+1].Open <= 0 {
				continue
			}
			for _, b := range buckets {
				if b.fn(cs[i], ind, i) {
					e := hfResearchEvent{dir: b.dir}
					for k, h := range reactionHorizons {
						e.returns[k] = float64(b.dir) * (cs[i+h].Close/cs[i+1].Open - 1)
					}
					byName[b.name] = append(byName[b.name], e)
				}
			}
		}
	}
	fmt.Println("============================================================")
	fmt.Printf("HF ST/Boll filter analysis: data=%s interval=%s symbols=%d bars=%d\n", dataDir, interval, symbols, bars)
	for _, b := range buckets {
		printHFResearch(b.name, byName[b.name])
	}
}

func analyzeStateCombinations(files []string, dataDir, interval string, start, end int64, minBars int, cfg config, symbolsFilter string, includeBTC bool) {
	type bucket struct {
		name string
		dir  int
		fn   func(types.Candle, []indicatorBar, int) bool
	}
	flatST := func(ind []indicatorBar, i int) bool {
		if i < 3 || bandWidth(ind[i]) <= 0 {
			return false
		}
		for j := i - 2; j <= i; j++ {
			if math.Abs(ind[j].st-ind[j-1].st) > 0.25*bandWidth(ind[i]) {
				return false
			}
		}
		return true
	}
	buckets := []bucket{
		{name: "lower_reclaim_price_below_mid", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return c.Low <= ind[i].lower && c.Close > ind[i].lower && c.Close < ind[i].mid
		}},
		{name: "lower_reclaim_st_inside", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return c.Low <= ind[i].lower && c.Close > ind[i].lower && ind[i].st >= ind[i].lower
		}},
		{name: "lower_reclaim_st_outside", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return c.Low <= ind[i].lower && c.Close > ind[i].lower && ind[i].st < ind[i].lower
		}},
		{name: "lower_reclaim_outer_cross24", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return c.Low <= ind[i].lower && c.Close > ind[i].lower && ind[i].lowerOuterAge <= 24
		}},
		{name: "upper_reject_price_above_mid", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return c.High >= ind[i].upper && c.Close < ind[i].upper && c.Close > ind[i].mid
		}},
		{name: "upper_reject_st_inside", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return c.High >= ind[i].upper && c.Close < ind[i].upper && ind[i].st <= ind[i].upper
		}},
		{name: "upper_reject_st_outside", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return c.High >= ind[i].upper && c.Close < ind[i].upper && ind[i].st > ind[i].upper
		}},
		{name: "upper_reject_outer_cross24", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return c.High >= ind[i].upper && c.Close < ind[i].upper && ind[i].upperOuterAge <= 24
		}},
		{name: "lower_reclaim_mid_rising", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return i >= 3 && c.Low <= ind[i].lower && c.Close > ind[i].lower && ind[i].mid > ind[i-3].mid
		}},
		{name: "lower_reclaim_mid_falling", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return i >= 3 && c.Low <= ind[i].lower && c.Close > ind[i].lower && ind[i].mid < ind[i-3].mid
		}},
		{name: "upper_reject_mid_falling", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return i >= 3 && c.High >= ind[i].upper && c.Close < ind[i].upper && ind[i].mid < ind[i-3].mid
		}},
		{name: "upper_reject_mid_rising", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return i >= 3 && c.High >= ind[i].upper && c.Close < ind[i].upper && ind[i].mid > ind[i-3].mid
		}},
		{name: "upper_outer_cross_down_bias", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			up, down, _ := stateBias(ind, i)
			return !up && down && c.High >= ind[i].upper && c.Close < ind[i].upper && ind[i].upperOuterAge <= 24
		}},
		{name: "upper_outer_cross_up_bias", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			up, down, _ := stateBias(ind, i)
			return up && !down && c.High >= ind[i].upper && c.Close < ind[i].upper && ind[i].upperOuterAge <= 24
		}},
		{name: "lower_reclaim_up_bias", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			up, down, _ := stateBias(ind, i)
			return up && !down && c.Low <= ind[i].lower && c.Close > ind[i].lower
		}},
		{name: "lower_reclaim_down_bias", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			up, down, _ := stateBias(ind, i)
			return !up && down && c.Low <= ind[i].lower && c.Close > ind[i].lower
		}},
		{name: "mid_up_flat_price_below", dir: 1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return ind[i].midUpAge >= 2 && ind[i].midUpAge <= 6 && c.Close < ind[i].mid && flatST(ind, i)
		}},
		{name: "mid_down_flat_price_above", dir: -1, fn: func(c types.Candle, ind []indicatorBar, i int) bool {
			return ind[i].midDownAge >= 2 && ind[i].midDownAge <= 6 && c.Close > ind[i].mid && flatST(ind, i)
		}},
	}
	byName := make(map[string][]hfResearchEvent)
	selectedSymbols := parseSymbolFilter(symbolsFilter)
	symbols, bars := 0, 0
	for _, f := range files {
		sym := strings.TrimSuffix(filepath.Base(f), "_"+interval+".json")
		if !analysisSymbolAllowed(sym, selectedSymbols, includeBTC) {
			continue
		}
		cs, err := loadCandles(f)
		if err != nil || len(cs) < minBars {
			continue
		}
		cs = filterTime(cs, start, end)
		if len(cs) < minBars {
			continue
		}
		ind := buildIndicators(cs, cfg)
		symbols++
		bars += len(cs)
		for i := 3; i+24 < len(cs)-1; i++ {
			if !ind[i].valid || bandWidth(ind[i]) <= 0 || cs[i+1].Open <= 0 {
				continue
			}
			for _, b := range buckets {
				if !b.fn(cs[i], ind, i) {
					continue
				}
				e := hfResearchEvent{dir: b.dir}
				for k, h := range reactionHorizons {
					e.returns[k] = float64(b.dir) * (cs[i+h].Close/cs[i+1].Open - 1)
				}
				byName[b.name] = append(byName[b.name], e)
			}
		}
	}
	fmt.Println("============================================================")
	fmt.Printf("ST/Boll state combination analysis: data=%s interval=%s symbols=%d bars=%d filter=%s include_btc=%t\n", dataDir, interval, symbols, bars, symbolsFilter, includeBTC)
	for _, b := range buckets {
		printHFResearch(b.name, byName[b.name])
	}
}

func printHFResearch(name string, events []hfResearchEvent) {
	if len(events) == 0 {
		fmt.Printf("%s: n=0\n", name)
		return
	}
	fmt.Printf("%s: n=%d\n", name, len(events))
	for k, h := range reactionHorizons {
		returns := make([]float64, 0, len(events))
		wins := 0
		for _, e := range events {
			returns = append(returns, e.returns[k])
			if e.returns[k] > 0 {
				wins++
			}
		}
		fmt.Printf("  h=%d: mean=%+.3f%% median=%+.3f%% win=%.1f%%\n", h, mean(returns)*100, medianFloat(returns)*100, pct(wins, len(events)))
	}
}

type reactionEvent struct {
	dir     int
	width   float64
	ratio   [3]float64
	returns [3]float64
}

var reactionHorizons = [...]int{6, 12, 24}

func analyzeCrossEvents(files []string, dataDir, interval string, start, end int64, minBars int, cfg config, symbolsFilter string, includeBTC bool) {
	events := make(map[string][]reactionEvent)
	symbols, bars := 0, 0
	selectedSymbols := parseSymbolFilter(symbolsFilter)
	for _, f := range files {
		sym := strings.TrimSuffix(filepath.Base(f), "_"+interval+".json")
		if !analysisSymbolAllowed(sym, selectedSymbols, includeBTC) {
			continue
		}
		cs, err := loadCandles(f)
		if err != nil || len(cs) < minBars {
			continue
		}
		cs = filterTime(cs, start, end)
		if len(cs) < minBars {
			continue
		}
		ind := buildIndicators(cs, cfg)
		symbols++
		bars += len(cs)
		for i := 1; i+reactionHorizons[len(reactionHorizons)-1] < len(cs); i++ {
			if !ind[i].valid || !ind[i-1].valid {
				continue
			}
			prev, cur := ind[i-1], ind[i]
			if cur.dir == 1 && cur.st > prev.st && prev.st >= prev.lower && cur.st < cur.lower {
				if e, ok := makeReactionEvent(1, i, cs, ind); ok {
					events["st_up_crosses_below_lower"] = append(events["st_up_crosses_below_lower"], e)
				}
			}
			if cur.dir == -1 && cur.st < prev.st && prev.st <= prev.upper && cur.st > cur.upper {
				if e, ok := makeReactionEvent(-1, i, cs, ind); ok {
					events["st_down_crosses_above_upper"] = append(events["st_down_crosses_above_upper"], e)
				}
			}

			// Count the first recovery bar of each support/resistance touch,
			// rather than every consecutive bar spent at the same level.
			stLong := cur.dir == 1 && cs[i].Low <= cur.st && cs[i].Close > cur.st
			stLongPrev := prev.dir == 1 && cs[i-1].Low <= prev.st && cs[i-1].Close > prev.st
			if stLong && !stLongPrev {
				if e, ok := makeReactionEvent(1, i, cs, ind); ok {
					events["price_reclaims_st_support"] = append(events["price_reclaims_st_support"], e)
				}
			}
			bbLong := cur.dir == 1 && cs[i].Low <= cur.lower && cs[i].Close > cur.lower
			bbLongPrev := prev.dir == 1 && cs[i-1].Low <= prev.lower && cs[i-1].Close > prev.lower
			if bbLong && !bbLongPrev {
				if e, ok := makeReactionEvent(1, i, cs, ind); ok {
					events["price_reclaims_boll_lower"] = append(events["price_reclaims_boll_lower"], e)
				}
			}
			stShort := cur.dir == -1 && cs[i].High >= cur.st && cs[i].Close < cur.st
			stShortPrev := prev.dir == -1 && cs[i-1].High >= prev.st && cs[i-1].Close < prev.st
			if stShort && !stShortPrev {
				if e, ok := makeReactionEvent(-1, i, cs, ind); ok {
					events["price_rejects_st_resistance"] = append(events["price_rejects_st_resistance"], e)
				}
			}
			bbShort := cur.dir == -1 && cs[i].High >= cur.upper && cs[i].Close < cur.upper
			bbShortPrev := prev.dir == -1 && cs[i-1].High >= prev.upper && cs[i-1].Close < prev.upper
			if bbShort && !bbShortPrev {
				if e, ok := makeReactionEvent(-1, i, cs, ind); ok {
					events["price_rejects_boll_upper"] = append(events["price_rejects_boll_upper"], e)
				}
			}
			clusterLong := cur.dir == 1 && bandWidth(cur) > 0 && math.Abs(cur.st-cur.lower)/bandWidth(cur) <= 0.25
			clusterLong = clusterLong && cs[i].Low <= math.Max(cur.st, cur.lower) && cs[i].Close > math.Max(cur.st, cur.lower)
			clusterLongPrev := prev.dir == 1 && bandWidth(prev) > 0 && math.Abs(prev.st-prev.lower)/bandWidth(prev) <= 0.25
			clusterLongPrev = clusterLongPrev && cs[i-1].Low <= math.Max(prev.st, prev.lower) && cs[i-1].Close > math.Max(prev.st, prev.lower)
			if clusterLong && !clusterLongPrev {
				if e, ok := makeReactionEvent(1, i, cs, ind); ok {
					events["price_reclaims_st_boll_support_cluster"] = append(events["price_reclaims_st_boll_support_cluster"], e)
				}
			}
			clusterShort := cur.dir == -1 && bandWidth(cur) > 0 && math.Abs(cur.st-cur.upper)/bandWidth(cur) <= 0.25
			clusterShort = clusterShort && cs[i].High >= math.Min(cur.st, cur.upper) && cs[i].Close < math.Min(cur.st, cur.upper)
			clusterShortPrev := prev.dir == -1 && bandWidth(prev) > 0 && math.Abs(prev.st-prev.upper)/bandWidth(prev) <= 0.25
			clusterShortPrev = clusterShortPrev && cs[i-1].High >= math.Min(prev.st, prev.upper) && cs[i-1].Close < math.Min(prev.st, prev.upper)
			if clusterShort && !clusterShortPrev {
				if e, ok := makeReactionEvent(-1, i, cs, ind); ok {
					events["price_rejects_st_boll_resistance_cluster"] = append(events["price_rejects_st_boll_resistance_cluster"], e)
				}
			}
		}
	}
	fmt.Println("============================================================")
	fmt.Printf("ST/Bollinger analysis: data=%s interval=%s symbols=%d bars=%d filter=%s include_btc=%t\n", dataDir, interval, symbols, bars, symbolsFilter, includeBTC)
	fmt.Println("Cross means ST line moved from inside the Bollinger envelope to outside while its SuperTrend direction was moving in that direction.")
	for _, kind := range []string{
		"st_up_crosses_below_lower",
		"st_down_crosses_above_upper",
		"price_reclaims_st_support",
		"price_reclaims_boll_lower",
		"price_rejects_st_resistance",
		"price_rejects_boll_upper",
		"price_reclaims_st_boll_support_cluster",
		"price_rejects_st_boll_resistance_cluster",
	} {
		printReaction(kind, events[kind])
	}
}

func makeReactionEvent(dir, i int, cs []types.Candle, ind []indicatorBar) (reactionEvent, bool) {
	entry := cs[i+1].Open
	width := bandWidth(ind[i])
	if entry <= 0 || width <= 0 {
		return reactionEvent{}, false
	}
	e := reactionEvent{dir: dir, width: width}
	for k, h := range reactionHorizons {
		j := i + h
		if j >= len(cs) || !ind[j].valid {
			return reactionEvent{}, false
		}
		e.ratio[k] = bandWidth(ind[j]) / width
		e.returns[k] = float64(dir) * (cs[j].Close/entry - 1)
	}
	return e, true
}

func printReaction(kind string, events []reactionEvent) {
	if len(events) == 0 {
		fmt.Printf("%s: n=0\n", kind)
		return
	}
	fmt.Printf("%s: n=%d\n", kind, len(events))
	for k, h := range reactionHorizons {
		ratios := make([]float64, 0, len(events))
		returns := make([]float64, 0, len(events))
		contracted := 0
		winning := 0
		for _, e := range events {
			ratios = append(ratios, e.ratio[k])
			returns = append(returns, e.returns[k])
			if e.ratio[k] < 1 {
				contracted++
			}
			if e.returns[k] > 0 {
				winning++
			}
		}
		fmt.Printf("  h=%d: width ratio mean=%.3f median=%.3f contracted=%.1f%% | fwd return mean=%+.3f%% median=%+.3f%% win=%.1f%%\n", h, mean(ratios), medianFloat(ratios), pct(contracted, len(events)), mean(returns)*100, medianFloat(returns)*100, pct(winning, len(events)))
	}
}

func bandWidth(b indicatorBar) float64 {
	if b.mid <= 0 || b.upper <= b.lower {
		return 0
	}
	return (b.upper - b.lower) / b.mid
}

func mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	var total float64
	for _, x := range v {
		total += x
	}
	return total / float64(len(v))
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
