package types

import "encoding/json"

// Candle is a normalized futures kline.
// Both live and backtest use this structure.
type Candle struct {
	Symbol    string  `json:"symbol,omitempty"` // 所属标的（引擎路由用；交易所返回的 K 线无此字段）
	OpenTime  int64   `json:"open_time"`
	Open      float64 `json:"open"`
	High      float64 `json:"high"`
	Low       float64 `json:"low"`
	Close     float64 `json:"close"`
	Volume    float64 `json:"volume"`
	CloseTime int64   `json:"close_time"`
}

// UnmarshalJSON accepts both snake_case and CamelCase field names
// to handle CSV-converted data with inconsistent casing.
func (c *Candle) UnmarshalJSON(data []byte) error {
	type rawCandle struct {
		OpenTime     int64   `json:"open_time"`
		OpenTimeAlt  int64   `json:"OpenTime"`
		Open         float64 `json:"open"`
		OpenAlt      float64 `json:"Open"`
		High         float64 `json:"high"`
		HighAlt      float64 `json:"High"`
		Low          float64 `json:"low"`
		LowAlt       float64 `json:"Low"`
		Close        float64 `json:"close"`
		CloseAlt     float64 `json:"Close"`
		Volume       float64 `json:"volume"`
		VolumeAlt    float64 `json:"Volume"`
		CloseTime    int64   `json:"close_time"`
		CloseTimeAlt int64   `json:"CloseTime"`
	}
	var raw rawCandle
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	c.OpenTime = firstInt64(raw.OpenTime, raw.OpenTimeAlt)
	c.Open = firstFloat64(raw.Open, raw.OpenAlt)
	c.High = firstFloat64(raw.High, raw.HighAlt)
	c.Low = firstFloat64(raw.Low, raw.LowAlt)
	c.Close = firstFloat64(raw.Close, raw.CloseAlt)
	c.Volume = firstFloat64(raw.Volume, raw.VolumeAlt)
	c.CloseTime = firstInt64(raw.CloseTime, raw.CloseTimeAlt)
	return nil
}

func firstInt64(primary, fallback int64) int64 {
	if primary != 0 {
		return primary
	}
	return fallback
}

func firstFloat64(primary, fallback float64) float64 {
	if primary != 0 {
		return primary
	}
	return fallback
}

// Position represents an open position.
type Position struct {
	Symbol     string  `json:"symbol"`
	Side       string  `json:"side"` // LONG or SHORT
	Amount     float64 `json:"amount"`
	EntryPrice float64 `json:"entry_price"`
}

// Order is a normalized order shape for the exchange client.
//
// 铁律：订单唯一标识 = 交易所签发 ID（OrderID）。
//   - 普通单（MARKET/LIMIT）= 交易所 orderId
//   - 条件单（STOP_MARKET/TAKE_PROFIT_MARKET，走 /fapi/v1/algoOrder）= 交易所 algoId
//   - 禁止生成/发送任何自定义 clientOrderId（newClientOrderId/clientAlgoId 一律不传）
type Order struct {
	Symbol         string `json:"symbol"`
	Side           string `json:"side"`
	Position       string `json:"position"` // LONG or SHORT
	Type           string `json:"type"`     // MARKET, LIMIT, STOP_MARKET, TAKE_PROFIT_MARKET
	Quantity       string `json:"quantity"`
	Price          string `json:"price,omitempty"`
	StopPrice      string `json:"stop_price,omitempty"`
	WorkingType    string `json:"working_type,omitempty"`
	ClosePosition  bool   `json:"close_position,omitempty"`
	ReduceOnly     bool   `json:"reduce_only,omitempty"`
	AutoCancelSecs int    `json:"auto_cancel_secs,omitempty"`
	CloseBypassMin bool   `json:"close_bypass_min,omitempty"`
	// 交易所签发订单 ID（下单响应返回；普通单=orderId，条件单=algoId）
	OrderID int64 `json:"order_id,omitempty"`
	// 是否为条件单（走 algo 接口；由 broker 从 openAlgoOrders 返回时标记）
	IsConditional bool `json:"is_conditional,omitempty"`
	// 语义角色（引擎挂单时填充，成交回调路由用；不是订单标识）
	Reason string `json:"reason,omitempty"`
}

// OrderResult is the normalized result returned by exchange order APIs.
type OrderResult struct {
	OrderID int64  `json:"orderId"`
	AlgoID  int64  `json:"algoId"`
	Symbol  string `json:"symbol"`
	Status  string `json:"status"`
}

// OrderRef 订单引用（轮询跟踪用）。
// 来源必须是策略持久化状态（pendingOrder/tpOrder，重启后从 state 恢复），
// 不能依赖进程内内存——重启后已成交的单也要能被轮询到并继续处理。
type OrderRef struct {
	OrderID       int64
	IsConditional bool // 条件单（algoId）
}

// OrderStatus 订单状态（broker 接口通用）
type OrderStatus struct {
	Symbol      string  `json:"symbol"`
	OrderID     int64   `json:"orderId"` // 普通单=orderId，条件单=algoId
	Status      string  `json:"status"`  // NEW / FILLED / CANCELED / EXPIRED / REJECTED
	Price       float64 `json:"price"`
	AvgPrice    float64 `json:"avgPrice"`
	OrigQty     float64 `json:"origQty"`
	ExecutedQty float64 `json:"executedQty"`
}

// SymbolInfo 交易对信息（broker 接口通用）
type SymbolInfo struct {
	PricePrecision    int     `json:"pricePrecision"`
	QuantityPrecision int     `json:"quantityPrecision"`
	TickSize          float64 `json:"tickSize"`
	MinQty            float64 `json:"minQty"`
	MinNotional       float64 `json:"minNotional"`
	StepSize          float64 `json:"stepSize"`
}

// Signal represents a directional trading signal.
type Signal int

const (
	SignalNone Signal = iota
	SignalBuy
	SignalSell
)

// SignalEvent describes a strategy signal with context.
type SignalEvent struct {
	Signal    Signal  `json:"signal"`
	Time      int64   `json:"time"`
	Price     float64 `json:"price"`
	BlueValue float64 `json:"blue_value,omitempty"`
	Info      string  `json:"info,omitempty"`
}

// Config is a generic config placeholder used by BLUE utilities.
type Config struct {
	Symbol         string  `json:"symbol"`
	Interval       string  `json:"interval"`
	OrderAmount    float64 `json:"order_amount"`
	PricePrecision int     `json:"price_precision"`
	NeedDays       int     `json:"need_days"`
	BlueSignalDays int     `json:"blue_signal_days"`
}

// BlueValue represents a BLUE indicator value at a timestamp.
type BlueValue struct {
	Value float64 `json:"value"`
	Time  int64   `json:"time"`
}

// LivePosition 交易所真实持仓（positionRisk 只读视图，供账实核对销账用）。
// 账目推导的持仓必须向它对齐（外部手动平仓/强平时账实会脱节）。
type LivePosition struct {
	Symbol       string
	PositionSide string  // "LONG" / "SHORT"（positionRisk 的 positionSide）
	PositionAmt  float64 // 持仓数量（正数=多，负数=空）
	EntryPrice   float64
}
