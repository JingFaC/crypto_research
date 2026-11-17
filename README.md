# crypto_research

加密货币与美股永续的**策略回测研究仓库**。只做研究，不含实盘、部署和交易代码。

## 我们在做什么

1. 把以往零散的策略想法，用统一的回测代码重新跑一遍；
2. 用同一套[评价标准](docs/方法/评价标准.md)给每个策略判定：达标 / 有研究价值 / 已否决 / 归档；
3. 每个结论都带回测数据，让人一眼看清哪些策略证明不行、哪些还值得继续研究。

## 当前结论

| 状态 | 数量 | 策略 |
|---|---|---|
| 达标 | 0 | — |
| 有研究价值 | 9 | EMA排列开关门DCA、日线ST门DCA、KDJ-EMA顺势、ST-SAR网格、ETH日线ST翻转做多、大币日线ST组合（前 4 币 / 老牌 10 币 / BIG20）、周ST下跌日MACD-V抄底门 |
| 已否决 | 6 | 日线ST翻转做多全币池、ETH 4h ST 下轨挂单（上轨 / 中轨止盈）、ETH 4h ST 反转通道补仓、ZigZag-DCA、BTC周线ST门DCA |
| 归档 | 8 | 见[归档](docs/策略/归档.md) |

详细数字见[策略总览](docs/策略/README.md)。

## 从哪里开始读

[文档目录](docs/README.md)列出了所有文档和阅读顺序。

## 仓库结构

```
docs/        文档（方法、策略、数据说明）
research/    回测代码：soxl、dca_macd、st_boll，统一入口 research/run.py
数据/        行情数据（不入库，已被 .gitignore 忽略）
.cache/      回测输出 JSON（不入库）
```

## 运行

环境：Python 3.11，`pip install -r requirements.txt`；`st_boll` 的部分策略需要 Go。

统一入口：`python -m research.run <组> --strategy <策略> [参数] --output .cache/<名>.json`

| 组 | 内容 | 可选 `--strategy` |
|---|---|---|
| `soxl` | KDJ、网格 | `kdj_ema` `grid_v11` `grid_v12` |
| `dca_macd` | EMA 门 DCA、ZigZag 等 | `ema_dca` `zigzag_dca` `st_macd` 等 |
| `st_boll` | ST、布林 | `daily_st` `eth_boll` 等 |

各策略的精确复现命令写在各自文档的头部表格里。

测试：

```
python -m unittest discover -s research/soxl -t . -p "test_*.py"
python -m unittest discover -s research/dca_macd -t . -p "test_*.py"
python -m unittest discover -s research/st_boll -t . -p "test_*.py"
```

## 约定

- 研究结论只写 Markdown，放在 `docs/`；运行结果放被忽略的 `.cache/`。
- 代码不依赖其他仓库；需要的少量输入已复制到 `research/soxl/inputs/`，数据见[数据说明](docs/数据/数据说明.md)。
- 协作规则见 [AGENTS.md](AGENTS.md)。
