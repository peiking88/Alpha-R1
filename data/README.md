# Data directory conventions

Raw data is **not** committed to this repository (see `.gitignore`). Place the
following inputs here before running the pipeline:

```
data/
├── stock_data/        # OHLCV daily bars as CSV, one file per instrument
│                      #   columns: date,open,high,low,close,volume[,vwap,factor]
│                      #   e.g. data/stock_data/SH600000.csv
│                      #   consumed by: scripts/prepare_qlib_data.py
├── price_market/      # daily price-market descriptions, one txt per day
│                      #   filename: YYYY-MM-DD_23:59:59.txt
│                      #   consumed by: scripts/build_market_memory.py
├── news/              # daily news descriptions, one txt per day
│                      #   filename: YYYY-MM-DD.txt
│                      #   consumed by: scripts/build_market_memory.py
└── market_memory/     # outputs of the iterative memory construction
                       #   (paper Section 3.1.2): weekly summaries + M_global.txt
```

Alternatives:

- qlib data: instead of converting `stock_data/` yourself, you may use an
  official qlib community data bundle (see the qlib documentation for the
  current data-collector commands) and point `configs/backtest.yaml:
qlib_data_dir` at it.
- price/news descriptions: any plain-text daily summaries work; keep the
  filename conventions above so the loaders can align them with trading days.
