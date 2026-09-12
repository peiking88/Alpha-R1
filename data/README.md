# Data directory conventions

Raw data is **not** committed to this repository (see `.gitignore`). Place the
following inputs here before running the pipeline:

```
data/
├── price_market/      # daily price-market descriptions, one txt per day
│                      #   filename: YYYY-MM-DD_23:59:59.txt
│                      #   consumed by: scripts/build_market_memory.py
├── news/              # daily news descriptions, one txt per day
│                      #   filename: YYYY-MM-DD.txt
│                      #   consumed by: scripts/build_market_memory.py
└── market_memory/     # outputs of the iterative memory construction
                       #   (paper Section 3.1.2): weekly summaries + M_global.txt
```

Notes:

- OHLCV market data is loaded directly from TDengine
  (`alpha_r1.data`); no local CSV/binary snapshot is kept.
- price/news descriptions: any plain-text daily summaries work; keep the
  filename conventions above so the loaders can align them with trading days.
