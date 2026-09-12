---
name: check-alpha-r1-env
description: >
  Use when verifying Alpha-R1 (src/alpha_r1) env readiness before training or
  backtest — conda quantaalpha, torch/CUDA, TDengine tdx.kline, FinStep/Alpha-R1
  LLM cache, betas.csv. Even if the user just says 检查环境/环境就绪吗. NOT for
  single lookups (pip list, one config, git status) or other projects.
user-invocable: true
---

# check-alpha-r1-env

Alpha-R1 项目专属环境体检：一条链路跑通「Python 环境 → 依赖 → GPU → TDengine → 模型资产 → 冒烟回测」，输出汇总表。全部检查项都在本机实测过；任何一步失败给出修复命令（国内镜像优先）。

## Prerequisites

- 项目根：`/home/li/financial/Alpha-R1`（含 `pyproject.toml`、`src/alpha_r1/`）
- conda 环境 `quantaalpha`（系统 Python 3.14 无项目依赖，项目根无 venv——**不要**用 `python`/`pip` 裸命令）

## Steps

1. **解释器与依赖**（用绝对路径，不依赖 activate）：

   ```bash
   PY=/home/li/miniconda3/envs/quantaalpha/bin/python
   $PY --version   # 3.10.x
   $PY -c "import torch, transformers, accelerate, taosws; assert torch.cuda.is_available(), 'CUDA unavailable'"
   ```

   缺依赖时安装（清华镜像）：`$PY -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
   注意：`huggingface-cli` 已弃用（打印帮助后 exit 0 假成功），下载用 `hf` 命令并设 `HF_ENDPOINT=https://hf-mirror.com`。

2. **模块导入冒烟**（子模块各自 import，定位缺失依赖到具体模块）：

   ```bash
   $PY -c "import sys; sys.path.insert(0,'src'); import alpha_r1.backtest, alpha_r1.data, alpha_r1.factors, alpha_r1.generation, alpha_r1.inference, alpha_r1.parsing"
   ```

3. **TDengine 数据源**（WebSocket 直连，库 tdx）：

   ```bash
   $PY -c "import taosws; c=taosws.connect('taosws://root:taosdata@localhost:6041'); r=c.query('SELECT COUNT(*) FROM tdx.kline'); assert r[0][0] > 100_000_000, r[0][0]"
   ```

   失败则查服务：`systemctl status taosd`（或 taosadapter）；行数断言防止空库假通过（总数验证，不逐行看）。

4. **模型资产**：
   - LLM 缓存：`du -sh ~/.cache/huggingface/hub/models--FinStep--Alpha-R1`（完整约 16G、4 个 safetensors；0 或很小 = 未下载，用 `HF_ENDPOINT=https://hf-mirror.com hf download FinStep/Alpha-R1` 补）
   - betas：`head -3 result/linear_model/betas.csv`（82 因子 + _intercept；缺失则跑 `scripts/train_linear_model.py --start-date <4年前> --end-date <今天> --device cpu --output result/linear_model/betas.csv`）

5. **端到端冒烟**（真实数据 + GPU，约 1 分钟）：
   ```bash
   $PY scripts/run_realtime_backtest.py --alphas 001 --zxg --device cuda
   ```
   成功标志：输出 `result/alpha_backtest/alpha001.json` 且 `factor/market/window/topk` 字段非 null；`data shape` 行数随自选股数变化（今日 483×105），不作为门槛。

## Verification

- 全部 5 步通过 → 汇总表报告 ✅，链路（训练/因子回测/策略回测）可用
- 任何一步失败：报告 ❌ + 关键错误输出 + 修复命令；后续步骤仍继续跑（单项失败不阻断体检）

## Notes

- **显存天花板**（RTX 4060 Laptop 8GB）：全市场（9352 只）GPU 因子计算必 OOM——因子回测须 `--zxg` 限定 universe，训练/策略回测须 `--device cpu`。LLM bf16 16.4GB 超显存，`device_map: auto` 会 offload 到内存（能跑但慢）。
- universe 已排除北交所（SQL `WHERE market <> 'bj'`）；仍含指数/ETF（15/16/51 开头基金代码会被当股票打分，已知未修）。
- `result/` 下产物（betas、alpha_backtest/*.json）缺失只算降级警告，不算环境失败。
- git 工作区应干净；`.skill-forge/`、`summary.md` 未跟踪属正常。
