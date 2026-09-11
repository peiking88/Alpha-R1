<p align="center">
  <img src="assets/Alpha-R1.png" alt="Alpha-R1" style="width: 100%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2512.23515" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.23515-B31B1B?logo=arxiv"/></a>
  <a href="https://huggingface.co/FinStep/Alpha-R1" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FinStep%2FAlpha--R1-yellow"/></a>
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"/></a>
  <a href="https://www.python.org/downloads/release/python-3100/" target="_blank"><img alt="Python Version" src="https://img.shields.io/badge/Python-3.10+-brightgreen.svg"/></a>
</div>
<div align="center">
  <a href="README_en.md">English</a> | <a href="README.md">中文</a>
</div>

---

# Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning

> **Alpha-R1** is a reasoning-enhanced LLM for quantitative alpha selection, trained with GRPO reinforcement learning on top of Qwen3-8B. This repository is the companion implementation of the paper
> [Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning](https://arxiv.org/abs/2512.23515).

## Overview

<p align="center">
  <img src="assets/framework.png" alt="Alpha-R1 framework overview" style="width: 100%;">
</p>

Alpha-R1 screens a candidate pool of [Alpha101](https://arxiv.org/abs/1601.00991) factors. Instead of treating alphas as bare time series, it reasons over **semantic factor descriptions** — how each factor works, when it works, and when it fails — and activates the factors that fit current market conditions:

```
qlib single-factor backtest (P_i) ─┐
                                   ├─→ LLM factor descriptions α_des (OpenRouter)
market memory (M_global) ──────────┘                 │
                                                     ↓
                           Alpha-R1 inference (FinStep/Alpha-R1)
                                                     ↓
                               parsed selections (selections.json)
                                                     ↓
                       end-to-end strategy backtest (NAV, AR/SR/MDD)
```

1. **Single-factor backtesting** (paper §3.1.3): each Alpha101 factor is evaluated on qlib — factor values, IC/RankIC, and a top-k portfolio — and saved as a performance vector `P_i`.
2. **Factor description generation** (§3.1.2/§3.2.1): an LLM (via OpenRouter) iteratively aggregates daily price/news text into a global market memory `M_global`, then maps `M_global + P_i` into a structured description `α_des` per factor.
3. **Alpha-R1 inference** (§3.3): descriptions are concatenated into the decision-context prompt; the model outputs the selected factors in `<alpha_list>...</alpha_list>`.
4. **Output parsing**: responses are validated and parsed into `selections.json` / `summary.csv`.
5. **Strategy backtest** (§3.3, Appendix F): a fixed linear model scores stocks with the selected factors, and the paper's execution protocol (slot rotation, VWAP fills, fees) produces the NAV curve and metrics.

## Installation

```bash
pip install -e .            # core (transformers inference + generation + parsing)
pip install -e .[vllm]      # optional high-throughput inference backend
pip install -e .[qlib]      # optional backtesting (pyqlib)
```

API keys (see `.env.example`):

```bash
export OPENROUTER_API_KEY=...   # description generation
export HF_TOKEN=...             # optional (e.g. for gated/private mirrors)
```

## Usage

All steps read their defaults from `configs/`.

### 1. Single-factor backtests

```bash
python scripts/prepare_qlib_data.py --csv-dir data/stock_data --qlib-dir ~/.qlib/qlib_data/alpha_r1
python scripts/run_factor_backtest.py --alphas all
```

Writes `result/alpha_backtest/alphaNNN.json` per factor. See `data/README.md` for data layout conventions.

### 2. Factor descriptions

```bash
python scripts/build_market_memory.py --start-date 2023-01-01 --end-date 2024-12-31
python scripts/generate_descriptions.py --model "anthropic/claude-3.7-sonnet" --alphas all
```

The OpenRouter model id is intentionally left blank in `configs/generation.yaml` — set it there or pass `--model`. Writes `data/market_memory/M_global.txt` and `result/alpha_des/alphaNNN.txt`.

### 3. Alpha-R1 inference

```bash
python scripts/run_inference.py \
    --factor-des-dir result/alpha_des \
    --start-date 2025-01-01 --end-date 2025-12-31
```

Loads `FinStep/Alpha-R1` via `from_pretrained` (default `hf` backend; set `backend: vllm` in `configs/inference.yaml` to switch to vLLM). Decoding defaults to `temperature=0, top_p=0.7` as in the paper. Writes `result/alpha_select/result_YYYYMMDD.json`. A minimal smoke run without real data works with `--factor-des-dir examples/factor_descriptions`.

Decision days default to weekdays in the given range (a trading-calendar approximation); pass `--market-state-dir` to restrict to days that actually have market data.

### 4. Parse outputs

```bash
python scripts/parse_outputs.py --result-dir result/alpha_select
```

Writes `selections.json` (date → factor list) and `summary.csv`, and reports format-invalid days.

### 5. Strategy backtest (end-to-end)

```bash
# estimate the fixed linear model on the historical window (paper: 2020-2023)
python scripts/train_linear_model.py --alphas all
# backtest the selections with the paper's execution protocol (slot rotation, VWAP, 10 bps)
python scripts/run_strategy_backtest.py \
    --selections result/alpha_select/parsed/selections.json \
    --betas result/linear_model/betas.csv
```

Turns parsed selections into a tradable top-10 equal-weight portfolio: capital rotates through `holding_days` slots (one slot rebalanced per day), fills use daily `$vwap` (falling back to `$close`), fees are 10 bps per side, idle cash earns the risk-free rate, and decision day t scores stocks with factor values of t-1. Writes metrics JSON (AR / excess SR / MDD / Sortino / Calmar / IR vs benchmark) and a daily NAV CSV to `configs/strategy.yaml: output_dir`. Passing a directory of per-round selections files as `--selections` additionally writes the multi-round average. Limit-lock filtering is available but off by default (daily data carries no limit flags; see `configs/strategy.yaml`).

## Results

Main experiment results (paper Table 1; 12-month out-of-sample testing period 2025-01-01 to 2025-12-31). AR = annualized return, SR = excess Sharpe ratio, MDD = max drawdown.

<p align="center">
  <img src="assets/main_result_sp500.png" alt="Backtest NAV comparison on the S&P 500 asset pool" style="width: 49%;">
  <img src="assets/main_result_csi300.png" alt="Backtest NAV comparison on the CSI 300 asset pool" style="width: 49%;">
</p>

<table>
  <thead>
    <tr>
      <th rowspan="2">Type</th>
      <th rowspan="2" width="160">Method</th>
      <th colspan="3">S&amp;P 500</th>
      <th colspan="3">CSI 300</th>
    </tr>
    <tr>
      <th>AR (%)</th>
      <th>SR</th>
      <th>MDD (%)</th>
      <th>AR (%)</th>
      <th>SR</th>
      <th>MDD (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="9">Non-LLM</td><td>Buy&nbsp;&amp;&nbsp;Hold</td><td>19.34</td><td>0.80</td><td>18.75</td><td>22.16</td><td>1.31</td><td>10.49</td></tr>
    <tr><td>PCA</td><td>7.98</td><td>0.27</td><td>17.30</td><td>2.93</td><td>0.17</td><td>14.46</td></tr>
    <tr><td>XGBoost</td><td>3.49</td><td>0.03</td><td>18.45</td><td>8.99</td><td>0.50</td><td>16.26</td></tr>
    <tr><td>LightGBM</td><td>-5.42</td><td>-0.43</td><td>20.93</td><td>18.44</td><td>1.05</td><td>14.92</td></tr>
    <tr><td>A2C</td><td>10.82</td><td>0.40</td><td>17.70</td><td>22.96</td><td>1.20</td><td>14.86</td></tr>
    <tr><td>PPO</td><td>7.68</td><td>0.25</td><td>14.97</td><td>14.96</td><td>0.81</td><td>12.95</td></tr>
    <tr><td>DDPG</td><td>2.53</td><td>-0.02</td><td>15.04</td><td>1.97</td><td>0.12</td><td>16.54</td></tr>
    <tr><td>TD3</td><td>5.54</td><td>0.14</td><td>16.58</td><td>8.66</td><td>0.52</td><td>10.26</td></tr>
    <tr><td>SAC</td><td>37.60</td><td>1.44</td><td>15.18</td><td>9.77</td><td>0.56</td><td>11.68</td></tr>
    <tr><td rowspan="5">LLM</td><td>Gemini&nbsp;2.5&nbsp;Pro</td><td>14.23</td><td>0.55</td><td>17.01</td><td>16.29</td><td>0.90</td><td>14.01</td></tr>
    <tr><td>Claude&nbsp;3.7&nbsp;Sonnet</td><td>10.92</td><td>0.40</td><td>18.88</td><td>10.13</td><td>0.57</td><td>14.49</td></tr>
    <tr><td>DeepSeek&#8209;R1</td><td>21.94</td><td>0.93</td><td><b>14.36</b></td><td>14.66</td><td>0.81</td><td>14.60</td></tr>
    <tr><td>Qwen3&#8209;8B</td><td>12.85</td><td>0.47</td><td>19.52</td><td>15.44</td><td>0.79</td><td>14.38</td></tr>
    <tr><td><b>Alpha&#8209;R1&nbsp;(Ours)</b></td><td><b>47.87</b></td><td><b>1.62</b></td><td>16.91</td><td><b>40.57</b></td><td><b>2.23</b></td><td><b>6.58</b></td></tr>
  </tbody>
</table>

On out-of-domain universes without retraining, Alpha-R1 reaches 80.54% AR (SR 2.46) on Russell 2000 and 73.52% AR (SR 2.80) on CSI 1000 (paper Table 2).

## Training

Alpha-R1 is trained with GRPO on Qwen3-8B using [verl](https://github.com/volcengine/verl) and a market-feedback reward (paper Section 3.4). `training/` contains the training configuration (`configs/grpo_alpha_r1.yaml`) and a simplified reference implementation of the reward (`reward.py`), pluggable via verl's `custom_reward_function` mechanism. See `training/README.md` for details.

## Repository layout

```
src/alpha_r1/
├── factors/       Alpha101 formula library + description loading/concatenation
├── backtest/      qlib data conversion, Alpha101→qlib expressions, single-factor
│                  backtest, linear model, slot-rotation strategy backtest
├── generation/    OpenRouter client, market memory, description generation
├── inference/     transformers / vLLM backends, prompt builder, selection loop
└── parsing/       <alpha_list> extraction and validation
scripts/           CLI entry points for each pipeline step
configs/           YAML configs for backtest / generation / inference / strategy
training/          GRPO training config + reference reward (verl)
data/              raw data lives here (gitignored, see data/README.md)
examples/          minimal example inputs
```

## Citation

```bibtex
@article{jiang2025alphar1,
  title={Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning},
  author={Jiang, Zuoyou and Zhao, Li and Sun, Rui and Sun, Ruohan and Li, Zhongjian and Li, Jing and Jiang, Daxin and Bai, Zuo and Hua, Cheng},
  journal={arXiv preprint arXiv:2512.23515},
  year={2025}
}
```

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).

## 📅 Roadmap & Updates

- **[2025.12]** 📄 Paper released on [arXiv](https://arxiv.org/abs/2512.23515).
- **[2026.09]** 🧩 Code release: qlib single-factor backtests, factor description generation (OpenRouter), Alpha-R1 inference, output parsing, end-to-end strategy backtest, and the GRPO training config + reference reward.
  - ✅ Inference code (Alpha Screening Pipeline)
  - ✅ Model weights ([`FinStep/Alpha-R1`](https://huggingface.co/FinStep/Alpha-R1))

_Please ⭐ Star this repo to stay updated!_
