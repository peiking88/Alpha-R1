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

> **Alpha-R1** 是一个面向量化 Alpha 筛选的推理增强型 LLM，基于 Qwen3-8B 通过 GRPO 强化学习训练。本仓库是论文
> [Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning](https://arxiv.org/abs/2512.23515)
> 的配套实现。

## Overview (项目概览)

<p align="center">
  <img src="assets/framework.png" alt="Alpha-R1 framework overview" style="width: 100%;">
</p>

Alpha-R1 从 [Alpha101](https://arxiv.org/abs/1601.00991) 候选因子池中筛选因子。它不把 alpha 当作裸的时间序列，而是基于**语义化的因子描述**进行推理——每个因子如何起作用、何时有效、何时失效——并激活与当前市场环境相匹配的因子：

```
TDengine direct single-factor backtest (P_i) ─┐
                                              ├─→ LLM factor descriptions α_des (OpenRouter)
market memory (M_global) ─────────────────────┘                 │
                                                                ↓
                                  Alpha-R1 inference (FinStep/Alpha-R1)
                                                                ↓
                                    parsed selections (selections.json)
                                                                ↓
                            end-to-end strategy backtest (NAV, AR/SR/MDD)
```

1. **单因子回测**（论文 §3.1.3）：每个 Alpha101 因子直连 TDengine 加载数据、GPU 计算——因子值、IC/RankIC 与 top-k 组合——并保存为绩效向量 `P_i`。
2. **因子描述生成**（§3.1.2/§3.2.1）：LLM（经由 OpenRouter）将每日行情/新闻文本迭代聚合为全局市场记忆 `M_global`，再将 `M_global + P_i` 映射为每个因子的结构化描述 `α_des`。
3. **Alpha-R1 推理**（§3.3）：因子描述拼接为决策上下文 prompt，模型以 `<alpha_list>...</alpha_list>` 输出所选因子。
4. **输出解析**：对响应进行校验，并解析为 `selections.json` / `summary.csv`。
5. **策略回测**（§3.3，附录 F）：用固定的线性模型按所选因子为股票打分，按论文的执行协议（槽位轮换、VWAP 成交、手续费）产出净值曲线与指标。

## Installation (安装)

```bash
pip install -e .            # core (transformers inference + generation + parsing)
pip install -e .[vllm]      # optional high-throughput inference backend
```

行情数据从 TDengine 直连加载（`taosws://localhost:6041`，库 `tdx`），无需本地 CSV/qlib 数据准备。

API 密钥（见 `.env.example`）：

```bash
export OPENROUTER_API_KEY=...   # description generation
export HF_TOKEN=...             # optional (e.g. for gated/private mirrors)
```

## Usage (使用方法)

所有步骤的默认参数均从 `configs/` 读取。

### 1. Single-factor backtests (单因子回测)

```bash
python scripts/run_realtime_backtest.py --alphas all            # 全市场（需大显存）
python scripts/run_realtime_backtest.py --alphas all --zxg      # 自选股 universe（默认路径）
```

每个因子输出 `result/alpha_backtest/alphaNNN.json`。`--device cpu` 可在显存不足时切换计算设备。

### 2. Factor descriptions (因子描述生成)

```bash
python scripts/build_market_memory.py --start-date 2023-01-01 --end-date 2024-12-31
python scripts/generate_descriptions.py --model "anthropic/claude-3.7-sonnet" --alphas all
```

`configs/generation.yaml` 中的 OpenRouter 模型 id 特意留空——请在配置中填写，或通过 `--model` 传入。输出 `data/market_memory/M_global.txt` 与 `result/alpha_des/alphaNNN.txt`。

### 3. Alpha-R1 inference (Alpha-R1 推理)

```bash
python scripts/run_inference.py \
    --factor-des-dir result/alpha_des \
    --start-date 2025-01-01 --end-date 2025-12-31
```

通过 `from_pretrained` 加载 `FinStep/Alpha-R1`（默认 `hf` 后端；在 `configs/inference.yaml` 中设置 `backend: vllm` 可切换为 vLLM）。解码默认 `temperature=0, top_p=0.7`，与论文一致。输出 `result/alpha_select/result_YYYYMMDD.json`。无真实数据时，可用 `--factor-des-dir examples/factor_descriptions` 做最小冒烟运行。

决策日默认为给定区间内的工作日（交易日历的近似）；传入 `--market-state-dir` 可限制为实际有行情数据的日期。

### 4. Parse outputs (输出解析)

```bash
python scripts/parse_outputs.py --result-dir result/alpha_select
```

输出 `selections.json`（日期 → 因子列表）与 `summary.csv`，并报告格式非法的日期。

### 5. Strategy backtest (端到端策略回测)

```bash
# estimate the fixed linear model on the historical window (paper: 2020-2023)
python scripts/train_linear_model.py --alphas all --device cpu
# backtest the selections with the paper's execution protocol (slot rotation, VWAP, 10 bps)
python scripts/run_strategy_backtest.py \
    --selections result/alpha_select/parsed/selections.json \
    --betas result/linear_model/betas.csv --device cpu
```

将解析出的因子选择转化为可交易的 top-10 等权组合：资金在 `holding_days` 个槽位间轮换（每日再平衡一个槽位），成交价使用当日 VWAP（缺失时回退收盘价），双边手续费 10 bps，闲置现金按无风险利率计息，决策日 t 使用 t-1 日的因子值打分。输出指标 JSON（AR / 超额 SR / MDD / Sortino / Calmar / IR，相对基准）与逐日净值 CSV 至 `configs/strategy.yaml: output_dir`。`--selections` 传入多轮选择文件目录时，会额外输出多轮平均结果。涨跌停过滤已实现但默认关闭（日线数据不含涨跌停标记；见 `configs/strategy.yaml`）。

## Results (实验结果)

主实验结果（论文 Table 1；12 个月样本外测试区间 2025-01-01 至 2025-12-31）。AR = 年化收益，SR = 超额夏普比率，MDD = 最大回撤。

<p align="center">
  <img src="assets/main_result_sp500.png" alt="Backtest NAV comparison on the S&P 500 asset pool" style="width: 49%;">
  <img src="assets/main_result_csi300.png" alt="Backtest NAV comparison on the CSI 300 asset pool" style="width: 49%;">
</p>

<table>
  <thead>
    <tr>
      <th rowspan="2">类型</th>
      <th rowspan="2" width="160">方法</th>
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

在域外股票池上无需重训：Alpha-R1 在 Russell 2000 上达到 80.54% AR（SR 2.46），在 CSI 1000 上达到 73.52% AR（SR 2.80）（论文 Table 2）。

## Training (训练)

Alpha-R1 基于 Qwen3-8B，使用 [verl](https://github.com/volcengine/verl) 以 GRPO 与市场反馈奖励训练（论文 Section 3.4）。`training/` 包含训练配置（`configs/grpo_alpha_r1.yaml`）与奖励的简化参考实现（`reward.py`），可通过 verl 的 `custom_reward_function` 机制接入。详见 `training/README.md`。

## Repository layout (仓库结构)

```
src/alpha_r1/
├── factors/       Alpha101 formula library + description loading/concatenation
├── backtest/      TDengine-direct single-factor backtest (GPU factors), linear
│                  model, slot-rotation strategy backtest
├── generation/    OpenRouter client, market memory, description generation
├── inference/     transformers / vLLM backends, prompt builder, selection loop
└── parsing/       <alpha_list> extraction and validation
scripts/           CLI entry points for each pipeline step
configs/           YAML configs for backtest / generation / inference / strategy
training/          GRPO training config + reference reward (verl)
data/              raw data lives here (gitignored, see data/README.md)
examples/          minimal example inputs
```

## Citation (引用)

```bibtex
@article{jiang2025alphar1,
  title={Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning},
  author={Jiang, Zuoyou and Zhao, Li and Sun, Rui and Sun, Ruohan and Li, Zhongjian and Li, Jing and Jiang, Daxin and Bai, Zuo and Hua, Cheng},
  journal={arXiv preprint arXiv:2512.23515},
  year={2025}
}
```

## License (开源协议)

本项目基于 [MIT License](https://opensource.org/licenses/MIT) 发布。

## 📅 Roadmap & Updates (路线图与更新)

- **[2025.12]** 📄 论文发布于 [arXiv](https://arxiv.org/abs/2512.23515)。
- **[2026.09]** 🧩 代码发布：qlib 单因子回测、因子描述生成（OpenRouter）、Alpha-R1 推理、输出解析、端到端策略回测，以及 GRPO 训练配置 + 参考奖励实现。
  - ✅ 推理代码（Alpha Screening Pipeline）
  - ✅ 模型权重（[`FinStep/Alpha-R1`](https://huggingface.co/FinStep/Alpha-R1)）
- **[2026.09]** 🔄 数据层直连化：移除 qlib/CSV 中间链路，全链路 TDengine 直连 + GPU 因子计算；universe 排除北交所；新增环境体检技能 `check-alpha-r1-env`。

_欢迎 ⭐ Star 本仓库，获取最新进展！_
