# Training

Alpha-R1 is trained with Group Relative Policy Optimization (GRPO) on top of
Qwen3-8B, using the [verl](https://github.com/volcengine/verl) RLHF framework
and a market-feedback reward (paper Section 3.4).

## Contents

- `configs/grpo_alpha_r1.yaml` — training hyperparameters (hydra-style config
  for `verl.trainer.main_ppo`): GRPO with group size 8, asymmetric clip ratios
  0.15/0.25, lr 1e-6, KL-free reward shaping, vLLM rollout.
- `reward.py` — simplified **reference implementation** of the market-feedback
  reward: structural validity penalties, linear-model stock scoring, top-N
  portfolio excess return over the holding period (Eq. 7), and an optional
  LLM-as-judge consistency adjustment (Eq. 8-9, stubbed off by default).

## Plugging in the reward

verl supports external reward functions out of the box:

```bash
python -m verl.trainer.main_ppo \
    --config-path training/configs --config-name grpo_alpha_r1 \
    custom_reward_function.path=training/reward.py \
    custom_reward_function.name=compute_score_batch
```

`reward.py` expects factor values, price data, benchmark series and offline
estimated linear coefficients (betas); see the `RewardContext` docstring for
the expected file formats. Betas are estimated on the pre-training window
(2020-2023 in the paper) by cross-sectional regression of forward returns on
factor values — `scripts/train_linear_model.py` produces a compatible
`betas.csv` from TDengine-direct data.

## Notes

- The reward script is a reference: the full production pipeline (slot-rotation
  VWAP execution, judge model, distributed batch reward) is left for users to
  adapt to their infrastructure.
- Training prompts are built from the factor descriptions produced by this
  repository's generation pipeline (`result/alpha_des/alphaNNN.txt`), sampled
  as random 40-factor subsets per decision day (paper Section 4.1.1).
