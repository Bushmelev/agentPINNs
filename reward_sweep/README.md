# Reward Sweep

This folder runs one `fixed` baseline and then trains `tiny_loss_weight` once per
agent reward. It writes per-run method artifacts plus sweep-level comparison
plots.

Run from the repository root:

```bash
.venv/bin/python reward_sweep/run_reward_sweep.py
```

Rewards are configured in `base_config.json` under `sweep_rewards`.

Useful overrides:

```bash
.venv/bin/python reward_sweep/run_reward_sweep.py --steps 2000
.venv/bin/python reward_sweep/run_reward_sweep.py --optimizer-mode adam_lbfgs --adam-steps 6000 --lbfgs-steps 2000
.venv/bin/python reward_sweep/run_reward_sweep.py --rewards normalized_baseline_gap_delta,relative_l2_baseline_gap_delta
```

Default output:

```text
artifacts/reward_sweep/<timestamp>/
```

Sweep-level plots are in:

```text
artifacts/reward_sweep/<timestamp>/burgers/reward_sweep/plots/
```
