# DynamoTest

This repository contains the DynamoTest approach and related baselines for generating adversarial traffic scenarios against a pre‑trained “ego” driving agent using Highway‑Env. The main entry point is `train_adversary.py`.

## Highlights
- Custom multi‑agent environment wrapper registered as `highwayadv-v0`.
- Multiple adversary strategies: DQN (CleanRL), SAC (CleanRL), Contextual MAB, Random, and GA‑based tester.
- Per‑run statistics and model checkpoints for reproducibility.

## Requirements
- Python 3.9–3.11 recommended
- PyTorch (required by Stable‑Baselines3)
- Python packages:
  - `gymnasium`, `highway-env`, `stable-baselines3`, `numpy`
  - `tyro` (CLI parsing for adversary configs)
  - `mabwiser` (MAB baseline)
  - `rl-agents` (for some ego policies loaded via config)

## Installation
It is recommended to use a virtual environment.

1) Create and activate a venv (example):
```
python -m venv .venv
. .venv/Scripts/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
```

2) Install core dependencies:
```
pip install -r requirement.txt
```
```
pip install gymnasium==0.29.0 tyro
```

3) Install PyTorch per your platform (CPU/GPU):
- See https://pytorch.org for the correct command (CUDA/CPU).


## Quick Start: Run `train_adversary.py`
Defaults inside the script:
- Episodes: `4005`
- Runs: `10`
- Ego types: `["baseline_defensive"]`
- Adversary algorithm: `"dqn"`

From the repo root:
```
python train_adversary.py
```

### What gets produced
- Stats: `stats/final_26_oct_uc2/rl_<date>-<episodes>-<algo>_<ego>_new_uc2_no_novelty/run_<n>/`
- Weights: `weights/final_26_oct_uc2/rl_<date>-<episodes>-<algo>_<ego>_new_uc2_no_novelty/run_<n>/`
- For DQN/SAC adversaries, checkpoints are saved every 200 episodes by default.

## Choosing the Adversary
Set `algo` in `train_adversary.py`:
- `"dqn"` (default): Uses tyro to parse flags defined in `agents/dqn_agent_cleanrl.py::Args`.
  - Example additional flags:
    - `python train_adversary.py --learning-rate 3e-4 --batch-size 256`
- `"sac"`: Uses flags from `agents/sac_agent_cleanrl.py::ArgsSac`.
- `"cmab"`: Contextual MAB (`mabwiser`).
- `"random"`: Random policy.
- `"ga"`: GA‑based tester (`ga/test_generator.py`) with `tester_config.yaml`.

Note: tyro only parses flags relevant to the selected adversary algorithm.

## Choosing the Ego Agent
Controlled via `ego_types` in `train_adversary.py`:
- Options include: `"baseline"`, `"retrain"`, `"aggressive"`, `"baseline_defensive"`, `"robust"`.
- Some options load an RL‑Agents DQN via `models/dqn.json` + checkpoints in `out/`.
- Others load SB3 DQN zip files from `models/`.
- Ensure referenced files exist, or switch `ego_types` to match available assets.

## Editing Experiment Settings
In `train_adversary.py`:
- `EPISODES`: Number of episodes per run.
- `runs`: Number of independent runs.
- `algo`: One of `"dqn" | "sac" | "cmab" | "random" | "ga"`.
- `ego_types`: List of ego setups (e.g., `["baseline_defensive"]`).
- `save_interval`: Checkpoint cadence (default `200`).

For DQN/SAC, additional hyperparameters can be passed via CLI flags (handled by tyro through the `Args`/`ArgsSac` dataclasses in the agent files).

## Outputs and Reproducibility
- Episode statistics are saved via `StatRecorder` under the `stats/` directory.
- Adversary checkpoints saved under `weights/` as `adversary_<episode>`.
- The script copies `envs/highway_env_adv.py` and, for DQN, `agents/dqn_agent_cleanrl.py` into the run’s stats folder to capture the exact code used.

## Troubleshooting
- Import errors for `rl-agents`:
  - Install from the Highway‑Env repository as shown above.
- Missing model checkpoints:
  - The script references specific files under `models/` and `out/`. Update those paths or adjust `ego_types`.
- PyTorch/CUDA issues:
  - Install the correct PyTorch build (CPU or matching your CUDA version).
- `highwayadv-v0` not found:
  - Ensure `envs/highway_env_adv.py` exists and run the script from the repo root so relative imports resolve.

## Notes
- Keep models and outputs versioned per experiment.
- Avoid changing `envs/highway_env_adv.py` mid‑experiment; it is copied to the stats folder for auditing.
