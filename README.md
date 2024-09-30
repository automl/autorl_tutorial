# AutoRL Tutorial
This repository contains utilities for André Biedenkapp's & Theresa Eimer's tutorial "Beyond Trial & Error: A Tutorial on Automated Reinforcement Learning". The full tutorial is best experienced on Colab TODO.

## Setup
We recommend installing the dependencies in a virtual environment, e.g. using uv:
```bash
pip install uv
uv venv --python 3.10
source .venv/bin/activate
```

Then you can use our utilities to install the dependecies:
```bash
make install
```

This will install all packages, but won't download any additional data for HPO-RL Bench. By default only static PPO data for Pong and Enduro are available. If you want additional data, you need to download the HPO-RL Bench data from [this link](https://drive.google.com/file/d/1AW5_6xGGiklteZgyyDBxSsf6kOLuFPkO/view?usp=share_link).

## HPO using HPO-RL-Bench
The *'run_hpo_rl_bench.py'* script interfaces HPO-RL Bench. By overriding the 'environment', 'seed' and 'algorithm' arguments, you can switch to a different set of results. 'budget' determines at which point in training the result is queried:
```bash
python run_hpo_rl_bench.py env=Enduro-v0 budget=10 seed=1
```

We can do HPO by using Hypersweeper in combination with this script. In the 'configs' directory, you'll find pre-configured versions of random search, SMAC and DEHB. You can run them by specifying the config you want to use and adding the '-m' flag:
```bash
python run_hpo_rl_bench.py --config-name=hpo_rl_bench_smac -m
```

## HPO using ARLBench
With 'run_arlbench.py' you can run RL agents with ARLBench. Since we actually execute the runs here, there are various configuration options. We pre-configure some of them in the 'configs/environment' and 'configs/algorithm' directories. You can switch between them similarly as above:
```bash
python run_arlbench.py environment=xland_empty_random algorithm=ppo autorl.seed=2
```

HPO with Hypersweeper works the same way as for HPO-RL Bench. We pre-configured random search, SMAC and PB2:
```bash
python run_arlbench.py --config-name=arlbench_pb2 -m
```