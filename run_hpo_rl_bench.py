import sys
sys.path.append("HPO-RL-Bench")
from benchmark_handler import BenchmarkHandler

import hydra
import numpy as np
from pathlib import Path

@hydra.main(config_path="configs", config_name="hpo_rl_bench", version_base="1.1")
def evaluate_hpo_rl_bench(config):
  benchmark = BenchmarkHandler(data_path=(Path(__file__).resolve().parent / "HPO-RL-Bench"))
  benchmark.set_env_space_seed(search_space=config["algorithm"], environment=config["environment"], seed=config["seed"])
  return np.mean(benchmark.get_metrics(config["hp_config"], budget=config["budget"])[config["metric_key"]]).item()

if __name__ == "__main__":
    evaluate_hpo_rl_bench()