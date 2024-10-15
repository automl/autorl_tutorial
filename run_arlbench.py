"""Console script for arlbench."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import logging
import sys
import traceback
from typing import TYPE_CHECKING

import hydra
import jax
import os
from arlbench import AutoRLEnv
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("divide", lambda x, y: x / y, replace=True)


@hydra.main(version_base=None, config_path="configs", config_name="arlbench_rs")
def execute(cfg: DictConfig):
    """Helper function for nice logging and error handling."""
    logging.basicConfig(
        filename="job.log", format="%(asctime)s %(message)s", filemode="w"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if cfg.jax_enable_x64:
        logger.info("Enabling x64 support for JAX.")
        jax.config.update("jax_enable_x64", True)
    try:
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return run(cfg, logger)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def run(cfg: DictConfig, logger: logging.Logger):
    """Console script for arlbench."""
    if "load" in cfg and cfg["load"]:
        checkpoint_path = os.path.join(
            cfg["load"],
            cfg["autorl"]["checkpoint_name"],
            "default_checkpoint_c_episode_1_step_1",
        )
    else:
        checkpoint_path = None

    # We check if we need to save a checkpoint for HyperPBT
    # If so, we need to adapt the autorl config accordingly
    if "save" in cfg and cfg["save"]:
        cfg["autorl"]["checkpoint_dir"] = str(cfg["save"]).replace(".pt", "")
        if cfg["algorithm"] == "PPO":
            cfg["autorl"]["checkpoint"] = ["opt_state", "params"]
        else:
            cfg["autorl"]["checkpoint"] = ["opt_state", "params", "buffer"]

    # Here, we define how the AutoRLEnv should behave
    env = AutoRLEnv(cfg["autorl"])
    _ = env.reset()

    if logger:
        logger.info("Your AutoRL config is:")
        logger.info(OmegaConf.to_yaml(cfg["autorl"]))
        logger.info("Training started.")
    _, objectives, _, _, info = env.step(cfg["hp_config"], checkpoint_path=checkpoint_path)
    if logger:
        logger.info("Training finished.")

    # Additionally, we store the evaluation rewards we had during training
    info["train_info_df"].to_csv("evaluation.csv", index=False)

    if len(objectives) == 1:
        objectives = objectives[next(iter(objectives.keys()))]
    else:
        objectives = tuple(objectives.values())

    with open("./performance.csv", "w+") as f:
        f.write(str(objectives))
    with open("./done.txt", "w+") as f:
        f.write("yes")

    print(f"Result: {objectives}")

    return objectives


if __name__ == "__main__":
    sys.exit(execute())  # pragma: no cover