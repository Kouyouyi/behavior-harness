#!/usr/bin/env python

import logging
import math
from collections.abc import Sequence
from numbers import Real
from pathlib import Path

from termcolor import colored

from lerobot.common.wandb_utils import cfg_to_group
from lerobot.configs.train import TrainPipelineConfig


class SwanLabLogger:
    """Log supervised training metrics and configuration to SwanLab."""

    def __init__(self, cfg: TrainPipelineConfig):
        import swanlab

        self.cfg = cfg.swanlab
        logdir = Path(self.cfg.logdir) if self.cfg.logdir else cfg.output_dir / "swanlab"
        mode = self.cfg.mode if self.cfg.mode in {"online", "local", "offline", "disabled"} else "online"
        run = swanlab.init(
            project=self.cfg.project,
            workspace=self.cfg.workspace,
            experiment_name=cfg.job_name,
            description=self.cfg.description,
            tags=cfg_to_group(cfg, return_list=True) if self.cfg.add_tags else None,
            config=cfg.to_dict(),
            logdir=str(logdir),
            mode=mode,
            id=self.cfg.run_id,
            resume="must" if cfg.resume and self.cfg.run_id else None,
        )
        public = getattr(run, "public", None)
        run_id = getattr(public, "run_id", None) or getattr(run, "id", None)
        if run_id:
            cfg.swanlab.run_id = run_id
        logging.info(colored("Logs will be synced with SwanLab.", "blue", attrs=["bold"]))
        cloud = getattr(public, "cloud", None)
        experiment_url = getattr(cloud, "experiment_url", None)
        if experiment_url:
            logging.info(f"Track this run --> {colored(experiment_url, 'yellow', attrs=['bold'])}")
        self._swanlab = swanlab

    def log_dict(self, d: dict, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        batch_data = {}
        for key, value in d.items():
            metric_key = f"{mode}/{key}"
            if isinstance(value, Real):
                batch_data[metric_key] = float(value)
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                values = [float(item) for item in value if isinstance(item, Real)]
                if len(values) == len(value) and values:
                    batch_data[f"{metric_key}/mean"] = sum(values) / len(values)
                    batch_data[f"{metric_key}/min"] = min(values)
                    batch_data[f"{metric_key}/max"] = max(values)
                    batch_data[f"{metric_key}/l2"] = math.sqrt(
                        sum(item * item for item in values) / len(values)
                    )
                    for index, item in enumerate(values):
                        batch_data[f"{metric_key}/{index:03d}"] = item
                    continue
            logging.warning(
                'SwanLab logging of key "%s" ignored: type "%s" is not scalar or numeric sequence.',
                key,
                type(value),
            )
        if batch_data:
            self._swanlab.log(batch_data, step=step)

    def finish(self) -> None:
        self._swanlab.finish()
