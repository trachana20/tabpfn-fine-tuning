from __future__ import annotations

import wandb


class WandBLogger:
    def __init__(self, project_name):
        self.project_name = project_name

    def _setup_wandb(
        self,
        setup_config,
    ):
        wandb.login()
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.project_name,
            # track hyperparameters and run metadata
            config=setup_config,
        )

    def log(self, metrics, step):
        wandb.log(metrics, step=step)
