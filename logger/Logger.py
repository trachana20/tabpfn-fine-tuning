from __future__ import annotations

import wandb
import os
import json


class Logger:
    def __init__(self, project_name, log_wandb, results_path):
        self.project_name = project_name
        self.evaluation_data = {}
        self.current_active_incumbent = None
        self.log_wandb = log_wandb
        self.results_path = results_path

    def setup_wandb(self, setup_config):
        wandb.login()
        # start a new wandb run to track this script

    def register_incumbent(self, random_state, dataset_id, fold_i, model):
        random_state_key = f"random_state_{random_state}"
        dataset_id_key = f"dataset_{dataset_id}"
        fold_i_key = f"fold_{fold_i}"
        model_key = f"{type(model).__name__}"

        if random_state_key not in self.evaluation_data:
            self.evaluation_data[random_state_key] = {}

        if dataset_id_key not in self.evaluation_data[random_state_key]:
            self.evaluation_data[random_state_key][dataset_id_key] = {}

        if fold_i_key not in self.evaluation_data[random_state_key][dataset_id_key]:
            self.evaluation_data[random_state_key][dataset_id_key][fold_i_key] = {}

        self.evaluation_data[random_state_key][dataset_id_key][fold_i_key][
            model_key
        ] = []
        self.current_active_incumbent = self.evaluation_data[random_state_key][
            dataset_id_key
        ][fold_i_key][model_key]

        config = {
            "random_state": random_state,
            "dataset_id": dataset_id,
            "fold_i": fold_i,
            "model": model_key,
        }

        if self.log_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project=self.project_name,
                # track hyperparameters and run metadata
                config=config,
            )

    def update_traing_metrics(self, performance_metrics, step):
        self.current_active_incumbent.append(performance_metrics)
        if self.log_wandb:
            self._log(metrics=performance_metrics, step=step)

    def _log(self, metrics, step):
        wandb.log(metrics, step=step)

    def save_results(self):
        # Check if evaluation_data is not empty
        if self.evaluation_data:
            # Create the directory if it doesn't exist

            os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

            file_name = self.results_path + "fit_predict.json"
            # Open the file in write mode and save the data as JSON
            with open(file_name, "w") as file:
                json.dump(self.evaluation_data, file)
        else:
            print("No evaluation data to save.")
