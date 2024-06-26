from __future__ import annotations

import pickle
import time
import warnings
from pathlib import Path

import torch
from evaluation.model_evaluation import classification_performance_metrics
from gym.utils import (
    average_epoch_metrics,
    to_numpy,
    update_epoch_metrics_with_batch_metrics,
)
from models.FineTuneTabPFNClassifier import FineTuneTabPFNClassifier
from torch import nn
from tqdm import tqdm

import loralib as lora
from torch.utils.tensorboard import SummaryWriter
# Filter out UserWarnings from torch.utils.checkpoint
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.*",
)
from contextlib import contextmanager


@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.*",
        )
        yield


class Trainer:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.writer = SummaryWriter()

    def fine_tune_model(
        self,
        train_loader,
        val_dataset,
        fine_tune_type,
        device,
        fine_tuning_configuration,
        **kwargs,
    ):
        # This function materializes/instantiates the tabpfn classifier
        # either an existing model is loaded or a tabpfn is finetuned
        weights_path = kwargs.get("architectural", {}).get("weights_path", None)

        tabpfn_classifier = kwargs.get("architectural", {}).get(
            "tabpfn_classifier",
            None,
        )

        # 1. check if weights_path exists and if so load the fine_tune model
        if Path(weights_path).exists():
            return FineTuneTabPFNClassifier(
                tabpfn_classifier=tabpfn_classifier,
                weights_path=weights_path,
            )

        # 2. if weights_path does not exist, fine_tune the model
        # call the correct fine tuning function
        if fine_tune_type == "LORA-finetuning":
            # register new writer in visualizer, which tracks the training process

            writer_name = f"{fine_tune_type}_{fine_tuning_configuration['augmentation']}_{fine_tuning_configuration['dataset_name']}_{fine_tuning_configuration['fold']}_{fine_tuning_configuration['random_state']}"
            self.visualizer.register_writer(
                writer_name=writer_name,
                config={
                    **kwargs,
                    **fine_tuning_configuration,
                    "fine_tune_type": fine_tune_type,
                },
            )

            return self.full_weight_fine_tuning(
                tabpfn_classifier=tabpfn_classifier.tabpfn_classifier,
                train_loader=train_loader,
                val_dataset=val_dataset,
                weights_path=weights_path,
                device=device,
                **kwargs,
            )
        else:
            raise ValueError(f"Fine tune type {fine_tune_type} not supported")

    def _store_model_weights_and_training_metrics(
        self,
        state_dict,
        weights_path,
        training_metrics,
        training_metrics_path,
    ):
        Path(Path(weights_path).parent).mkdir(parents=True, exist_ok=True)
        # torch.save(state_dict, weights_path)
        ## !! LORA SAVE PATH CHANGED
        torch.save(state_dict, weights_path)
        # Save the datasets to a pickle file
        with open(training_metrics_path, "wb") as file:
            pickle.dump(training_metrics, file)

    def full_weight_fine_tuning(
        self,
        tabpfn_classifier,
        train_loader,
        val_dataset,
        weights_path,
        architectural,
        training,
        device,
    ):
        tabpfn_model = tabpfn_classifier.model[2]
        # measure validation performance before any fine-tuning
        # Call validation function after each epoch
        with torch.no_grad():
            validation_metrics = self.validate_tabpfn_model(
                tabpfn_classifier=tabpfn_classifier,
                tabpfn_model=tabpfn_model,
                val_dataset=val_dataset,
                train_dataset=train_loader.dataset,
            )        
        current_lowest_log_loss = validation_metrics["log_loss"]
        print("Evaluation before training: ", current_lowest_log_loss)

        print("Starting Training Process...")

        update_lowest_model_counts = 0

        tabpfn_model.train()

        optimizer = training["optimizer"](
                params=tabpfn_model.parameters(),
                lr=training["learning_rate"],
            )         

        criterion = training["criterion"]()
        training.get("early_stopping_threshold", 0.1)
        
        tabpfn_model.to(device)
        tabpfn_classifier.device = device

        num_classes = train_loader.dataset.num_classes


        current_state_dict = tabpfn_model.state_dict()

        

        # start fine-tuning iteration
        # in each iteration, we first load a databatch and compute the
        # gradients to update the models weights. Then we evaluate the
        # validation performance on a validation dataset. If the log_loss
        # performance on the validation dataset improves more than the
        # early stopping threshold then we cache the models statedict of
        # the current iteration.
        training_metrics = []

        for epoch_i in tqdm(range(training["epochs"])):
            epoch_metrics = {}
            for _batch_i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x = x.transpose(0, 1).to(device)  # batch_first = False
                y = y.transpose(0, 1).to(device)  # batch_first = False

                single_eval_pos = int(x.shape[0] * 0.8)  # 80% for training 20% query
                y_train = y[:single_eval_pos]
                y_query = y[single_eval_pos:]

                # prepare x_data with padding with zeros up to 100 features
                x_data = nn.functional.pad(x, (0, 100 - x.shape[-1])).float()

                # y_train shape: sequence_length
                #  -> sequence_length, batch_size=1
                y_data = y_train.unsqueeze(1).to(device).float()

                y_preds = tabpfn_model(
                    (x_data, y_data),
                    single_eval_pos=single_eval_pos,
                ).reshape(-1, 10)[:, :num_classes]

                # y_preds are raw logits -> normalize with softmax
                # TODO find out what softmax temperature TabPFN uses
                y_preds = y_preds.softmax(dim=-1)

                y_query = y_query.long().flatten()
                loss = criterion(y_preds, y_query)

                # compute gradients and store in nn.Parameters
                loss.backward()

                ## Monitor Grad Stats
                # for name, param in tabpfn_model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name} gradient mean: {param.grad.mean()}")
                #         print(f"{name} gradient std: {param.grad.std()}")

                    
                # update weights with stored gradients
                optimizer.step()

                # compute batch-specific performance metrics
                batch_metrics = classification_performance_metrics(
                    y_preds=to_numpy(y_preds),
                    y_true=to_numpy(y_query),
                )
                # enrich batch metrics with additional information
                batch_metrics["train_loss"] = loss.item()
                batch_metrics["num_classes"] = y_preds.shape[-1]
                batch_metrics["num_features"] = x_data.shape[-1]
                batch_metrics["sequence_length"] = x.shape[0]

                # aggregate metrics over multiple batches
                epoch_metrics = update_epoch_metrics_with_batch_metrics(
                    epoch_metrics=epoch_metrics,
                    batch_metrics=batch_metrics,
                )

            # aggregate over batches and visualize metrics
            epoch_metrics = average_epoch_metrics(epoch_metrics=epoch_metrics)

            # call visualization

            for metric, value in epoch_metrics.items():
                self.visualizer.update_scalar_value(
                    f"training/{metric}",
                    value,
                    epoch=epoch_i,
                )
                self.writer.add_scalar(f"training/{metric}", value, epoch_i)


            # Call validation function after each epoch
            with torch.no_grad():
                validation_metrics = self.validate_tabpfn_model(
                    tabpfn_classifier=tabpfn_classifier,
                    tabpfn_model=tabpfn_model,
                    val_dataset=val_dataset,
                    train_dataset=train_loader.dataset,
                )
                # visualize validation metrics

                for metric, value in validation_metrics.items():
                    self.visualizer.update_scalar_value(
                        f"validation/{metric}",
                        value,
                        epoch=epoch_i,
                    )
                    self.writer.add_scalar(f"validation/{metric}", value, epoch_i)
                # Validation early stopping. We use the log loss performance
                # on a validation set to estimate the models fine-tuning performance

                if current_lowest_log_loss > validation_metrics["log_loss"]:
                    # the lower the better

                    current_state_dict = tabpfn_model.state_dict()
                    current_lowest_log_loss = validation_metrics["log_loss"]
                    update_lowest_model_counts += 1

            # store training metrics in one file to store later
            training_metrics.append(
                {
                    **{f"training_{k}": v for k, v in epoch_metrics.items()},
                    **{f"validation_{k}": v for k, v in validation_metrics.items()},
                    "update_lowest_model_counts": update_lowest_model_counts,
                },
            )

            # print("Validation Accuracy", epoch_i,training_metrics[epoch_i]["validation_accuracy"])
            # print("Validation Log Loss", epoch_i,training_metrics[epoch_i]["validation_log_loss"])

            print(training_metrics[epoch_i])


        # save weights with lowest validation performance
        self._store_model_weights_and_training_metrics(
            state_dict = current_state_dict,
            weights_path=weights_path,
            training_metrics=training_metrics,
            training_metrics_path=f"{weights_path[:-4]}_training_metrics.pkl",
        )

        # load tabpfn model with best validation performance weights
        ## LORA checkpoints
        tabpfn_model.load_state_dict(current_state_dict, strict=False)

        # set nn.Module back into eval model, so no further gradients are computed
        tabpfn_model.eval()

        # just to be sure the nn.Moduel, which we finetuned is really used
        tabpfn_classifier.model = (None, None, tabpfn_model)
        return tabpfn_classifier

    def validate_tabpfn_model(
        self,
        tabpfn_classifier,
        tabpfn_model,
        val_dataset,
        train_dataset,
    ):
        # for the validation we insert the nn.Module back into the tabpfnclassifier
        # instance so we mimic the exact way that TabPFN does predictions
        # (pre-processing, softmax temperature, etc.)
        tabpfn_classifier.model = (None, None, tabpfn_model)

        single_eval_pos = int(2 / 3 * val_dataset.number_rows)

        # for the proper evaluation we also use numpy arrays instead
        # of tensors. 1. we don't run into the risk of updating weights
        # accidentally and we mimic again the later use-case

        x_train = val_dataset.features[:single_eval_pos]
        y_train = val_dataset.labels[:single_eval_pos]
        x_query = val_dataset.features[single_eval_pos:]
        y_true = val_dataset.labels[single_eval_pos:]

        start_time = time.time()
        tabpfn_classifier.fit(x_train, y_train)
        fitting_time = time.time() - start_time

        start_time = time.time()
        with suppress_warnings():
            y_preds = tabpfn_classifier.predict_proba(x_query)
        prediction_time = time.time()

        # print(y_preds, y_true)

        # for this evaluation we do not need to convert to numpy
        # arrays as the predictions are returned as numpy arrays
        val_metrics = classification_performance_metrics(
            y_preds=y_preds,
            y_true=y_true,
        )

        # enrich validation metrics with additional information
        val_metrics["fitting_time"] = fitting_time
        val_metrics["prediction_time"] = prediction_time

        return val_metrics



def return_model_named_parameters(model):
    lora_layers_names = []
    non_lora_layers_names = []

    lora_layers_params = []
    non_lora_layers_params = []

    for name, params in model.named_parameters():
        if "lora" in name:
            lora_layers_names.append(name)
            lora_layers_params.append({"params": params})
        else:
            non_lora_layers_names.append(name)
            non_lora_layers_params.append({"params": params})

    return lora_layers_names, non_lora_layers_names, lora_layers_params, non_lora_layers_params