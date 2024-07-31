import optuna
from sklearn.metrics import accuracy_score

from data.DataManager import DataManager
from gym.Trainer import Trainer
from main import modelkwargs_dict, setup_config, preprocessor, visualizer
from utils import set_seed_globally
from torch.utils.data import DataLoader
from data.FullRealDataDataset import FullRealDataDataset
from data.RealDataDataset import RealDataDataset
from preprocessing.PreProcessor import PreProcessor

Trainer = Trainer(visualizer=visualizer)

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32])
    epochs = trial.suggest_int("epochs", 50, 200)

    # Update the model training hyperparameters
    modelkwargs_dict["FineTuneTabPFNClassifier_full_weight"]["training"].update({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
    })

    # Run your training and evaluation pipeline here
    # This is a simplified version of your training loop

    # Example training loop
    set_seed_globally(setup_config["random_states"][0])
    dataset_id, dataset_name = next(iter(setup_config["dataset_mapping"].items()))
    data_manager = DataManager(
        dir_path=setup_config["dataset_dir"] + dataset_name + ".csv",
        dataset_id=dataset_id if dataset_id != 0 else None,
    )
    data_k_folded = data_manager.k_fold_train_test_split(
        k_folds=setup_config["k_folds"],
        val_size=setup_config["val_size"],
        random_state=setup_config["random_states"][0],
    )

    fold = data_k_folded[0]  # Take the first fold for simplicity
    train_data = fold["train"]
    val_data = fold["val"]

    train_data["data"] = preprocessor.augment_dataset(
        train_data["data"], train_data["data"], train_data["target"]
    )
    train_dataset = FullRealDataDataset(
        data=train_data["data"],
        target=train_data["target"],
        name=train_data["name"],
    )

    batch_size = train_dataset.number_rows if batch_size == "full" else batch_size

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset._collate_fn,
        batch_size=batch_size,
    )

    model = Trainer.fine_tune_model(
        train_loader=train_loader,
        val_dataset=RealDataDataset(data=val_data["data"], target=val_data["target"], name=val_data["name"]),
        fine_tune_type=modelkwargs_dict["FineTuneTabPFNClassifier_full_weight"]["architectural"]["fine_tune_type"],
        device=setup_config["device"],
        fine_tuning_configuration={},
        **modelkwargs_dict["FineTuneTabPFNClassifier_full_weight"]["training"],
    )

    val_predictions = model.predict(val_data["data"])
    accuracy = accuracy_score(val_data["target"], val_predictions)

    return accuracy


# Run the optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print(f"Best hyperparameters: {study.best_params}")
