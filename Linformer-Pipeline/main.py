import torch
from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import gym
import set_config
import os
from tensorboardX import SummaryWriter
from data.DataManager import DataManager
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import argparse
import pandas as pd
import time
import numpy as np
from sklearn.metrics import accuracy_score

DATASET_IDS = {37:"diabetes"} ## change or add more data id, name instances
CRITERION = CrossEntropyLoss()
OPTIMIZER = Adam

MODELS = {
    "LIN_TABPFN": TabPFNClassifier(batch_size_inference=5),
    "RF": RandomForestClassifier(),
    "DT": DecisionTreeClassifier() 
}

train_logs = {
    "epochs": [],
    "loss" : [],
    "accuracy": []
}

model_evaluation_logs = {
    "model_name": [],
    "dataset_name": [],
    "time": [],
    "acc": []

}
valid_logs = {
    "epochs": [],
    "accuracy" : [],
    "time": []
}
other_model_logs = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Model training configuration")

    parser.add_argument("--aug_flag", type=str, choices=["gan", "rag"], default="", help="Augmentation flag")
    parser.add_argument("--model_config_flag", type=str, choices=["linformer"], default="linformer", help="Model configuration flag")

    parser.add_argument("--gan_epochs", type=int, default=50, help="Number of GAN training epochs")
    
    parser.add_argument("--num_folds", type=int, default=3, help="Number of folds for cross-validation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")

    parser.add_argument("--train_id", type=str, default="model", help="Training ID based on applied flags")
    parser.add_argument("--target_col_name", type=str, default="survived", help="Training ID based on applied flags")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--test_path",type=str, default="finetune_model/model.pth", help="Model Path for testing")

    args = parser.parse_args()
    return args

def evaluation(MODEL_PATH,model_config_flag,X_test,y_test, X_train, y_train,MODELS):
    '''
    Test Evaluation for all models
    '''
    for model_id, model in MODELS.items():
        if model_id == "LIN_TABPFN":
            model = set_config.set_model_config(model,flag=model_config_flag)
            model.model[2].load_state_dict(torch.load(MODEL_PATH))
            X = pd.concat((X_train, X_test), axis=0)
            y = pd.concat((y_train, y_test), axis=0)
            print("Model: ", model_id)
            gym.validation(model.model[2],(X,y),num_cls=2,log=False,single_eval_pos=len(X_train))
        else:
            start_time = time.time()
            model.fit(X_train, y_train)
            y_preds = model.predict_proba(X_test)
            total_time = time.time()-start_time

            accuracy = accuracy_score(y_test, np.argmax(y_preds, axis=1))

            print("Model: ", model_id)
            print("Time: ", total_time)
            print("Accuracy: ", accuracy)

if __name__ == "__main__":
    args = parse_args()
    for dataset_id, dataset_name in DATASET_IDS.items():
        
        MODEL_PATH  = "finetune_model/" + args.train_id +"_" + dataset_name + ".pth"
        
        if os.path.exists(MODEL_PATH):
            print("Evaluation of Models on test set...")
            X_test,y_test, X_train, y_train = gym.test_dataset(dataset_id)
            evaluation(MODEL_PATH,args.model_config_flag,X_test,y_test, X_train, y_train,MODELS)

        else:
            print("Training TabPFN model because model path does not exist")
            print("Rerun file for evaluation and comparison process")
            print("Dataset Name: ", dataset_name)
            
            data_manager = DataManager(
                dir_path="data/dataset",
                dataset_id=dataset_id
                )
            data_k_folded = data_manager.k_fold_train_test_split(
                    k_folds=args.num_folds,
                    val_size=0.2,
                    random_state=1,
                )
            
            model = MODELS["LIN_TABPFN"]
            model = set_config.set_model_config(model,flag=args.model_config_flag)
            
            if os.path.exists(MODEL_PATH):
                print("loading weights")
                model.model[2].load_state_dict(torch.load(MODEL_PATH))

            for f in range(args.num_folds):
                
                fold = data_k_folded[f]
                train_data = fold["train"]
                val_data = fold["val"]
                test_data = fold["test"]

                train_X, train_y = gym.df_wrapper(train_data)
                val_X, val_y = gym.df_wrapper(val_data)
                test_X, test_y = gym.df_wrapper(test_data)
                
                log_dir = os.path.join("runs", dataset_name,str(f)+args.train_id)
                writer = SummaryWriter(log_dir) 

                _,_,train_X, train_y = set_config.set_augmentator(train_X=train_X, 
                                                                    train_y=train_y,
                                                                    test_X=test_X,
                                                                    val_X=val_X,
                                                                    target_col_name=train_y.name,
                                                                    device=args.device,
                                                                    flag=args.aug_flag,
                                                                    gan_epochs=args.gan_epochs)

                # SPLITTING VALIDATION DATA FOR EVALUATION PURPOSES

                train_dataloader = DataLoader(dataset=TensorDataset(
                    torch.tensor(train_X.values,dtype=torch.float32),
                    torch.tensor(train_y.values,dtype=torch.float32)),
                    shuffle=True,batch_size=512)
                start_time = time.time()
                gym.train(
                    model=model.model[2],
                    train_dataloader=train_dataloader,
                    valid_dataset=(val_X, val_y),
                    num_cls=args.num_classes,
                    epochs=args.epochs,
                    criterion=CRITERION,
                    optimizer=OPTIMIZER,
                    lr=args.lr,
                    device=args.device,
                    writer=writer,
                    flag=args.model_config_flag,
                    train_logs=train_logs,
                    valid_logs=valid_logs,
                    model_path=MODEL_PATH
                    ) 
                end_time = time.time()
                print("Total Train time: ", end_time-start_time) 
                                
                

                


