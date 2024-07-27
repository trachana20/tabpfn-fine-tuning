import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os
import time
import tqdm
from tensorboardX import SummaryWriter
from tabpfn import TabPFNClassifier

from data.DataManager import DataManager
from data.RealDataDataset import RealDataDataset
from data.FullRealDataDataset import FullRealDataDataset
import peft
from performer_pytorch import SelfAttention
from preprocessing.PreProcessor import PreProcessor
import warnings
warnings.filterwarnings("ignore")

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


preprocessor = PreProcessor()

def train(
        model,
        train_dataloader,
        valid_dataset,
        epochs,
        criterion,
        optimizer,
        device,
        writer
):
    optimizer = optimizer(
            params=model.parameters(),
            lr=TABPFN_HP_GRID["learning_rate"][0],
        )
    for e in range(epochs):
        model.train()
        model.to(device)
        batch_loss = []
        batch_accuracy = []
        for b,(x,y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            if apply_performer:
                x = x.unsqueeze(-2)
                y = y.unsqueeze(-1)
            single_eval_pos = int(x.shape[0] * 0.8)  # 80% for training 20% query
            y_train = y[:single_eval_pos]
            y_query = y[single_eval_pos:]

            x_data = nn.functional.pad(x, (0, 100 - x.shape[-1])).float()
            y_data = y_train.unsqueeze(1).to(device).float()

            y_preds = model(
                    (x_data, y_data),
                    single_eval_pos=single_eval_pos,
            ).reshape(-1, 10)[:, :NUM_CLASSES]

            y_preds = y_preds.softmax(dim=-1)

            y_query = y_query.long().flatten()

            loss = criterion(y_preds, y_query)

            accuracy = accuracy_score(y_query.cpu().detach().numpy(), np.argmax(y_preds.cpu().detach().numpy(), axis=1))

            loss.backward()
            optimizer.step()

            batch_loss.append(loss)
            batch_accuracy.append(accuracy)
        
        avg_loss = sum(batch_loss) / len(train_dataloader)
        avg_accuracy = sum(batch_accuracy) / len(train_dataloader)
        
        train_logs["epochs"].append(e)
        train_logs["loss"].append(avg_loss)
        train_logs["accuracy"].append(avg_accuracy)

        writer.add_scalar('Train/Loss', avg_loss, e)
        writer.add_scalar('Train/Accuracy', avg_accuracy, e)

        print("Train Epoch : {}, Loss {}, Accuracy: {}".format(e,avg_loss,avg_accuracy))
        if e%5==0:
          validation(model, e, valid_dataset,writer)
    
    save_model(model,MODEL_PATH)

def validation(model, epoch, valid_dataset,writer):
    with torch.no_grad():
        X_train, X_test,y_train,y_test = valid_dataset
        model.eval()
        tabpfn = TabPFNClassifier()
        tabpfn.model = (None, None, model)
        start_time = time.time()
        tabpfn.fit(X_train, y_train)
        y_preds = tabpfn.predict_proba(X_test)
        total_time = time.time()-start_time

        accuracy = accuracy_score(y_test, np.argmax(y_preds, axis=1))
        
        valid_logs["epochs"].append(epoch)
        valid_logs["accuracy"].append(accuracy)
        valid_logs["time"].append(total_time)

        print("Validation Epoch: {}, Accuracy: {}, Prediction Time: {}".format(epoch,accuracy,total_time))
        writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        writer.add_scalar('Validation/Time', total_time, epoch)

def save_model(model,path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def model_evaluation(model,model_name,dataset_name,X_train, X_test,y_train,y_test):
    print("Model name ", model_name)
    print("Dataset name", dataset_name)
    
    start_time = time.time()
    model.fit(X_train,y_train)
    y_preds = model.predict_proba(X_test)
    total_time = time.time()-start_time
   
    print("Total Fitting + Prediction time: ", total_time)

    accuracy = accuracy_score(y_test,np.argmax(y_preds,axis=1))
    print("Accuracy: ", accuracy)

    model_evaluation_logs["model_name"].append(model_name)
    model_evaluation_logs["dataset_name"].append(dataset_name)
    model_evaluation_logs["time"].append(time)
    model_evaluation_logs["acc"].append(accuracy)

def get_folds(dataset_id,f):
    data_manager = DataManager(
                dir_path="data/dataset",
                dataset_id=dataset_id,
            )
    data_k_folded = data_manager.k_fold_train_test_split(
        k_folds=K_FOLDS,
        val_size=0.2,
        random_state=1,
        apply_gans=apply_gans,
    )
    fold = data_k_folded[f]
    train_data = fold["train"]
    val_data = fold["val"]
    test_data = fold["test"]

    if apply_cosine_similarity_with_test_set:
        train_data["data"] = preprocessor.augment_dataset(
            train_data["data"], 
            test_data["data"], 
            train_data["target"]
        )

    train_dataset = RealDataDataset(
                        data=train_data["data"],
                        target=train_data["target"],
                        name=train_data["name"],
                    )
                    # validation and test data is never augmented
    val_dataset = RealDataDataset(
        data=val_data["data"],
        target=val_data["target"],
        name=val_data["name"],
    )

    test_dataset = RealDataDataset(
        data=test_data["data"],
        target=test_data["target"],
        name=test_data["name"],
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate script with various flags.")
    parser.add_argument("--apply_lora", type=bool, default=False, help="Set to True to apply LoRA")
    parser.add_argument("--apply_performer", type=bool, default=False, help="Set to True to apply Performer")
    parser.add_argument("--apply_gans", type=bool, default=False, help="Set to True to apply GANs with cosine similarity")
    parser.add_argument("--apply_cosine_similarity_with_test_set", type=bool, default=False, help="Set to True to apply cosine similarity with test set")
    parser.add_argument("--train_model", type=bool, default=True, help="Set to True to train the model")
    parser.add_argument("--train_id", type=str, default="model_performer_lora", help="Set to model id to train the model")
    parser.add_argument("--model_path", type=str, default="finetune_model/model_performer_lora.pth", help="Set to specific model path if you want to load the model")
    parser.add_argument("--dataset_ids", type=dict, default={168746: "Titanic", 9982: "Dress-Sales"}, help="Set to train the model with specific dataset ids")
    parser.add_argument("--device", type=str, default="cuda", help="Set to device to run the model")
    parser.add_argument("--k_folds", type=int, default=5, help="Set to number of k-folds")
    parser.add_argument("--num_classes", type=int, default=2, help="Set to number of classes")
    parser.add_argument("--n_estimators", type=list, default=[100, 500, 1000], help="Set to number of estimators")
    parser.add_argument("--max_depth", type=list, default=[10, 50, 100], help="Set to max depth")
    parser.add_argument("--models", type=dict, default={"OG_TABPFN": TabPFNClassifier(batch_size_inference=5), "RF": RandomForestClassifier(), "DT": DecisionTreeClassifier()}, help="Set to models to evaluate")
    parser.add_argument("--epochs", type=list, default=[10, 100, 1000], help="Set to number of epochs")
    parser.add_argument("--learning_rate", type=list, default=[1e-6], help="Set to learning rate")
    parser.add_argument("--early_stopping", type=list, default=[0.1], help="Set to early stopping")
    parser.add_argument("--criterion", type=list, default=[CrossEntropyLoss()], help="Set to criterion")
    parser.add_argument("--optimizer", type=list, default=[Adam], help="Set to optimizer")
    args = parser.parse_args()

    # Set global variables
    global apply_lora, apply_performer, apply_gans, apply_cosine_similarity_with_test_set,\
        train_model, TRAIN_ID, MODEL_PATH, DATASET_IDS, DEVICE, K_FOLDS, NUM_CLASSES,\
        OTHERS_HP_GRID, MODELS
    apply_lora = args.apply_lora
    apply_performer = args.apply_performer
    apply_gans = args.apply_gans
    apply_cosine_similarity_with_test_set = args.apply_cosine_similarity_with_test_set
    train_model = args.train_model
    TRAIN_ID = args.train_id
    MODEL_PATH = "finetune_model/" + TRAIN_ID + ".pth"
    DATASET_IDS = args.dataset_ids
    NUM_CLASSES = args.num_classes
    K_FOLDS = args.k_folds
    MODELS = args.models
    DEVICE = torch.device("cpu")
    if args.device == "cuda":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OTHERS_HP_GRID = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
    }
    TABPFN_HP_GRID = {
        'epochs' : args.epochs,
        'learning_rate': args.learning_rate,
        'early_stopping':args.early_stopping,
        'criterion': args.criterion,
        'optimizer' : args.optimizer
    }
    if train_model:
        ## Preparing data for training
        print("Starting Train Process")
        model = MODELS["OG_TABPFN"]
        if apply_performer:
            print("Initialising and applying Performer")
            
            num_heads = model.model[2].transformer_encoder.layers[0].self_attn.num_heads 
            embed_dim = model.model[2].transformer_encoder.layers[0].self_attn.embed_dim
            attn = SelfAttention(
                dim = embed_dim,
                heads = num_heads,
                causal = False,
                )
            
            for i in range(len(model.model[2].transformer_encoder.layers)):
                model.model[2].transformer_encoder.layers[i] = attn
        if apply_lora:
            print("Initialising and applying LoRA")
            
            linear_layers = []
            for name, submodule in model.model[2].named_modules():
                if isinstance(submodule, nn.Linear):
                    linear_layers.append(name)

            print(linear_layers)
            config = peft.LoraConfig(
                r=8,
                target_modules=linear_layers
            )

            peft_model = peft.get_peft_model(model.model[2], config)
            peft_model.print_trainable_parameters()

            model.model = (None, None, peft_model) 

    
    ## ABLATION LOOP
    for dataset_id, dataset_name in DATASET_IDS.items():
        print("Dataset Name: ", dataset_name)
        for f in range(K_FOLDS):
            log_dir = os.path.join("runs", dataset_name,str(f)+TRAIN_ID)  # Create a unique log directory for each dataset
            writer = SummaryWriter(log_dir) 
            train_dataset, val_dataset, test_dataset = get_folds(dataset_id,f)
            # SPLITTING VALIDATION DATA FOR EVALUATION PURPOSES
            X_train, X_test,y_train,y_test = train_test_split(val_dataset.features,
                                                                val_dataset.labels,test_size=0.6)

            if train_model:
                ## Training Loop for TABPFN
                if os.path.exists(MODEL_PATH):
                    print("loading weights")
                    model.model[2].load_state_dict(torch.load(MODEL_PATH))
                
                val_data = X_train, X_test,y_train,y_test
                train_dataloader = DataLoader(
                dataset=train_dataset,
                shuffle=True,
                batch_size=512)
                
                train(model.model[2],train_dataloader,val_data,TABPFN_HP_GRID["epochs"][0],
                    TABPFN_HP_GRID["criterion"][0],TABPFN_HP_GRID["optimizer"][0],DEVICE,writer)
                print("Average Validation Prediction Time: ", sum(valid_logs["time"])/len(X_test))
                print("Average Validation Prediction Accuracy: ", sum(valid_logs["accuracy"])/len(X_test))

            else:
                ## EVALUATION of other models
                print("Starting Evaluation...")
                for model_id, model in MODELS:
                    if "OG_TABPFN" in model_id:
                        model.model[2].load_state_dict(torch.load(MODEL_PATH))
                        model_evaluation(model,model_id,dataset_name,X_train, y_train,X_test,y_test)
                    else:
                        model_evaluation(model,model_id,dataset_name,X_train, y_train,X_test,y_test)