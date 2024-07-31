import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.DataManager import DataManager


def forward_new(model,x,y,single_eval_pos,num_cls):
    '''
    custom forward epoch for linformer based on https://github.com/automl/TabPFN
    '''
    x_src = model.encoder(x)
    y_src = model.y_encoder(y)
    train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
    src= torch.cat([train_x, x_src[single_eval_pos:]], 0)
    output = model.transformer_encoder(src)
    output = model.decoder(output)
    output = output[single_eval_pos:]
    y_preds = output.reshape(-1, 10)[:, :2]

    return y_preds


def prepare_data(x,y,single_eval_pos):
    '''
    preparing data for forward pass
    '''
    x = x.unsqueeze(0).transpose(0, 1)
    y = y.unsqueeze(0).transpose(0, 1)
    y_train = y[:single_eval_pos]
    y_query = y[single_eval_pos:]
    x_data = nn.functional.pad(x, (0, 100 - x.shape[-1])).float()
    y_data = y_train.unsqueeze(1).float()
    return x_data,y_data,y_query


def train(
        model,
        train_dataloader,
        valid_dataset,
        num_cls,
        epochs,
        criterion,
        optimizer,
        lr,
        device,
        writer,
        flag,
        train_logs,
        valid_logs,
        model_path,
):
    optimizer = optimizer(
            params=model.parameters(),
            lr=lr,
        )
    
    '''
    Train model for finetuning tabPFN with Linformer
    '''
    for e in range(epochs):
        model.train()
        model.to(device)
        batch_loss = []
        batch_accuracy = []
        for b,(x,y) in enumerate(train_dataloader):
            
            optimizer.zero_grad()
            
            single_eval_pos = int(x.shape[0] * 0.8)
            
            x_data,y_data,y_query = prepare_data(x,y,single_eval_pos)
            
            x_data = x_data.to(device)
            x_data = x_data.to(device)
            
            y_preds = forward_new(model,x_data,y_data,single_eval_pos,num_cls)
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
        
        validation(model,valid_dataset,num_cls,writer,valid_logs,epoch=e)
    
    save_model(model,model_path)


def validation(model, valid_dataset,num_cls,writer=None,valid_logs=None,epoch=0,log=True,single_eval_pos=0):
    '''
    Validate model for finetuning tabPFN with Linformer
    '''
    
    with torch.no_grad():
        val_X, val_y = valid_dataset
        single_eval_pos = int(val_X.shape[0] * 0.8)
        val_X, val_y = torch.tensor(val_X.values,dtype=torch.float32), torch.tensor(val_y.values,dtype=torch.float32)
        x_data,y_data,y_query = prepare_data(val_X,val_y,single_eval_pos)

        model.eval()
        start_time = time.time()
        y_preds = forward_new(model,x_data,y_data,single_eval_pos,num_cls)
        total_time = time.time()-start_time
        accuracy = accuracy_score(y_query.cpu().detach().numpy(), np.argmax(y_preds.cpu().detach().numpy(),axis=1))
        print("Evaluation Accuracy: {}, Evaluation Time: {}".format(accuracy,total_time))
        if log:
            valid_logs["epochs"].append(epoch)
            valid_logs["accuracy"].append(accuracy)
            valid_logs["time"].append(total_time)

            
            writer.add_scalar('Validation/Accuracy', accuracy, epoch)
            writer.add_scalar('Validation/Time', total_time, epoch)




def save_model(model,path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def numpy_to_torch(*arrays):
    return [torch.from_numpy(arr).float() if arr.dtype in [np.float32, np.float64]
             else torch.from_numpy(arr) for arr in arrays]

def df_wrapper(dataset):
    full_df = pd.DataFrame(dataset["data"])
    y = full_df[dataset["target"]]
    X = full_df.drop(dataset["target"],axis=1)
    return X,y

def plot_data_comparison(new_data, old_data,dirname):
    output_dir = os.path.join("plots2", dirname)
    os.makedirs(output_dir, exist_ok=True)

    for column in old_data.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(new_data[column], color='blue', kde=True, label='New Data', stat="density", bins=30)
        sns.histplot(old_data[column], color='orange', kde=True, label='Old Data', stat="density", bins=30)
        plt.title(f'Distribution Comparison of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend()
        plt_path = os.path.join(output_dir, f'{column}_histogram.png')
        plt.savefig(plt_path)
        plt.close()

def test_dataset(dataset_id):
    '''
    Get whole test dataset and train dataset for evaluation of all models
    '''
    data_manager = DataManager(
    dir_path="data/dataset",
    dataset_id=dataset_id
    )
    data_k_folded = data_manager.k_fold_train_test_split(
            k_folds=1,
            val_size=0.2,
            random_state=1,
        )
    fold = data_k_folded[0]
    train_data = fold["train"]
    test_data = fold["test"]

    train_df = pd.DataFrame(train_data["data"])
    target = train_data["target"]

    y_train = train_df[target]
    X_train = train_df.drop(target,axis=1)

    test_df = pd.DataFrame(test_data["data"])
    y_test = test_df[target]
    X_test = test_df.drop(target,axis=1)

    return X_test,y_test, X_train, y_train



