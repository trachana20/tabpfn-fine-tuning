
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# from performer_pytorch import SelfAttention
# import peft

from linformer import LinformerSelfAttention,Linformer
from gym import plot_data_comparison
from GAN import GAN

from sklearn.neighbors import NearestNeighbors

def apply_linformer(model):
    print("Initialising and applying linformer")
    num_heads = model.model[2].transformer_encoder.layers[0].self_attn.num_heads 
    embed_dim = model.model[2].transformer_encoder.layers[0].self_attn.embed_dim
    lin_enc = Linformer(
        dim=embed_dim,
        seq_len=1000,
        depth=12,
        k=64,
        heads=num_heads,
        one_kv_head=True,
        share_kv=True,
)
    model.model[2].transformer_encoder = lin_enc 

    # attn = LinformerSelfAttention(
    #     dim = embed_dim,
    #     seq_len = 1000,
    #     heads = num_heads,
    #     k = 128,
    #     one_kv_head = True,
    #     share_kv = True
    # )
    # for i in range(len(model.model[2].transformer_encoder.layers)):
    #     model.model[2].transformer_encoder.layers[i].self_attn = attn
    
    print("New model: ")
    print(model.model[2])
    return model

def apply_rag_augmentation(train_X, 
                            train_y,
                            test_X,
                            val_X,
                            target_col_name,
                            device,
                            gan_epochs):
    print("Creating external context with GANs...")
    old_df,syn_df,_,_ = apply_gans_augmentation(train_X, 
                                                train_y,
                                                target_col_name,
                                                device,
                                                gan_epochs)
    print("Deriving similar samples from train data using GAN samples...")
    similar_samples_gans = get_similar_samples(syn_df,
                                               old_df,
                                               num_samples=10,
                                               unknown=False,
                                               target_col_name=target_col_name)
    
    print("Creating external context with Validation and Test Features only...")
    unknown_df = pd.concat((test_X,val_X),axis=0).reset_index(drop=True)
    print("Deriving similar samples from train data using ...")
    similar_samples_unknown = get_similar_samples(unknown_df,
                                                  old_df,
                                                  num_samples=10,
                                                  unknown=True,
                                                  target_col_name=target_col_name)
    print("Generate random samples from aggregated extended context...")

    similar_samples_gans['weight'] = 0.3  # Assign a lower weight
    similar_samples_unknown['weight'] = 0.7  # Assign a higher weight

    similar_samples_df = pd.concat([similar_samples_gans, similar_samples_unknown], axis=0)

    # Calculate the number of samples needed
    required_num_samples = 1000 - len(old_df)

    # Sample with weights
    augmented_samples = similar_samples_df.sample(n=required_num_samples, weights='weight')
    augmented_data = pd.concat((old_df,augmented_samples),axis=0)

    print("Plotting data")
    plot_data_comparison(augmented_data,old_df,target_col_name)
    
    train_y = augmented_data[target_col_name]
    train_X = augmented_data.drop(target_col_name,axis=1)

    return _,_,train_X, train_y

# TODO: GANS conditioning; no conditioning applied for now
def apply_gans_augmentation(train_X, 
                            train_y,
                            target_col_name,
                            device,
                            gan_epochs):
    
    df = pd.concat((train_X, train_y),axis=1)
    
    num_rows, num_cols = df.shape
    required_num_rows = 1000 - num_rows
    
    features_tensor = torch.tensor(df.values, dtype=torch.float32)
    target_tensor = torch.tensor(np.zeros(num_rows), dtype=torch.float32)

    dataset = TensorDataset(features_tensor, target_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Training GAN on given data...")
    gan = GAN(input_dim=num_cols, output_dim=num_cols, device=device)
    gan.train(data_loader,epochs=gan_epochs,batch_size=32)

    print("Generating {} new samples...".format(required_num_rows))
    noise = torch.randn((required_num_rows, gan.input_dim)).to(gan.device)
    
    synthetic_data = gan.generator(noise).cpu().detach().numpy()
    syn_df = pd.DataFrame(synthetic_data,columns=df.columns)
    syn_df[target_col_name] = abs(syn_df[target_col_name].astype(int))
    
    new_data = pd.concat((df,syn_df),axis=0).reset_index(drop=True)
    print("Ex. old samples...")
    print(new_data.head())
    print("Ex. new samples...")
    print(new_data.tail())

    print("Plotting data")
    plot_data_comparison(new_data,df,target_col_name)

    train_y = new_data[target_col_name]
    train_X = new_data.drop(target_col_name,axis=1)

    return df,syn_df,train_X, train_y

def evolute_neighbours(neighbours,method="mean"):
    if method == "mean":
        return np.mean(neighbours, axis=0)
    else:
        raise ValueError(f"Unsupported method: {method}")

def get_similar_samples(unk_df,old_df,num_samples,unknown=True,target_col_name=""):
    if unknown:
        old_target_array = old_df.to_numpy()
        old_array = old_df.drop(target_col_name,axis=1).to_numpy()
    else:
        old_array = old_df.to_numpy()
    new_array = unk_df.to_numpy()

    nbrs = NearestNeighbors(n_neighbors=num_samples, metric='cosine').fit(old_array)
    
    new_samples = []
    
    for row in new_array:
        _, indices = nbrs.kneighbors([row])
        
        if unknown:
            neighbours = old_target_array[indices[0]]
        else:
            neighbours = old_array[indices[0]]
        
        new_sample = evolute_neighbours(neighbours)
        
        new_samples.append(new_sample)
    
    new_samples_df = pd.DataFrame(new_samples, columns=old_df.columns)
    
    return new_samples_df


# def apply_lora(model):
#     print("Initialising and applying LoRA")
    
#     linear_layers = []
#     for name, submodule in model.model[2].named_modules():
#         if isinstance(submodule, nn.Linear):
#             linear_layers.append(name)

#     config = peft.LoraConfig(
#         r=8,
#         target_modules=linear_layers
#     )

#     peft_model = peft.get_peft_model(model.model[2], config)
#     peft_model.print_trainable_parameters()

#     model.model = (None, None, peft_model) 
#     print("New model: ")
#     print(model.model[2])
#     return model

# def apply_performer(model):
#     print("Initialising and applying Performer")
#     num_heads = model.model[2].transformer_encoder.layers[0].self_attn.num_heads 
#     embed_dim = model.model[2].transformer_encoder.layers[0].self_attn.embed_dim
#     attn = SelfAttention(
#         dim = embed_dim,
#         heads = num_heads,
#         causal = False,
#         )
#     for i in range(len(model.model[2].transformer_encoder.layers)):
#         model.model[2].transformer_encoder.layers[i].self_attn = attn
#     print("New model: ")
#     print(model.model[2])
#     return model