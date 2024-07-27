# Fine-Tuning TabPFN 
A project to fine tune a generically pretrained TabPFN (see: https://github.com/automl/TabPFN) for a specific dataset. 
## Intro
TabPFN demonstrates robust, rapid classification performance on diverse datasets. To improve performance on a particular dataset, we fine-tune a pretrained TabPFN model for optimal results on that specific task.
## Methodology 
The Methodology involves utilizing dataset-specific prior knowledge to its fullest extent. This includes training on either heavily augmented real-world data or synthetically sampled priors that closely reflect the underlying structure of the data.
## Data
The data can be either a locally loadable dataset, which you specify the target column explicitly, or an OpenML dataset, specified by the id. 
## Results 
TODO 
# Getting Started & Installation  
To get started with this project, follow these steps:

## Setup Repository & Environment
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/PufferFishCode/FineTune-TabPFN.git
   cd FineTune-TabPFN
2. **Create Conda Environment:**
    ```bash
   conda env create -f environment.yml
3. **Activate the Environment:**
    ```bash
   conda activate finetune
4. **Create Conda Environment:**
    ```bash
   conda env create -f environment.yml
## Run Fine-Tuning TabPFN

1. **Make sure to be in the root directory & activate conda environment**
    ```bash
   cd FineTune-TabPFN
   conda activate finetune 
2. **Run final_evaluation.py:**
   This file contains a script to evaluate different machine learning models with various configurations. The script uses keyword arguments (kwargs) for customization and can be run directly from the command line.
    ```bash
   python final_evaluation.py

3. **Add Arguments:**
   --apply_lora: Set to True to apply LoRA (default: False)\
   --apply_performer: Set to True to apply Performer (default: False)\
   --apply_gans: Set to True to apply GANs with cosine similarity (default: False)\
   --apply_cosine_similarity_with_test_set: Set to True to apply cosine similarity with test set (default: False)\
   --train_model: Set to True to train the model (default: True)\
   --train_id: Model ID to train the model (default: model_performer_lora)\
   --model_path: Path to the model if you want to load it (default: finetune_model/model_performer_lora.pth)\
   --dataset_ids: Dictionary of dataset IDs to train the model with (default: {168746: "Titanic", 9982: "Dress-Sales"})\
   --device: Device to run the model (default: cuda)\
   --k_folds: Number of k-folds (default: 5)\
   --num_classes: Number of classes (default: 2)\
   --n_estimators: List of number of estimators (default: [100, 500, 1000])\
   --max_depth: List of max depth values (default: [10, 50, 100])\
   --models: Dictionary of models to evaluate (default: {"OG_TABPFN": TabPFNClassifier(batch_size_inference=5),\ "RF": RandomForestClassifier(), "DT": DecisionTreeClassifier()})\
   --epochs: List of number of epochs for TABPFN HP Grid (default: [10, 100, 1000])\
   --learning_rate: List of learning rates for TABPFN HP Grid (default: [1e-6])\
   --early_stopping: List of early stopping values for TABPFN HP Grid (default: [0.1])\
   --criterion: List of criterion functions for TABPFN HP Grid (default: [CrossEntropyLoss()])\
   --optimizer: List of optimizer functions for TABPFN HP Grid (default: [Adam])\

    ```bash
   python final_eval_script.py --apply_lora=True --apply_performer=True --device=cpu --train_id=my_custom_model

## Reproduce Evaluation 

1. **Requirements**

    To reproduce the evaluation you either have to run the fine tuning run to collect all the important metrics & data. The ``evaluationNotebook.ipynb`` is the entry point to reproduce the evaluation plots. 

 
