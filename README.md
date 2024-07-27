# Fine-Tuning TabPFN per dataset with data augmentation techniques

A project to fine-tune a generically pretrained TabPFN (see: [TabPFN GitHub Repository](https://github.com/automl/TabPFN)) for specific datasets.

## Introduction

TabPFN demonstrates robust and rapid classification performance across diverse datasets. To enhance performance on specific datasets, we fine-tune a pretrained TabPFN model to achieve optimal results for particular tasks.

## Methodology

### 1. Dataset Preparation

**Datasets Used:**

We used a total of 8 OpenML datasets:

- **168746:** Titanic
- **9982:** Dress-Sales
- **15:** Breast-W
- **37:** Diabetes
- **3783:** FRI_C2_500_50
- **3562:** Lupus
- **3778:** Plasma Retinol
- **3748:** Transplant

The datasets vary in size, ranging from 100 to 900 instances.

**Synthetic Data Generation:**

**Generative Adversarial Networks (GANs):**

- **Purpose:** Generate synthetic data to ensure a minimum of 1000 instances per dataset.
- **Architecture:** Conditional GANs (CGANs) to preserve categorical feature integrity.
- **Training Configuration:**
  - Epochs: 3000, 1000, 500, and 100
  - Batch Size: 128
  - Input Dimension: 100

### 2. Fine-Tuning Techniques

**Full Fine-Tuning:**

*ToDo:* Write the details of the full fine-tuning process.

**Low-Rank Adaptation (LoRA):**

*ToDo:* Provide a detailed description of LoRA implementation.

- **Overview:**
  - Adapts the weights of the pre-trained TabPFN.
  - Injects low-rank matrices into the model to efficiently fine-tune it.

### 3. Data Augmentation with RAG

**Cosine Similarity-Based Augmentation:**

- Generated synthetic data using GANs.
- Calculated cosine similarity between original and synthetic data.
- Augmented the training data with the mean of the top 5 most similar synthetic rows.

### 4. Training Procedures

*ToDo:* Detail the experiment setup.

### Ablation Study

**GANs:**

- Tested different epochs: [1, 100, 500, 1000, 3000, 5000]
- Tested different batch sizes: [128, 64, 32]

**Full Fine-Tuning:**

- Tested different epochs: [1, 100, 500, 1000]


## Data
The data can be either a locally loadable dataset, which you specify the target column explicitly, or an OpenML dataset, specified by the id. 
## Results 
TODO: Write all the results, or create a folder with plots and tables and refer it here!

# Getting Started & Installation  
To get started with this project, follow these steps:

## Setup Repository & Environment
1. **Clone the Repository:**
   ```bash
   https://github.com/trachana20/tabpfn-fine-tuning.git
   cd tabpfn-fine-tuning
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
   cd tabpfn-fine-tuning
   conda activate finetune 
2. **Run main.py:**
    ```bash
   python main.py
## Reproduce Evaluation 

1. **Requirements**

    To reproduce the evaluation you either have to run the fine tuning run to collect all the important metrics & data. The ``evaluationNotebook.ipynb`` is the entry point to reproduce the evaluation plots. 

 
