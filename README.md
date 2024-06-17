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
   git clone https://github.com/automl-private/fine-tuning-tabpfn.git
   cd fine-tuning-tabpfn
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
2. **Run main.py:**
    ```bash
   python main.py
## Reproduce Evaluation 

1. **Requirements**

    To reproduce the evaluation you either have to run the fine tuning run to collect all the important metrics & data. The ``evaluationNotebook.ipynb`` is the entry point to reproduce the evaluation plots. 

 
