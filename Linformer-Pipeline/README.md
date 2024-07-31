# Finetunig-TabPFN

DL Lab project 2024

Brief description of your project.

## To-Do List

- [x] Complete Evaluation function for general sklearn models
- [x] RAG approximation implementation
- [x] Implement argparse
- [x] Generate distribution plots for GANs data
- [x] Generate Ablation results
- [ ] Test GPU runs
- [x] Update readme

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

Make sure you have [Anaconda](https://www.anaconda.com/products/distribution#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/ds-brx/Finetunig-TabPFN.git
   cd your-repo-name
   ```

2. **Create a new Conda environment with Python 3.10**

   ```bash
   conda create --name myenv python=3.10
   ```

3. **Activate the environment**

   ```bash
   conda activate myenv
   ```

4. **Install the required packages**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Example : Run with default settings:

Here is an example of how to run the project with custom arguments:

```bash
python main.py
```

### Example : Run with custom arguments:

The project uses `argparse` to handle command-line arguments for configuration. Below are the available arguments and their descriptions:

```bash
python main.py --aug_flag "gan" --model_config_flag "performer" --gan_epochs 100 --num_folds 3 --device "cuda" --num_classes 4 --train_id "experiment_1" --epochs 20 --lr 1e-4
```

- `--aug_flag`: Augmentation flag

  - Type: `str`
  - Choices: `["gan", "rag"]`
  - Default: `""`
  - Description: Augmentation flag

- `--model_config_flag`: Model configuration flag

  - Type: `str`
  - Choices: `["linformer"]`
  - Default: `"linformer"`
  - Description: Model configuration flag

- `--gan_epochs`: Number of GAN training epochs

  - Type: `int`
  - Default: `50`

- `--num_folds`: Number of folds for cross-validation

  - Type: `int`
  - Default: `5`

- `--device`: Device to use for training

  - Type: `str`
  - Default: `"cuda"` if available, otherwise `"cpu"`

- `--num_classes`: Number of classes

  - Type: `int`
  - Default: `2`

- `--train_id`: Training ID based on applied flags

  - Type: `str`
  - Default: `"model"`

- `--epochs`: Number of epochs

  - Type: `int`
  - Default: `10`

- `--lr`: Learning rate
  - Type: `float`
  - Default: `1e-6`
