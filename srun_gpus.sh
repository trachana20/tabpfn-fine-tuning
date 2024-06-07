#!/bin/bash

# Prompt the user to select the partition
echo "Select the partition you want to work on:"
echo "1 = hiwi"
echo "2 = test"
read -p "Enter the number (1 or 2): " PARTITION_CHOICE

# Validate the partition choice
if [[ "$PARTITION_CHOICE" == "1" ]]; then
  PARTITION="mlhiwidlc_gpu-rtx2080"
elif [[ "$PARTITION_CHOICE" == "2" ]]; then
  PARTITION="testdlc_gpu-rtx2080"
else
  echo "Error: Please enter a valid partition choice (1 or 2)."
  exit 1
fi

# Prompt the user to enter the number of GPUs
read -p "Enter the number of GPUs you want to use: " NUM_GPUS

# Validate if the input is a number
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
  echo "Error: Please enter a valid number for GPUs."
  exit 1
fi

# Prompt the user to enter the number of CPUs
read -p "Enter the number of CPUs you want to use: " NUM_CPUS

# Validate if the input is a number
if ! [[ "$NUM_CPUS" =~ ^[0-9]+$ ]]; then
  echo "Error: Please enter a valid number for CPUs."
  exit 1
fi

# Run the srun command with the specified partition, number of GPUs, and number of CPUs
srun --partition $PARTITION --gres=gpu:$NUM_GPUS --cpus-per-task=$NUM_CPUS --pty bash
