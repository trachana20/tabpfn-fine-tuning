#!/bin/bash

# List all conda environments
envs=$(conda env list | awk '{print $1}' | tail -n +4)

# Iterate over each environment and install openml 0.14.2
for env in $envs; do
    echo "Installing openml 0.14.2 in environment: $env"
    conda activate $env
    pip install openml==0.14.2
    conda deactivate
done

echo "Installation complete in all environments."
