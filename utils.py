import torch


def set_seed_globally(random_state):
    torch.manual_seed(random_state)  # Set seed for torch RNG
    torch.cuda.manual_seed(random_state)  # Set seed for CUDA RNG
    torch.cuda.manual_seed_all(random_state)  # Set seed for all CUDA devices
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cudnn
    # Disable cudnn benchmark for reproducibility
    torch.backends.cudnn.benchmark = False
