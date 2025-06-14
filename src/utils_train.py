from transformers import set_seed
import os
import torch

def set_seeds(seed):
    """
        Enable deterministic behavior.
        https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L58
    """
    set_seed(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False