HDR_ROOT = "/work/u8083200/Thesis/datasets/SingleHDR_training_data"

WEIGHT_NAME = "rawhdr"
# Training
EPOCH = 61
BATCH_SIZE = 8

AUG = True

# Loss setting

loss_weight = {"rec_loss": 1.0,
               "lpips_loss": 0.5,
               "mask_loss": 0.5,
               } 
    
RESULT_SAVE_PATH = "/work/u8083200/Thesis/SOTA/rawhdr/result"
WEIGHT_SAVE_PATH = "/work/u8083200/Thesis/SOTA/rawhdr/weight"


LEARNING_RATE = 1e-4
DEVICE = "cuda"

import torch
import random
import numpy as np
def set_random_seed(seed):
    # Set the random seed for Python RNG
    random.seed(seed)
    np.random.seed(seed)

    # Set the random seed for PyTorch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False