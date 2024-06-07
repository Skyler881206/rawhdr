hdr_root = "/work/u8083200/Thesis/datasets/SingleHDR_training_data"

WEIGHT_NAME = "rawhdr"
# Training
EPOCH = 61
BATCH_SIZE = 8

AUG = True

# Loss setting
LDR_DOMAIN = True # ev_loss in ldr domain

EV_NORMALIZE = False # Normalize ev_loss in average 0.5
EV_NORM_LDR = False # Normalize shold in ldr domain
EV = 4
# EV_focus = "matrix_norm"
EV_focus = None
P_CONV = True
TANH = True

loss_weight = {"rec_loss": 1.0,
               "lpips_loss": 0.5,
               "mask_loss": 0.5,
               }

loss_config = {"ev_loss": {"ev": EV,
                           "tanh": TANH}}

if "hdr" in WEIGHT_NAME:
    LDR_DOMAIN = False
    
if "au" in WEIGHT_NAME:
    AUG = True
    
if "ev" in WEIGHT_NAME:
    EV_NORMALIZE = True

if "ldr" in WEIGHT_NAME:
    EV_NORM_LDR = True

if "no_tanh" in WEIGHT_NAME:
    TANH = False
    
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