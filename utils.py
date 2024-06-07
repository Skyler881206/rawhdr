import sys
sys.path.append("/work/u8083200/Thesis/SOTA/rawhdr/utils")

import torch
import torch.nn as nn
import lpips

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()
p_fn_loss = lpips.LPIPS(net='vgg').to(device="cuda")

def loss(predict, target, loss_weight):
    def tone_mapping(hdr_image, mu = 5000):
        return torch.log(torch.add(mu * hdr_image, 1)) / torch.log(torch.tensor(1 + mu))
    
    def rec_loss(pd_hdr, target_hdr):
        # pd_hdr = torch.clamp(pd_hdr, min=0)
        # target_hdr = torch.clamp(target_hdr, min=0)
        return l2_loss(torch.log(pd_hdr + 1e-6), torch.log(target_hdr + 1e-6))
        # return l2_loss(tone_mapping(pd_hdr), tone_mapping(target_hdr))
    
    def lpips_loss(pd_hdr, target_hdr):
        return torch.mean(p_fn_loss(pd_hdr, target_hdr))
    
    def mask_loss(mask, target):
        return l1_loss(mask["over"], target["over"]) + l1_loss(mask["under"], target["under"])
    
    loss_dict = {}
    for key, val in loss_weight.items():
        if key == "rec_loss":
            loss_dict[key] = rec_loss(predict["output"], target["output"])
        elif key == "lpips_loss":
            loss_dict[key] = lpips_loss(predict["output"], target["output"])
        elif key == "mask_loss":
            loss_dict[key] = mask_loss(predict["mask"], target["mask"])
        else:
            raise ValueError("Invalid loss key: {}".format(key))
        
    for idx, (key, val) in enumerate(loss_dict.items()):
        if idx == 0:
            loss = val * loss_weight[key]
            continue
        loss += val * loss_weight[key]
    loss_dict["loss"] = loss
    return loss, loss_dict