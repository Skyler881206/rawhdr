import sys
sys.path.append("/work/u8083200/Thesis/SOTA/rawhdr/module")

from arch import dual_intensity_guidance, bgr2rbgg
from block import mask_estimation, global_spatial_guidance
from Reconstruct import Reconstruct
import torch
import torch.nn as nn

class rawhdr_model(nn.Module):
    def __init__(self):
        super(rawhdr_model, self).__init__()
        
        self.bgr2rbgg = bgr2rbgg()
        
        self.mask_estimation_over = nn.Sequential(
            mask_estimation(4, 1),
            # mask_estimation(32, 1),
            nn.Sigmoid()
            )
        
        self.mask_estimation_under = nn.Sequential(
            mask_estimation(4, 1),
            # mask_estimation(32, 1),
            nn.Sigmoid()
            )
        
        self.dual_intensity_guidance = dual_intensity_guidance(in_channels=4, out_channels=3)

        self.global_spatial_guidance = global_spatial_guidance()
        
        self.Reconstruct = nn.Sequential(
            Reconstruct(3, 32),
            Reconstruct(32, 3)
            )
      
    def forward(self, x):
        x = self.bgr2rbgg(x)
        mask_over = self.mask_estimation_over(x)
        mask_under = self.mask_estimation_under(x)
        y_rb_, y_gg_ = self.dual_intensity_guidance(x)
        Y_di = self.mask_sum(y_rb_, y_gg_, mask_over, mask_under)
        Y_sg = self.global_spatial_guidance(x)
        
        return torch.relu(self.Reconstruct(Y_sg + Y_di)), {"over": mask_over, "under": mask_under}
    
    
    @staticmethod
    def mask_sum(y_rb, y_gg, mask_over, mask_under):
        return y_gg * mask_under + y_rb * mask_over

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
    
if __name__ == "__main__":
    import torch
    model = rawhdr_model().to("cuda")
    x = torch.randn(2, 3, 512, 512).to("cuda")
    y = model(x)
    print(y.shape)