import torch
import torch.nn as nn

class Unet_encoder(nn.Module):
    def __init__(self, in_channels, encoder_channels):
        super(Unet_encoder, self).__init__()
        
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, encoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(),
            # nn.Conv2d(encoder_channels, encoder_channels, kernel_size=3, padding=1),
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=3, padding=1)
            )
        
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(encoder_channels, encoder_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels * 2),
            nn.ReLU(),
            # nn.Conv2d(encoder_channels * 2, encoder_channels * 2, kernel_size=3, padding=1),
            nn.Conv2d(encoder_channels * 2, encoder_channels * 2, kernel_size=3, padding=1)
            )
        
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(encoder_channels * 2, encoder_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels * 4),
            nn.ReLU(),
            # nn.Conv2d(encoder_channels * 4, encoder_channels * 4, kernel_size=3, padding=1),
            nn.Conv2d(encoder_channels * 4, encoder_channels * 4, kernel_size=3, padding=1)
            )
        
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(encoder_channels * 4, encoder_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels * 8),
            nn.ReLU(),
            # nn.Conv2d(encoder_channels * 8, encoder_channels * 8, kernel_size=3, padding=1),
            nn.Conv2d(encoder_channels * 8, encoder_channels * 8, kernel_size=3, padding=1)
            )
        
        self.downsample = nn.MaxPool2d(2)
    
    def forward(self, x):
        feature_1 = self.encoder_1(x) # 1/1
        feature_2 = self.encoder_2(self.downsample(feature_1)) # 1/2
        feature_3 = self.encoder_3(self.downsample(feature_2)) # 1/4
        feature_4 = self.encoder_4(self.downsample(feature_3)) # 1/8
        
        return (feature_1, feature_2, feature_3, feature_4)
    

class Unet_decoder(nn.Module):
    def __init__(self, out_channels, decoder_channels):
        super(Unet_decoder, self).__init__()
        
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(decoder_channels * 8, decoder_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels * 8),
            nn.ReLU(),
            # nn.Conv2d(decoder_channels * 8, decoder_channels * 8, kernel_size=3, padding=1),
            nn.Conv2d(decoder_channels * 8, decoder_channels * 8, kernel_size=3, padding=1)
            )
        
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(int(decoder_channels * 4 / 2) + decoder_channels * 4, decoder_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels * 4),
            nn.ReLU(),
            # nn.Conv2d(decoder_channels * 4, decoder_channels * 4, kernel_size=3, padding=1),
            nn.Conv2d(decoder_channels * 4, decoder_channels * 4, kernel_size=3, padding=1)
            )
        
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(int(decoder_channels * 2 / 2) + decoder_channels * 2, decoder_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels * 2),
            nn.ReLU(),
            # nn.Conv2d(decoder_channels * 2, decoder_channels * 2, kernel_size=3, padding=1),
            nn.Conv2d(decoder_channels * 2, decoder_channels * 2 , kernel_size=3, padding=1)
            )
        
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(int(decoder_channels / 2) + decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(),
            # nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=3, padding=1)
            )
        
        self.pixelshuffle = nn.PixelShuffle(2)
        
    def forward(self, x):
        feature = self.decoder_4(x[-1]) # 1/8
        feature = self.pixelshuffle(feature) # Upsample
        feature = self.decoder_3(torch.cat([feature, x[-2]], dim=1)) # 1/4
        feature = self.pixelshuffle(feature) # Upsample
        feature = self.decoder_2(torch.cat([feature, x[-3]], dim=1)) # 1/2
        feature = self.pixelshuffle(feature) # Upsample
        feature = self.decoder_1(torch.cat([feature, x[-4]], dim=1)) # 1/1 
        return feature

class dual_intensity_guidance(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(dual_intensity_guidance, self).__init__()
        self.ue_rb = Unet_encoder(int(in_channels / 2), 32)
        self.ue_local = Unet_encoder(in_channels, 32)
        self.ue_gg = Unet_encoder(int(in_channels / 2), 32)
        
        self.ud_rb = Unet_decoder(out_channels, 32 + 32)
        self.ud_gg = Unet_decoder(out_channels, 32 + 32)
        
    def forward(self, x):
        y_rb = self.ue_rb(x[:, 0:2, :, :])
        y_local = self.ue_local(x)
        y_gg = self.ue_gg(x[:, 2:4, :, :])
        
        y_rb_ = self.ud_rb(self.concat_feature(y_rb, y_local))
        y_gg_ = self.ud_gg(self.concat_feature(y_gg, y_local))
        
        return y_rb_, y_gg_

    @staticmethod
    def concat_feature(feature_1, feature_2):
        feature = []
        for i in range(len(feature_1)):
            feature.append(torch.cat([feature_1[i], feature_2[i]], dim=1))
        return feature

class bgr2rbgg(nn.Module):
    def __init__(self):
        super(bgr2rbgg, self).__init__()
        
    def forward(self, x):
        I_b = x[:, 0:1, :, :]
        I_g = x[:, 1:2, :, :]
        I_r = x[:, 2:3, :, :]
        
        return torch.cat([I_r, I_b, I_g, I_g], dim=1)

if __name__ == "__main__":
    # encode_model = Unet_encoder(3, 64)
    # decode_model = Unet_decoder(3, 64)
    model = dual_intensity_guidance(4, 2).to("cuda")
    transfer_model = bgr2rbgg().to("cuda")
    
    test_tensor = torch.randn(2, 3, 512, 512).to("cuda")
    test_tensor = transfer_model(test_tensor)
    # feature = encode_model(test_tensor)
    # output = decode_model(feature)
    feature = model(test_tensor)