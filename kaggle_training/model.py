import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SiameseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Shared Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Decoder (Takes concatenated features from both branches)
        # So in channels are doubled for the first step
        self.up1 = nn.ConvTranspose2d(1024 * 2, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512 + 512 * 2, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256 + 256 * 2, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128 + 128 * 2, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64 + 64 * 2, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward_encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    def forward(self, t1, t2):
        # Pass both images through the shared encoder
        t1_1, t1_2, t1_3, t1_4, t1_5 = self.forward_encoder(t1)
        t2_1, t2_2, t2_3, t2_4, t2_5 = self.forward_encoder(t2)
        
        # Concatenate features at the bottleneck
        x = torch.cat([t1_5, t2_5], dim=1)
        
        # Decode and concatenate skip connections from both branches
        x = self.up1(x)
        x = torch.cat([x, t1_4, t2_4], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, t1_3, t2_3], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, t1_2, t2_2], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, t1_1, t2_1], dim=1)
        x = self.conv4(x)
        
        logits = self.outc(x)
        return logits
