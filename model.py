import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_generator=True):
        super(ConvBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel) if is_generator else nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, img):
        return self.conv(img)

class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConv, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, img1, img2):
        img = torch.cat([img1, img2], dim=1)
        return self.upconv(img)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # input image shape (3, 512, 512)
        self.conv1 = ConvBlock(3, 64) # 256 256
        self.conv2 = ConvBlock(64, 128) # 128, 128
        self.conv3 = ConvBlock(128, 256) # 64, 64
        self.conv4 = ConvBlock(256, 512) # 32, 32
        self.bottleneck = ConvBlock(512, 512) # 32, 32
        
        self.up1 = UpConv(1024, 256) # 64, 64
        self.up2 = UpConv(512, 128) # 128, 128
        self.up3 = UpConv(256, 64) # 256, 256
        self.bottleneck2 = ConvBlock(128, 3)
        self.mp = nn.MaxPool2d(2) 
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, input):
        d_1 = self.conv1(input)
        d_1_1 = self.mp(d_1)
        d_2 = self.conv2(d_1_1)
        d_2_1 = self.mp(d_2)
        d_3 = self.conv3(d_2_1)
        d_3_1 = self.mp(d_3)
        d_4 = self.conv4(d_3_1)
        d_4_1 = self.mp(d_4)
        d_4_2 = self.bottleneck(d_4_1)

        up1 = self.upsample(d_4_2)
        up2 = self.up1(up1, d_4)
        up3 = self.up2(up2, d_3)
        up4 = self.up3(up3, d_2)
        to_out = torch.cat([up4, d_1], dim=1)
        out = self.bottleneck2(to_out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down = nn.Sequential(
            ConvBlock(6, 64, False),
            nn.MaxPool2d(2), # 128
            ConvBlock(64, 128, False),
            nn.MaxPool2d(2), #64
            ConvBlock(128, 256, False),
            nn.MaxPool2d(2), # 32
            ConvBlock(256, 128, False),
            nn.MaxPool2d(2),
            ConvBlock(128, 1, False),
        )


    def forward(self, img_real, img_fake):
        img = torch.cat([img_real, img_fake], dim=1)
        return self.down(img)

if __name__ == "__main__":
    from torchsummary import summary
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1, 3"
    model = Discriminator()
    summary(model.to('cuda'), (6, 256, 256))