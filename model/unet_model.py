""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))
        

    def forward(self, x):
        x1 = self.inc(x)
        print(f'x1.shape = {x1.size()}')
        x2 = self.down1(x1)
        print(f'x2.shape = {x2.size()}')
        x3 = self.down2(x2)
        print(f'x3.shape = {x3.size()}')
        x4 = self.down3(x3)
        print(f'x4.shape = {x4.size()}')
        x5 = self.down4(x4)
        print(f'x5.shape = {x5.size()}')
        x = self.up1(x5, x4)
        print(f'up1_x.shape = {x.size()}')
        x = self.up2(x, x3)
        print(f'up2_x.shape = {x.size()}')
        x = self.up3(x, x2)
        print(f'up3_x.shape = {x.size()}')
        x = self.up4(x, x1)
        print(f'up4_x.shape = {x.size()}')
        logits = self.outc(x)
        return logits


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)