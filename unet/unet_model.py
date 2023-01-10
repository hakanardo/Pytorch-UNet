""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        a = torch.cuda.memory_allocated()
        x1 = self.inc(x)
        b = torch.cuda.memory_allocated()
        print((b-a)/1024/1024, 'MB')
        a = b
        x2 = self.down1(x1)
        b = torch.cuda.memory_allocated()
        print((b-a)/1024/1024, 'MB')
        a = b
        x3 = self.down2(x2)
        b = torch.cuda.memory_allocated()
        print((b-a)/1024/1024, 'MB')
        a = b
        x4 = self.down3(x3)
        b = torch.cuda.memory_allocated()
        print((b-a)/1024/1024, 'MB')
        a = b
        x5 = self.down4(x4)
        b = torch.cuda.memory_allocated()
        print((b-a)/1024/1024, 'MB')
        a = b
        x = self.up1(x5, x4)
        b = torch.cuda.memory_allocated()
        print((b-a)/1024/1024, 'MB')
        a = b
        x = self.up2(x, x3)
        b = torch.cuda.memory_allocated()
        print((b-a)/1024/1024, 'MB')
        a = b
        x = self.up3(x, x2)
        b = torch.cuda.memory_allocated()
        print((b-a)/1024/1024, 'MB')
        a = b
        x = self.up4(x, x1)
        b = torch.cuda.memory_allocated()
        print((b-a)/1024/1024, 'MB')
        a = b
        logits = self.outc(x)
        b = torch.cuda.memory_allocated()
        print((b-a)/1024/1024, 'MB')
        a = b
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


class UNet4k(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        self.down4 = (Down(128, 256))
        self.down5 = (Down(256, 512))
        self.down6 = (Down(512, 512))
        self.down7 = (Down(512, 512))
        self.up1 = (UpAlt(512, 512))
        self.up2 = (UpAlt(512, 512))
        self.up3 = (UpAlt(512, 256))
        self.up4 = (UpAlt(256, 128))
        self.up5 = (UpAlt(128, 64))
        self.up6 = (UpAlt(64, 32))
        self.up7 = (UpAlt(32, 16))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        logits = self.outc(x)
        return logits
