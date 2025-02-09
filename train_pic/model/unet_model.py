""" Full assembly of the parts to form the complete network """

from .unet_parts import *


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
        # print(f'x1.shape = {x1.size()}')
        x2 = self.down1(x1)
        # print(f'x2.shape = {x2.size()}')
        x3 = self.down2(x2)
        # print(f'x3.shape = {x3.size()}')
        x4 = self.down3(x3)
        # print(f'x4.shape = {x4.size()}')
        x5 = self.down4(x4)
        # print(f'x5.shape = {x5.size()}')
        x = self.up1(x5, x4)
        # print(f'up1_x.shape = {x.size()}')
        x = self.up2(x, x3)
        # print(f'up2_x.shape = {x.size()}')
        x = self.up3(x, x2)
        # print(f'up3_x.shape = {x.size()}')
        x = self.up4(x, x1)
        # print(f'up4_x.shape = {x.size()}')
        logits = self.outc(x)
        return logits


    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up4 = torch.utils.checkpoint(self.up4)
    #     self.outc = torch.utils.checkpoint(self.outc)

    def train_one_step(self, x, y, mask, minmax, criterion, k, metric_names={}):
        '''
        x:      [bs, var, lat, lon]
        y:      [bs, depth, lat, lon]
        mask:   [1, lat, lon]
        pred:   [bs, var, lat, lon]

        return: loss
        '''
        # process data
        info = {}
        bs, depth, lat, lon = y.shape

        mask_x = mask.unsqueeze(1).repeat(1 ,x.shape[1], 1, 1)  # (1, lat, lon) --> (bs, var, lat, lon)
        mask_y = mask.unsqueeze(1).repeat(1 , depth, 1, 1).reshape(bs, depth, -1)  # (1, lat, lon) --> (bs, depth, lat, lon)

        # Apply mask to x 
        x = torch.where(mask_x, x, 0.0)

        # pred
        pred = self(x).reshape(bs, depth, -1)   # pred shape: (bs, var, lat, lon)
        y = y.reshape(bs, depth, -1) 
        
        if metric_names:
            mask_y = mask_y.unsqueeze(1)

            # Count of valid elements for each batch
            valid_count = mask_y.sum(dim=-1)  # [bs, 1, depth]

            # Mask the predicted and true values
            masked_pred = torch.where(mask_y, pred.unsqueeze(1), 0.0)  # [bs, 1, depth, -1]
            masked_y = torch.where(mask_y, y.unsqueeze(1), 0.0)

            # Compute loss
            loss = criterion(masked_pred, masked_y, metric_names, valid_count)

            pred = torch.where(mask_y.reshape(bs, -1), pred.reshape(bs, -1), torch.tensor(float('nan')))
            pred = pred.reshape(bs, depth, lat, lon)
        else:
            # 掩码处理：删去了mask为false的值，得到一个一维数组
            masked_pred = torch.masked_select(pred, mask_y)
            masked_y = torch.masked_select(y, mask_y)
 
            loss = criterion(masked_pred, masked_y)

        return loss, pred, info

        
# x1.shape = torch.Size([1, 32, 108, 200])
# x2.shape = torch.Size([1, 64, 54, 100])
# x3.shape = torch.Size([1, 128, 27, 50])
# x4.shape = torch.Size([1, 256, 14, 25])
# x5.shape = torch.Size([1, 512, 7, 13])
# up1_x.shape = torch.Size([1, 256, 14, 25])
# up2_x.shape = torch.Size([1, 128, 27, 50])
# up3_x.shape = torch.Size([1, 64, 54, 100])
# up4_x.shape = torch.Size([1, 32, 108, 200])