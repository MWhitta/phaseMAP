##UNET variant with no linear output layer
# batch normalization in the double convolution steps
import torch
import torch.nn as nn

class double_conv(nn.Module):
    def __init__(self, in_c, out_c, padding=3, kernel_size=6):
        super().__init__()
        self.double_conv = nn.Sequential(
        nn.Conv1d(in_c, out_c, padding=padding, kernel_size=kernel_size, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_c, out_c, padding=padding, kernel_size=kernel_size, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.double_conv(x)


def pad_1d(small, big):
    diff = big.size()[2] - small.size()[2] #sample tensors of [n,chan,feature]
    pad_tup = (diff // 2, diff - diff // 2)
    return nn.functional.pad(small, pad_tup) #padding given left,right


class LIBSUNet(nn.Module):
    """ UNet for single channel LIBS spectra data """
    def __init__(self, max_z, l_spec):
        super(LIBSUNet, self).__init__()
        #max_z defines #elements, we add artifact and noise channels
        self.channels = max_z + 2
        #l_spec defines the length of spectral input array
        self.l_spec = l_spec
        chan = self.channels
          
        self.down_conv_1 = double_conv(1, chan)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.down_conv_2 = double_conv(chan, 2*chan)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.down_conv_3 = double_conv(2*chan, 4*chan)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.down_conv_4 = double_conv(4*chan, 8*chan)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.down_conv_5 = double_conv(8*chan, 16*chan)

        self.up_trans_1 = nn.ConvTranspose1d(
            in_channels=16*chan,
            out_channels=8*chan,
            kernel_size=2,
            stride=2)
        self.up_conv_1 = double_conv(16*chan, 8*chan, padding=3)

        self.up_trans_2 = nn.ConvTranspose1d(
            in_channels=8*chan,
            out_channels=4*chan,
            kernel_size=2,
            stride=2)
        self.up_conv_2 = double_conv(8*chan, 4*chan)

        self.up_trans_3 = nn.ConvTranspose1d(
            in_channels=4*chan,
            out_channels=2*chan,
            kernel_size=2,
            stride=2)

        self.up_conv_3 = double_conv(4*chan, 2*chan, padding=3)
        self.up_trans_4 = nn.ConvTranspose1d(
            in_channels=2*chan,
            out_channels=chan,
            kernel_size=2,
            stride=2)
        self.up_conv_4 = double_conv(2*chan, chan)
                
    def forward(self, spec):
        #encoder, spec is the input array for composite spectrum
        x1 = self.down_conv_1(spec)
        x2 = self.max_pool_1(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_3(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_4(x7)
        x9 = self.down_conv_5(x8)

        # #decoder
        x10 = self.up_trans_1(x9)
        x10_pad = pad_1d(x10,x7)
        x11 = self.up_conv_1(torch.cat([x7, x10_pad], 1))
        x12 = self.up_trans_2(x11)
        x5_pad = pad_1d(x5, x12)
        x13 = self.up_conv_2(torch.cat([x5_pad, x12], 1))
        x14 = self.up_trans_3(x13)
        x14_pad = pad_1d(x14, x3)
        x15 = self.up_conv_3(torch.cat([x3, x14_pad], 1))
        x16 = self.up_trans_4(x15)
        x1_pad = pad_1d(x1, x16)
        x17 = self.up_conv_4(torch.cat([x1_pad, x16], 1))
        y = pad_1d(x17, spec)      
        return y