import torch
import torch.nn as nn


def double_conv(in_c, out_c, kernel_size=6):
    conv = nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=kernel_size),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_c, out_c, kernel_size=kernel_size),
        nn.ReLU(inplace=True))
        
    return conv

#TODO consider if this needs to fail when target is larger
def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[-1]
    tensor_size = tensor.size()[-1]
    delta = abs(target_size - tensor_size)
    
    if delta % 2 > 0:
        delta = delta // 2 
        return tensor[:, :, delta:(tensor_size - delta - 1)]
    else:
        delta = delta // 2
        return tensor[:, :, delta:tensor_size - delta]


class LIBSUNet(nn.Module):
    """ UNet for single channel LIBS speactra data """
    def __init__(self, max_z, l_spec):
        super(LIBSUNet, self).__init__()
        #max_z defines #elements, we add artifact and noise channels
        self.channels = max_z + 2
        #l_spec defines the length of spectral input array
        self.l_spec = l_spec
        chan = self.channels
          
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, chan)
        self.down_conv_2 = double_conv(chan, 2*chan)
        self.down_conv_3 = double_conv(2*chan, 4*chan)
        self.down_conv_4 = double_conv(4*chan, 8*chan)
        self.down_conv_5 = double_conv(8*chan, 16*chan)
        
        
        self.up_trans_1 = nn.ConvTranspose1d(
            in_channels=16*chan,
            out_channels=8*chan,
            kernel_size=2,
            stride=2)
        self.up_conv_1 = double_conv(16*chan, 8*chan)
        
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
        self.up_conv_3 = double_conv(4*chan, 2*chan)
        
        self.up_trans_4 = nn.ConvTranspose1d(
            in_channels=2*chan,
            out_channels=chan,
            kernel_size=2,
            stride=2)
        self.up_conv_4 = double_conv(2*chan, chan)
        
        #need to fully build out feature variation formulae. Using 298 from worked example
        self.out = nn.Linear(298, l_spec)
        
    def forward(self, spec):
        #encoder, spec is the input array for composite spectrum
        x1 = self.down_conv_1(spec) #will crop/cat
        x2 = self.max_pool(x1)
        x3 = self.down_conv_2(x2) #will crop/cat
        x4 = self.max_pool(x3)
        x5 = self.down_conv_3(x4) #will crop/cat
        x6 = self.max_pool(x5)
        x7 = self.down_conv_4(x6) #will crop/cat
        x8 = self.max_pool(x7)
        x9 = self.down_conv_5(x8)
        
        #decoder
        x10 = self.up_trans_1(x9)
        y7 = crop_img(x7, x10)
        x11 = self.up_conv_1(torch.cat([y7, x10], 1))
        x12 = self.up_trans_2(x11)
        y5 = crop_img(x5, x12)
        x13 = self.up_conv_2(torch.cat([y5, x12], 1))
        x14 = self.up_trans_3(x13)
        y3 = crop_img(x3, x14)
        x15 = self.up_conv_3(torch.cat([y3, x14], 1))
        x16 = self.up_trans_4(x15)
        y1 = crop_img(x1, x16)
        x17 = self.up_conv_4(torch.cat([y1, x16], 1))
        y = self.out(x17)
        
        return y