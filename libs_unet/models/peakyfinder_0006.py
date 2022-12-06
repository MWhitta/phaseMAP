##UNET variant with FC output for directly predicting element composition.
# retain the low/high intensity input split from 0002 network.
# add .Relu to final layer to force positive outputs
#https://www.kaggle.com/code/shtrausslearning/pytorch-cnn-binary-image-classification
#stick with FC -> Relu(FC) from 0005 but add raw input spectra (2 chan) to flattened UNet outputs
#Note this step seems to prevent learning, we see predictions of zero weights across samples!


import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_c, out_c, padding=3, kernel_size=6):
        super().__init__()
        self.double_conv = nn.Sequential(
        nn.Conv1d(in_c, out_c, padding=padding, kernel_size=kernel_size, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
        nn.Conv1d(out_c, out_c, padding=padding, kernel_size=kernel_size, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU())
    
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
          
        self.down_conv_1 = double_conv(2, chan)
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

        # compute the flatten size, pick 1000 for middle num_fc1
        # output "classes" here is 80 elements, predicting percentages
        # since we concatenate 2 inputs to final conv output, we have 84 
        self.num_flatten = (chan + 2) * self.l_spec
        self.dropout_rate = 0.2
        self.fc_mid = 1000
        self.fc1 = nn.Linear(self.num_flatten, self.fc_mid)
        self.fc2 = nn.Linear(self.fc_mid, 80)
                
    def forward(self, spec): #spec:  torch.Size([1, 2, 782])
        #encoder, spec is the input array for composite spectrum
        x1 = self.down_conv_1(spec) #x1: torch.Size([1, 82, 784])
        x2 = self.max_pool_1(x1) #x2:  torch.Size([1, 82, 393])
        x3 = self.down_conv_2(x2) #x3:  torch.Size([1, 164, 395])
        x4 = self.max_pool_2(x3) #x4:  torch.Size([1, 164, 198])
        x5 = self.down_conv_3(x4) #x5:  torch.Size([1, 328, 200])
        x6 = self.max_pool_3(x5) #x6:  torch.Size([1, 328, 101])
        x7 = self.down_conv_4(x6) #x7:  torch.Size([1, 656, 103])
        x8 = self.max_pool_4(x7) #x8:  torch.Size([1, 656, 52])
        x9 = self.down_conv_5(x8) #x9:  torch.Size([1, 1312, 54])

        # #decoder
        x10 = self.up_trans_1(x9) #x10:  torch.Size([1, 656, 108])
        #note pad seems to accept negative args, such that first can be shrunk to second!
        x10_pad = pad_1d(x10,x7) #x10_pad:  torch.Size([1, 656, 103])
        x11 = self.up_conv_1(torch.cat([x7, x10_pad], 1)) #x11:  torch.Size([1, 656, 105])
        x12 = self.up_trans_2(x11) #x12:  torch.Size([1, 328, 210])
        x5_pad = pad_1d(x5, x12) #x5_pad:  torch.Size([1, 328, 210])
        x13 = self.up_conv_2(torch.cat([x5_pad, x12], 1)) #x13:  torch.Size([1, 328, 212])
        x14 = self.up_trans_3(x13) #x14:  torch.Size([1, 164, 424])
        x14_pad = pad_1d(x14, x3) #x14_pad:  torch.Size([1, 164, 395])
        x15 = self.up_conv_3(torch.cat([x3, x14_pad], 1)) #x15:  torch.Size([1, 164, 397])
        x16 = self.up_trans_4(x15) #x16:  torch.Size([1, 82, 794])
        x1_pad = pad_1d(x1, x16) #x1_pad:  torch.Size([1, 82, 794])
        x17 = self.up_conv_4(torch.cat([x1_pad, x16], 1)) #x17:  torch.Size([1, 82, 796])
        x18 = pad_1d(x17, spec) #x18:  torch.Size([1, 82, 782])
#new stuff here flattens all 82 channels plus the two input channels
# then applies fc-relu-fc to get 80 element fracs
#note -1 in view indicates to figure out what number to put in that dimension on other specified
#Concat on the channel dimension =1
        x19 = torch.cat([x18, spec], dim=1) #x19:  torch.Size([1, 84, 782])
#Now with effectively 84 channels, flatten those channels
        x20 = x19.view(-1, self.num_flatten) #x20:  torch.Size([1, 65688])
        x21 = self.fc1(x20) #x21:  torch.Size([1, 1000])
        x22 = F.dropout(x21, self.dropout_rate) #x22:  torch.Size([1, 1000])
        x23 = F.relu(self.fc2(x22)) #x23:  torch.Size([1, 80])
        return x23