import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from libs_unet.training.model_delta import state_diff
from libs_unet.training.spec_maker import spectrum_maker


def train_loop(dataloader, model, loss_fn, optimizer, writer, epoch, log_interval=10, debug=False, bsize=1):
    #capture starting state of model for difference reporting under debug
    if debug == True:
        #note named_parameters is an iterator
        #may access new values if simply assigned to a variable now
        #a comprehension on the iterator seems to get deferred and hence has newer values
        #To actually store state and accomlish subscript reference (diff two) we make dict
        init_wts = {}
        for k, v in model.named_parameters():
            init_wts[k] = v.clone() #v is a Parameter. Clone is detached tensor

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        batch_n = batch+1
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #leverages tensor gradients from backward()

        if batch_n  % log_interval == 0: #write to tensorboard
            print(f"loss: {loss}")
            loss, current = loss.item(), epoch * batch_n * bsize
            writer.add_scalar("Loss/train", loss, current)
            #detailed logging on nodes info for debug=True
            if debug == True:
                #returns a dictionary node:tensor (3x3) new/old/diff with meanabsval, range, var
                update_dict = state_diff(model.named_parameters(), init_wts)
                #print(update_dict.items())
                for key, value in update_dict.items():
                    writer.add_scalars(key, value, epoch * batch_n * bsize)
                
                #store model dictionary parameter state to compare against with next check            
                for k, v in model.named_parameters():
                    init_wts[k] = v.clone()
    return loss


def test_loop(dataloader, model, loss_fn, writer, epoch):
    num_batches = len(dataloader)
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches #should be a mean of batch means
    writer.add_scalar("Loss/valid", test_loss, epoch)
    #print(f"Test Error: \n Avg loss: {test_loss:.7E} \n")
    return test_loss

class Custom_Wgt_MSE(nn.Module): 
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
    
    def forward(self, input, target):
        el_err = torch.sum(((input - target) ** 2), axis=2)
        el_wt_err = torch.mul(el_err, self.weights)
        loss_value = torch.mean(el_wt_err)
        return loss_value

class El80Dataset(Dataset):
    def __init__(self, x_data_path, y_data_path):
        #define constant(s) for raw to model transforms
        self.nist_mult = 1.17
        # self.mmapped acts like a numpy array
        self.x_data = np.load(x_data_path, mmap_mode='r+')
        # loading the labels
        self.y_data = np.load(y_data_path, mmap_mode='r+')

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        x_samp = self.nist_mult * torch.tensor(self.x_data[idx,None,:])
        y_samp = self.nist_mult * torch.tensor(self.y_data[idx])
        #log transform data, add small offset 1 so zero points remain ~zero on log scale
        x_samp = torch.log(x_samp + 1)
        y_samp = torch.log(y_samp + 1)
        sample = (x_samp, y_samp)
        return sample

# https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
class RandomSpectrumDataset(IterableDataset):
    def __init__(self, min_el, max_el):
        super(RandomSpectrumDataset).__init__()
        self.maker = spectrum_maker()
        #random generator
        self.rng = np.random.default_rng()
        self.el_range = range(min_el, max_el + 1) #allowable number of elements
        self.ind_range = range(self.maker.max_z)

        
    def __next__(self):
        fracs = np.zeros(self.maker.max_z)
        num_el = self.rng.choice(self.el_range, 1) #determine number of elements
        el_ind = self.rng.choice(self.ind_range, num_el) #randomly pick el indices
        fracs[el_ind] = self.rng.random(num_el)
        fracs /= np.sum(fracs) #fractions sum to 1
        wave, spec, spec_array = self.maker.make_spectra(fracs)
        x_data = torch.tensor(spec[None,:].astype('float32'))
        y_data = torch.tensor(spec_array.astype('float32'))

        return (x_data, y_data)

    def __iter__(self):
        return self