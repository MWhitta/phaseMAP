import torch
import torch.nn as nn
from libs_unet.training.model_delta import state_diff


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
