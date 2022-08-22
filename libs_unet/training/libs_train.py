import torch
from libs_unet.training.model_delta import state_diff


def train_loop(dataloader, model, loss_fn, optimizer, writer, epoch):
    #capture starting state of model for difference reporting
    init_dict = model.state_dict()

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #leverages tensor gradients from backward()

        if batch % 10 == 0:
            loss, current = loss.item(), epoch * batch * len(X)
            writer.add_scalar("Loss/train", loss, current)
            #returns a dictionary node:tensor (3x3) new/old/diff with meanabsval, range, var
            update_dict = state_diff(model.state_dict(), init_dict)
            #create dict for TensorBoard https://pytorch.org/docs/stable/tensorboard.html
            #create graph for each node with new and delta data (old just lags new by one)
            #rely on filtering in TB with node name if needed (has regex)
            #the "x" variable for all the "y" tags will be epoch*batch*len
            #may make sense to just make state_diff return this format directly
            
            for key in update_dict.keys():
                node_dict = {}
                node_dict['new_mav'] = update_dict[key][0][0]
                node_dict['new_rng'] = update_dict[key][0][1]
                node_dict['new_var'] = update_dict[key][0][2]
                node_dict['diff_mav'] = update_dict[key][2][0]
                node_dict['diff_rng'] = update_dict[key][2][1]
                node_dict['diff_var'] = update_dict[key][2][2]
                #write the node statistics to TB
                writer.add_scalars(key, node_dict, epoch * batch * len(X))
            
            #store model dictionary state to compare against with next check            
            init_dict = model.state_dict()


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
