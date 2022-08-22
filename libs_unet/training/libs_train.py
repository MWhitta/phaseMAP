import torch
from libs_unet.training.model_delta import state_diff


def train_loop(dataloader, model, loss_fn, optimizer, writer, epoch, debug=False, nodes=[]):
    # number of batches
    size = len(dataloader.dataset)
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
            #print(f"loss: {loss:.7E}  [{current:>5d}/{size:>5d}]")
            if debug == True:
                update_dict = state_diff(model.state_dict(), init_dict)
                #print(update_dict)
                if len(nodes) == 0:
                    for key,value in update_dict.items():
                        print(f"{key}:\n{value}")
                else:
                    for key,value in update_dict.items():
                        if key in nodes:
                            print(f"{key}:\n{value}")
                            
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
