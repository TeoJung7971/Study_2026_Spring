import torch


def train_loop(dataloader, model, loss_fn, optimizer, device, grad_clip=None):
    '''
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
    model : torch.nn.Module
    loss_fn : callable
    optimizer : torch.optim.Optimizer
    device : str or torch.device
    grad_clip : float, optional
    '''
    model.train()
    size = len(dataloader.dataset)
    running_loss = 0.0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        running_loss += loss.item() * X.size(0)
        correct      += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            print(f'loss: {loss.item():.4f}  [{batch * len(X):>5d}/{size:>5d}]')

    return running_loss / size, correct / size
