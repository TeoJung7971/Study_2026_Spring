import torch


def test_loop(dataloader, model, loss_fn, device):
    '''
    Evaluate the model on a dataloader and return epoch-level metrics.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
    model : torch.nn.Module
    loss_fn : callable
    device : str or torch.device

    Returns
    -------
    tuple of float
        ``(avg_loss, accuracy)`` over the full dataset.
    '''
    model.eval()
    size        = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred       = model(X)
            test_loss += loss_fn(pred, y).item()
            correct   += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy   = correct / size
    print(f'Accuracy: {accuracy * 100:.1f}%,  Avg loss: {test_loss:.4f}\n')

    return test_loss, accuracy
