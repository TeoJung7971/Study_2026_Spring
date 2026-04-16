import torch


def save_checkpoint(state, filename):
    '''
    Save a checkpoint dict via ``torch.save``.

    Parameters
    ----------
    state : dict
        State dictionary. Typically contains ``epoch``, ``model_state_dict``,
        ``optimizer_state_dict``, ``scheduler_state_dict``, ``best_acc``.
    filename : str
        Destination path.
    '''
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None, device='cpu'):
    '''
    Load a checkpoint and restore model/optimizer/scheduler states in-place.

    Parameters
    ----------
    filename : str
        Path to the checkpoint file.
    model : torch.nn.Module
        Model to restore weights into.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to restore state into.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Scheduler to restore state into.
    device : str or torch.device, optional
        ``map_location`` passed to ``torch.load``.

    Returns
    -------
    dict
        ``{'epoch': int, 'best_acc': float}``. Missing fields default to 0 / 0.0.
    '''
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch':    checkpoint.get('epoch', 0),
        'best_acc': checkpoint.get('best_acc', 0.0),
    }
