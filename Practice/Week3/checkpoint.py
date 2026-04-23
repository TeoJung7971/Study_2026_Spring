import torch


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, device='cpu'):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        # Resume하여 학습 진행하면 Optimizer state도 복원 필요
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def read_checkpoint(filename, device='cpu'):
    return torch.load(filename, map_location=device)
