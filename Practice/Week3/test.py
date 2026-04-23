import torch

from train import get_loss_and_logits
from util import move_batch_to_device


def test_loop(dataloader, model, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_size = len(dataloader.dataset)

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss, logits = get_loss_and_logits(outputs)

            batch_size = batch['labels'].size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == batch['labels']).sum().item()

    return total_loss / total_size, total_correct / total_size
