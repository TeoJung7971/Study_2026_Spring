import torch

from util import move_batch_to_device


def get_loss_and_logits(outputs):
    '''
    Custom GRU Output과 Huggingface Model output 중 공통 부분만 사용
    -> 형식 차이로 인한 혼동 없도록
    '''
    if isinstance(outputs, dict):
        return outputs.get('loss'), outputs.get('logits')
    return outputs.loss, outputs.logits


def train_loop(dataloader, model, optimizer, device, grad_clip=None):
    '''
    One Epoch의 Training step을 수행

    - Forward
    - Backward
    - Gradient clipping
    - Parameter update
    - Accuracy 
    '''
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_size = len(dataloader.dataset)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        # set_to_none=True: 효율 향상
        optimizer.zero_grad(set_to_none=True)

        outputs = model(**batch)
        loss, logits = get_loss_and_logits(outputs)
        loss.backward()

        if grad_clip is not None:
            # Gradient Clipping Function
            # grad_clip 값은 arg로 받음 (default = 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        batch_size = batch['labels'].size(0)
        total_loss += loss.item() * batch_size # 가중합 통한 전체 Loss 계산
        total_correct += (logits.argmax(dim=1) == batch['labels']).sum().item()

    return total_loss / total_size, total_correct / total_size
