import torch
from torchvision.models import resnet18


class NeuralNetwork(torch.nn.Module):
    '''
    MNIST용 Vanilla NN

    구조: Flatten → Linear(784, 512) → ReLU → Linear(512, 512) → ReLU → Linear(512, 10)

    Notes
    -----
    입력은 (N, 1, 28, 28) 또는 (N, 28, 28) 텐서
    출력은 소프트맥스 이전의 raw logits (shape: (N, 10))
    *사용하지 않음
    '''

    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork_color(torch.nn.Module):
    '''
    CIFAR-10용 Vanilla NN

    구조: Flatten → Linear(3072, 512) → ReLU → Linear(512, 512) → ReLU → Linear(512, 10)

    Notes
    -----
    입력은 (N, 3, 32, 32) 텐서
    단순 FC 구조이므로 공간적 특징을 학습하지 못해 정확도 한계 있음
    출력은 raw logits (shape: (N, 10))
    *사용하지 않음
    '''

    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(3 * 32 * 32, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def build_resnet18(num_classes=10, device=None):
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, num_classes)

    if device is not None:
        model = model.to(device)

    return model
