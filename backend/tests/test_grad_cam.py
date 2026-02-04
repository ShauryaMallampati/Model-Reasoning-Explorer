import torch
from torch import nn

from mre_backend.analyzers.grad_cam import GradCamAnalyzer


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


def test_grad_cam_basic():
    model = TinyCNN()
    x = torch.randn(1, 3, 8, 8)
    x.requires_grad_(True)
    logits = model(x)
    loss = logits.max()
    loss.backward()

    class DummyContext:
        task_type = "image_classification"
        capture = type(
            "Capture",
            (),
            {
                "activations": {"conv": model.conv(x).detach()},
                "gradients": {"conv": model.conv(x).detach()},
            },
        )

    analyzer = GradCamAnalyzer()
    out = analyzer.run(DummyContext())
    assert "heatmap" in (out.arrays or {})
