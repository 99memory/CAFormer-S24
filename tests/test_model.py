"""Basic unit tests for CAFormer."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from caformer import build_caformer


def test_forward_pass_output_shape():
    model = build_caformer(num_classes=5)
    images = torch.randn(2, 3, 224, 224)
    logits = model(images)
    assert logits.shape == (2, 5)


def test_forward_pass_requires_grad():
    model = build_caformer(num_classes=3)
    images = torch.randn(1, 3, 224, 224)
    targets = torch.tensor([1])
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item()
    assert grad_norm > 0
