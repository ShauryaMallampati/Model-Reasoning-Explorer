from __future__ import annotations

DEMO_MODELS = [
    {
        "id": "sshleifer/tiny-gpt2",
        "task_type": "text_lm",
        "adapter": "text_transformer",
        "description": "Tiny GPT-2 for quick demos",
    },
    {
        "id": "distilbert-base-uncased-finetuned-sst-2-english",
        "task_type": "text_classification",
        "adapter": "text_transformer",
        "description": "DistilBERT SST-2 classifier",
    },
    {
        "id": "resnet18",
        "task_type": "image_classification",
        "adapter": "vision_resnet",
        "description": "ResNet18 ImageNet",
    },
]
