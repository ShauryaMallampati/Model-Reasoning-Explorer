#!/usr/bin/env python3
"""Download the demo weights into the standard Hugging Face and Torch caches."""

from torchvision.models import ResNet18_Weights, resnet18
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

print("Downloading tiny GPT-2...")
AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2", trust_remote_code=False)
AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2", trust_remote_code=False)

print("Downloading the SST-2 text classifier...")
classifier = "distilbert-base-uncased-finetuned-sst-2-english"
AutoTokenizer.from_pretrained(classifier, trust_remote_code=False)
AutoModelForSequenceClassification.from_pretrained(classifier, trust_remote_code=False)

print("Downloading ResNet-18...")
resnet18(weights=ResNet18_Weights.DEFAULT)
print("Models downloaded to the standard Hugging Face and Torch caches.")
