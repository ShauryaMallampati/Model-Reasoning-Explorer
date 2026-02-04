#!/usr/bin/env python3
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "backend" / ".mre_cache"
CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(CACHE))

print("Downloading text LM (tiny GPT-2)...")
AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

print("Downloading text classifier (SST-2)...")
AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

print("Downloading vision model (resnet18)...")
resnet18(weights=ResNet18_Weights.DEFAULT)

print("Done. Cached to", CACHE)
