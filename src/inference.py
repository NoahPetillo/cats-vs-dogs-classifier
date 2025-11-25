from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms

from src.cnn import CNN

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model7.pth"
CLASS_NAMES = ("Cat", "Dog")
DEVICE = torch.device("cpu")

_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def _preprocess(image: Image.Image) -> torch.Tensor:
    return _TRANSFORMS(image.convert("RGB")).unsqueeze(0).to(DEVICE)


@lru_cache(maxsize=1)
def load_model() -> CNN:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. Please add model7.pth."
        )
    model = CNN()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
    """Run inference on a PIL image and return label, confidence, and all probs."""
    model = load_model()
    tensor = _preprocess(image)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs))
    confidence = float(probs[pred_idx])
    prob_map = {label: float(prob) for label, prob in zip(CLASS_NAMES, probs)}
    return CLASS_NAMES[pred_idx], confidence, prob_map

