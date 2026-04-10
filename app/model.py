"""Feature extraction models for image similarity search."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

LOGGER = logging.getLogger(__name__)
ModelName = Literal["resnet50", "clip"]


class FeatureExtractor:
    """Image feature extractor with pluggable model backend."""

    def __init__(self, model_name: ModelName = "resnet50") -> None:
        """Initialize model and transforms for embedding extraction."""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(model_name)
        self.transform = self._build_transform(model_name)

    def _build_model(self, model_name: ModelName) -> nn.Module:
        """Build and return a pretrained embedding model."""
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            model.fc = nn.Identity()
        elif model_name == "clip":
            if not hasattr(models, "clip_vit_b_32"):
                raise ValueError("CLIP model not available in installed torchvision version.")
            weights = models.CLIP_ViT_B_32_Weights.DEFAULT
            model = models.clip_vit_b_32(weights=weights)
            model = model.visual
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        model.eval()
        model.to(self.device)
        LOGGER.info("Loaded %s on %s", model_name, self.device)
        return model

    def _build_transform(self, model_name: ModelName):
        """Create preprocessing transform for selected model."""
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            return weights.transforms()

        weights = models.CLIP_ViT_B_32_Weights.DEFAULT
        return weights.transforms()

    @torch.inference_mode()
    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract a single embedding from PIL image.

        Returns:
            A numpy float32 vector with shape (embedding_dim,).
        """
        image_rgb = image.convert("RGB")
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        output = self.model(tensor)

        if isinstance(output, tuple):
            output = output[0]

        embedding = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return embedding
