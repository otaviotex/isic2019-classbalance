"""
src/models.py

Monta a ResNet50 pré-treinada (ImageNet) adaptada para as 8 classes do ISIC 2019.
"""

import torch.nn as nn
from torchvision import models


NUM_CLASSES = 8


def get_resnet50(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
    """
    Args:
        num_classes: número de classes de saída (8 para o ISIC 2019)
        pretrained: usar pesos pré-treinados no ImageNet
        freeze_backbone: se True, congela todas as camadas exceto a última
                         (treina só o classificador final — mais rápido, mas
                         geralmente com desempenho pior que fine-tuning completo)
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Troca a última camada (originalmente 1000 classes do ImageNet) por 8 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    # A camada nova (model.fc) sempre é treinável, mesmo com freeze_backbone=True

    return model