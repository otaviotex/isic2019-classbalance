"""
src/losses.py

Focal Loss para tratamento de desbalanceamento de classes.

Focal Loss reduz o peso de exemplos "fáceis" (já bem classificados) e aumenta
o peso relativo de exemplos "difíceis" — o que ajuda automaticamente as
classes minoritárias, que tendem a ser mais difíceis de classificar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Args:
        alpha: peso por classe (tensor de tamanho num_classes) ou None.
               Pode usar os mesmos pesos calculados para Class Weights.
        gamma: fator de foco. gamma=0 equivale ao CrossEntropyLoss padrão.
               Valores maiores (2, como no artigo original) aumentam o foco
               nos exemplos difíceis.
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Perda de entropia cruzada "crua", por exemplo (sem redução/média ainda)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")

        # pt = probabilidade que o modelo deu para a classe correta
        pt = torch.exp(-ce_loss)

        # Fator de foco: (1 - pt)^gamma
        # Se pt é alto (modelo já acerta com confiança), o fator fica pequeno
        # -> a perda desse exemplo é reduzida (o modelo "já sabe", não precisa focar nele)
        # Se pt é baixo (modelo erra ou tem pouca confiança), o fator fica próximo de 1
        # -> a perda desse exemplo é mantida quase igual (o modelo precisa aprender mais)
        focal_term = (1 - pt) ** self.gamma

        focal_loss = focal_term * ce_loss
        return focal_loss.mean()


def calcular_class_weights(contagem_por_classe, device):
    """
    Calcula pesos por classe para uso em CrossEntropyLoss(weight=...) ou FocalLoss(alpha=...).

    Usa a fórmula padrão: peso_classe = total_amostras / (num_classes * contagem_da_classe)
    Classes com menos exemplos recebem peso maior.

    Args:
        contagem_por_classe: Series/array com a contagem de imagens por classe,
                              na MESMA ORDEM de CLASSES (retornado por
                              ISIC2019Dataset.contagem_por_classe())
        device: torch.device
    """
    contagem = torch.tensor(contagem_por_classe.values, dtype=torch.float32)
    total = contagem.sum()
    num_classes = len(contagem)

    pesos = total / (num_classes * contagem)
    return pesos.to(device)