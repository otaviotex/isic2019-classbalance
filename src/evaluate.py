"""
src/evaluate.py

Calcula as métricas de avaliação definidas para o projeto:
Macro F1, Recall por classe, F1 por classe, Balanced Accuracy,
Accuracy, Precision, matriz de confusão.
"""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    balanced_accuracy_score, accuracy_score, confusion_matrix,
)


CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "SCC", "VASC"]


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Roda o modelo em todo o dataloader e retorna (rótulos_reais, rótulos_previstos)."""
    model.eval()
    todos_rotulos = []
    todas_predicoes = []

    for imagens, rotulos in dataloader:
        imagens = imagens.to(device)
        saidas = model(imagens)
        predicoes = torch.argmax(saidas, dim=1).cpu().numpy()

        todos_rotulos.extend(rotulos.numpy())
        todas_predicoes.extend(predicoes)

    return np.array(todos_rotulos), np.array(todas_predicoes)


def calcular_metricas(y_true, y_pred):
    """Calcula todas as métricas definidas para o projeto."""
    metricas = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Métricas por classe
    f1_por_classe = f1_score(y_true, y_pred, average=None, zero_division=0)
    recall_por_classe = recall_score(y_true, y_pred, average=None, zero_division=0)
    precision_por_classe = precision_score(y_true, y_pred, average=None, zero_division=0)

    metricas["por_classe"] = {
        classe: {
            "f1": f1_por_classe[i],
            "recall": recall_por_classe[i],
            "precision": precision_por_classe[i],
        }
        for i, classe in enumerate(CLASSES)
    }

    metricas["matriz_confusao"] = confusion_matrix(y_true, y_pred)

    return metricas


def imprimir_metricas(metricas):
    """Imprime um resumo legível das métricas."""
    print(f"Accuracy:          {metricas['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metricas['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {metricas['macro_f1']:.4f}")
    print(f"Macro Precision:   {metricas['macro_precision']:.4f}")
    print(f"Macro Recall:      {metricas['macro_recall']:.4f}")
    print()
    print(f"{'Classe':6s} {'F1':>6s} {'Recall':>8s} {'Precision':>10s}")
    for classe, valores in metricas["por_classe"].items():
        print(f"{classe:6s} {valores['f1']:6.3f} {valores['recall']:8.3f} {valores['precision']:10.3f}")
    print()
    print("Matriz de confusão (linhas=real, colunas=previsto):")
    print(CLASSES)
    print(metricas["matriz_confusao"])
