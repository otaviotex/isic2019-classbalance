"""
src/train.py

Loop de treinamento para o baseline: ResNet50, CrossEntropyLoss padrão
(sem nenhum tratamento de desbalanceamento — isso entra nas fases seguintes,
comparando com Class Weights e Focal Loss).
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    perda_total = 0.0
    acertos = 0
    total = 0

    for imagens, rotulos in dataloader:
        imagens, rotulos = imagens.to(device), rotulos.to(device)

        optimizer.zero_grad()
        saidas = model(imagens)
        perda = criterion(saidas, rotulos)
        perda.backward()
        optimizer.step()

        perda_total += perda.item() * imagens.size(0)
        predicoes = torch.argmax(saidas, dim=1)
        acertos += (predicoes == rotulos).sum().item()
        total += rotulos.size(0)

    return perda_total / total, acertos / total


@torch.no_grad()
def validar(model, dataloader, criterion, device):
    model.eval()
    perda_total = 0.0
    acertos = 0
    total = 0

    for imagens, rotulos in dataloader:
        imagens, rotulos = imagens.to(device), rotulos.to(device)
        saidas = model(imagens)
        perda = criterion(saidas, rotulos)

        perda_total += perda.item() * imagens.size(0)
        predicoes = torch.argmax(saidas, dim=1)
        acertos += (predicoes == rotulos).sum().item()
        total += rotulos.size(0)

    return perda_total / total, acertos / total


def treinar_modelo(model, train_loader, val_loader, device, criterion=None,
                    num_epochs=5, lr=1e-4, checkpoint_path="checkpoints/model_best.pth"):
    """
    Treina o modelo com o criterion (função de perda) fornecido.
    Salva o melhor checkpoint com base na perda de validação.

    Args:
        criterion: função de perda (ex: nn.CrossEntropyLoss(), nn.CrossEntropyLoss(weight=...),
                   FocalLoss(...)). Se None, usa CrossEntropyLoss() padrão (comportamento do baseline).
    """
    import os
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    melhor_perda_val = float("inf")
    historico = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoca in range(1, num_epochs + 1):
        inicio = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validar(model, val_loader, criterion, device)

        duracao = time.time() - inicio

        historico["train_loss"].append(train_loss)
        historico["train_acc"].append(train_acc)
        historico["val_loss"].append(val_loss)
        historico["val_acc"].append(val_acc)

        print(f"Época {epoca:2d}/{num_epochs} — "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} — "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} — "
              f"({duracao:.1f}s)")

        if val_loss < melhor_perda_val:
            melhor_perda_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Novo melhor modelo salvo em {checkpoint_path}")

    return historico


def treinar_baseline(model, train_loader, val_loader, device,
                      num_epochs=5, lr=1e-4, checkpoint_path="checkpoints/baseline_best.pth"):
    """
    Treina o modelo baseline com CrossEntropyLoss padrão (sem tratamento de
    desbalanceamento). Mantido por compatibilidade — equivale a chamar
    treinar_modelo(..., criterion=None).
    """
    return treinar_modelo(model, train_loader, val_loader, device,
                           criterion=None, num_epochs=num_epochs, lr=lr,
                           checkpoint_path=checkpoint_path)
