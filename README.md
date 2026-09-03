
# ISIC 2019 — Tratamento de Desbalanceamento de Classes

Projeto acadêmico de Deep Learning (Engenharia de Software) sobre classificação de lesões cutâneas no dataset ISIC 2019, com foco em técnicas de tratamento do desbalanceamento entre classes.

## Objetivo

Investigar como técnicas de tratamento do desbalanceamento de classes podem melhorar a classificação de lesões cutâneas minoritárias, utilizando Transfer Learning (ResNet50) no conjunto ISIC 2019.

**Pergunta central:** Qual estratégia consegue melhorar o desempenho nas classes minoritárias sem causar perda excessiva no desempenho geral?

## Dataset

[ISIC 2019 Skin Lesion Images for Classification](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification) — 25.331 imagens em 8 classes (MEL, NV, BCC, AK, BKL, DF, SCC, VASC).

## Abordagem

- **Baseline:** ResNet50 pré-treinada (ImageNet), Transfer Learning
- **Tratamentos comparados:** Data Augmentation, Class Weights, Focal Loss
- **Split:** treino/validação/teste (70/15/15), agrupado por `lesion_id` e estratificado por classe, para evitar vazamento de dados

## Métricas

Macro F1, Recall por classe, F1 por classe, Balanced Accuracy, Accuracy, Precision, matriz de confusão.

## Estrutura do repositório
├── data/ # dataset (não versionado)
├── notebooks/ # notebooks do Colab
├── src/ # código-fonte (dataset, models, losses, train, evaluate)
├── configs/ # configurações de experimentos (YAML)
├── isic2019_split.csv # split treino/val/teste por lesion_id
└── requirements.txt

## Ambiente

Treinamento via Google Colab (GPU T4). Dependências em `requirements.txt`.
