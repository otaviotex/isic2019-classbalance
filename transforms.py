"""
Transforms de pré-processamento para uso com ResNet50 pré-treinada no ImageNet.
get_train_transform(): transforms para treino, incluindo augmentation leve
(a estratégia completa de Data Augmentation como tratamento de desbalanceamento
é decidida na Fase 7 - aqui só entra o augmentation "padrão" de treino)
get_eval_transform(): transforms para validação/teste, sem augmentation
"""
from torchvision import transforms

IMAGE_SIZE = 224
# Médias e desvios-padrão do ImageNet (esperados pela ResNet50 pré-treinada)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transform(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transform(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])