import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "SCC", "VASC"]
CLASSE_PARA_INDICE = {classe: i for i, classe in enumerate(CLASSES)}

class ISIC2019Dataset(Dataset):
    """
    Dataset PyTorch para o ISIC 2019.
    Lê o isic2019_split.csv (colunas: image, lesion_id, classe, split) e carrega
    as imagens a partir das pastas por classe (data/MEL/, data/NV/,...).
    
    Args:
        csv_path: caminho para isic2019_split.csv
        data_dir: pasta raiz onde estão as subpastas por classe (MEL/, NV/, ...)
        split: "train", "val" ou "test"
        transform: transform torchvision a ser aplicado em cada imagem
    """
    def __init__(self, csv_path, data_dir, split, transform=None):
        df = pd.read_csv(csv_path)
        if split not in ("train", "val", "test"):
            raise ValueError(f"split inválido: {split!r}. Use 'train', 'val' ou 'test'")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        
        if len(self.df) == 0:
            raise ValueError(f"Nenhuma linha encontrada para split={split!r} em {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        classe = row["classe"]
        image_id = row["image"]
        img_path = os.path.join(self.data_dir, classe, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        label = CLASSE_PARA_INDICE[classe]
        return image, label

    def contagem_por_classe(self):
        """Retorna quantas imagens desse split existem por classe (útil para class weights)"""
        return self.df["classe"].value_counts().reindex(CLASSES, fill_value=0)

if __name__ == "__main__":
    # Teste rápido
    import argparse
    # ajuste os caminhos conforme seu ambiente
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="isic2019_split.csv")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    
    for split in ["train", "val", "test"]:
        ds = ISIC2019Dataset(args.csv, args.data_dir, split)
        print(f"[{split}]: {len(ds)} imagens")
        print(ds.contagem_por_classe())
        img, label = ds[0]
        print(f"Exemplo - tamanho da imagem: {img.size}, label: {label} ({CLASSES[label]})")
        print()