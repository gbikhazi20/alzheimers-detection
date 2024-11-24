import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from transformers import ViTForImageClassification
import models

# Constants
NUM_CLASSES = 4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ['Non Demented', 'Very mild Dementia', 'Mild Dementia', 'Moderate Dementia']


class AlzheimersDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            with Image.open(self.image_paths[idx]) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            return torch.zeros((3, 224, 224)), self.labels[idx]
        

# Assumes model saved at model_path is a state_dict
# this is jank :')
def load_model(model_path):
    model_map = {
        r'models\resnet.pth': models.Resnet(),
        r'models\resnet_attention.pth': models.ResnetAttention(),
        r'models\vit.pth': ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=NUM_CLASSES)
    }

    if model_path not in model_map:
        raise ValueError(f"Model {model_path} not found")
    
    state_dict = torch.load(model_path, map_location='cpu')
    
    model = model_map[model_path]
    model.load_state_dict(state_dict)

    return model


def create_dataset_df(base_path):
    data = []
    class_mapping = {
        'Non Demented': 0,
        'Very mild Dementia': 1,
        'Mild Dementia': 2,
        'Moderate Dementia': 3
    }

    for class_name, label in class_mapping.items():
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} not found")
            continue

        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in image_files:
            data.append({
                'path': os.path.join(class_path, img_file),
                'label': label,
                'class': class_name
            })

    df = pd.DataFrame(data)

    return df

def create_val_loader(df, val_transform, batch_size=32, val_size=0.2):
    train_loader, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df['label'],
        random_state=42
    )

    val_dataset = AlzheimersDataset(
        image_paths=val_df['path'].values,
        labels=val_df['label'].values,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return val_loader