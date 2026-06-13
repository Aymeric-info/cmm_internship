import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

"""
class LoadDataset(Dataset):
    def __init__(self, image_folder, is_grayscale):
        super().__init__()
        self.image_folder = image_folder
        self.is_grayscale = is_grayscale

        self.images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # self.images = self.images[:500]
        
        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_file = self.images[idx]
        image = Image.open(os.path.join(self.image_folder, image_file))

        if self.is_grayscale:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        image = self.transform(image)

        try:
            target_str = image_file.split("_")[0]
            target = torch.tensor(int(target_str), dtype=torch.long)
            
        except ValueError:
            target = torch.tensor(-1, dtype=torch.long) 

        return image, target


def load_data(is_training, data_type, is_grayscale):
    if is_training:
        if data_type == "anomaly":
            image_folder = "data/train/anomaly"
        elif data_type == "normal":
            image_folder = "data/train/normal"
        else:
            raise ValueError("data_type does not exist, use anomaly or normal.")
    else:
        if data_type == "anomaly":
            image_folder = "data/test/anomaly"
        elif data_type == "normal":
            image_folder = "data/test/normal"
        else:
            raise ValueError("data_type does not exist, use anomaly or normal.")

    
    dataset = LoadDataset(image_folder, is_grayscale)
    return DataLoader(dataset, batch_size=128, shuffle=True)
"""

def load_data(is_training, data_type, is_grayscale=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2))
    ])

    root_dir = './data'

    if data_type == "normal":
        split_type = 'digits'
        target_transform = lambda y: 0

    elif data_type == "anomaly":
        split_type = 'letters'
        target_transform = None

    else:
        raise ValueError("data_type doit être 'normal' ou 'anomaly'.")

    dataset = datasets.EMNIST(
        root=root_dir,
        split=split_type,
        train=is_training,
        download=True,
        transform=transform,
        target_transform=target_transform
    )
    
    return DataLoader(dataset, batch_size=128, shuffle=True)