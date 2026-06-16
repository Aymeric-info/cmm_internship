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

def load_data(is_training, data_type, is_grayscale=True, num_samples=None):
    from torch.utils.data import Subset
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2))
    ])

    root_dir = './data'

    if data_type == "normal":
        split_type = 'digits'

    elif data_type == "anomaly":
        split_type = 'letters'

    else:
        raise ValueError("data_type doit être 'normal' ou 'anomaly'.")

    # Charger le dataset sans target_transform d'abord pour pouvoir filtrer
    dataset = datasets.EMNIST(
        root=root_dir,
        split=split_type,
        train=is_training,
        download=True,
        transform=transform,
        target_transform=None
    )
    
    # Traiter num_samples selon le type de données (proposé par l'ia)
    if num_samples is not None:
        targets = dataset.targets
        
        if data_type == "normal":
            # Pour les chiffres : équilibrer entre les classes
            # Chaque classe aura num_samples / nombre_de_classes samples
            num_classes = len(torch.unique(targets))
            samples_per_class = num_samples // num_classes
            
            indices = []
            for class_id in range(num_classes):
                class_indices = torch.where(targets == class_id)[0]
                # Prendre aléatoirement samples_per_class indices de cette classe
                perm = torch.randperm(len(class_indices))[:samples_per_class]
                indices.extend(class_indices[perm].tolist())
            
            dataset = Subset(dataset, indices)
        
        elif data_type == "anomaly":
            # Pour les lettres : prendre seulement 5 lettres et num_samples par lettre
            unique_labels = torch.unique(targets).sort()[0]
            selected_labels = unique_labels[:5]  # Les 5 premières lettres
            
            indices = []
            for label in selected_labels:
                label_indices = torch.where(targets == label)[0]
                # Prendre aléatoirement num_samples indices pour cette lettre
                perm = torch.randperm(len(label_indices))[:num_samples]
                indices.extend(label_indices[perm].tolist())
            
            dataset = Subset(dataset, indices)
    
    # Appliquer la target_transform pour "normal" (mettre tous les targets à 0)
    if data_type == "normal":
        class TargetTransformWrapper(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                image, target = self.dataset[idx]
                return image, 0
        
        dataset = TargetTransformWrapper(dataset)
    
    return DataLoader(dataset, batch_size=128, shuffle=True)