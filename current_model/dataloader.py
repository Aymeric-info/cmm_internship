import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

def load_data(is_training, data_type, target_anomaly_idx=None, is_grayscale=True, num_samples=None, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2))
    ])

    root_dir = './data'

    # Configuration du split et de la transformation des labels (target_transform)
    if data_type == "normal":
        split_type = 'digits'
        target_transform = None
    elif data_type == "anomaly":
        split_type = 'letters'
        if target_anomaly_idx is not None and not (1 <= target_anomaly_idx <= 5):
            raise ValueError(f"L'index de l'anomalie ({target_anomaly_idx}) doit être entre 1 (A) et 5 (E).")
        
        # Le décalage de +9 est appliqué nativement par le Dataloader à la récupération de l'élément
        target_transform = transforms.Lambda(lambda y: y + 9)
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
    
    if num_samples is not None:
        # Note : dataset.targets n'est pas affecté par target_transform, 
        # il contient toujours les labels bruts (0-9 pour digits, 1-26 pour letters)
        targets = dataset.targets
        
        if data_type == "normal":
            num_classes = len(torch.unique(targets))
            samples_per_class = num_samples // num_classes
            
            indices = []
            for class_id in range(num_classes):
                class_indices = torch.where(targets == class_id)[0]
                perm = torch.randperm(len(class_indices))[:samples_per_class]
                indices.extend(class_indices[perm].tolist())
            
            dataset = Subset(dataset, indices)
        
        elif data_type == "anomaly":
            if target_anomaly_idx is None:
                raise ValueError("target_anomaly_idx est requis pour le data_type 'anomaly'.")

            label_indices = torch.where(targets == target_anomaly_idx)[0]
            
            if len(label_indices) == 0:
                raise ValueError(f"Aucune donnée trouvée pour l'index {target_anomaly_idx}")
                
            samples_to_take = min(len(label_indices), num_samples)
            perm = torch.randperm(len(label_indices))[:samples_to_take]
            
            dataset = Subset(dataset, label_indices[perm].tolist())

    return DataLoader(dataset, batch_size, shuffle=True)