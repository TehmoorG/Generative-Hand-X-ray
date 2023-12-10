# Standard library imports
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# Third-party library imports
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


# CustomHandDataset class definition
class CustomHandDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

# Function to create data loaders
def create_data_loaders(image_dir, batch_size=64, test_size=0.2, transform=None):
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpeg')]
    train_paths, test_paths = train_test_split(image_paths, test_size=test_size, random_state=42)
    
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    train_dataset = CustomHandDataset(train_paths, transform=transform)
    test_dataset = CustomHandDataset(test_paths, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
