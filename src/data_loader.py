# Standard library imports
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# Third-party library imports
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class CustomHandDataset(Dataset):
    """
    A custom dataset class for hand X-ray images.

    This class is used to load hand X-ray images from a given list of file paths.
    It supports transforming the images as specified.

    Attributes:
        image_paths (list): A list of file paths to the images.
        transform (callable, optional): An optional transform to be applied to the images.

    Args:
        image_paths (list): A list of file paths to the images.
        transform (callable, optional): An optional transform to be applied on a sample.
    """

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image from the dataset at the specified index.

        Args:
            idx (int): The index of the image to be retrieved.

        Returns:
            PIL.Image.Image: The transformed image at the specified index.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image


# Function to create data loaders
def create_data_loaders(image_dir, batch_size=64, test_size=0.2, transform=None):
    """
    Creates data loaders for training and testing datasets of hand X-ray images.

    This function takes an image directory and optional parameters to create
    DataLoader objects for the training and testing datasets.

    Args:
        image_dir (str): Directory containing hand X-ray images.
        batch_size (int, optional): Batch size for the data loaders. Default is 64.
        test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
        transform (callable, optional): Optional transform to be applied on a sample. Default is None.

    Returns:
        tuple: A tuple containing the DataLoader objects for the training and testing datasets.
    """
    image_paths = [
        os.path.join(image_dir, img)
        for img in os.listdir(image_dir)
        if img.endswith(".jpeg")
    ]
    train_paths, test_paths = train_test_split(
        image_paths, test_size=test_size, random_state=42
    )

    if transform is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    train_dataset = CustomHandDataset(train_paths, transform=transform)
    test_dataset = CustomHandDataset(test_paths, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
