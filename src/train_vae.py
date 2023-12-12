import torch
from vae_model import VAE, vae_loss


def train(model, device, train_loader, optimizer):
    """
    Trains the VAE model for one epoch.

    This function iterates over the training dataset and updates the model's weights based on the calculated loss.

    Args:
        model (torch.nn.Module): The VAE model to be trained.
        device (torch.device): The device (CPU or GPU) to run the training on.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer to use for updating model weights.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()  # Ensure the model is in training mode
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(
            data
        )  # Use the passed model instead of a global vae
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # Calculate and return average loss
    average_loss = train_loss / len(train_loader.dataset)
    return average_loss


def validate(model, device, test_loader):
    """
    Validates the VAE model on a test dataset.

    This function runs the model in evaluation mode and calculates the loss on the test dataset.
    No backpropagation or weight updates are performed during validation.

    Args:
        model (torch.nn.Module): The VAE model to be validated.
        device (torch.device): The device (CPU or GPU) to perform validation on.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        float: The average validation loss for the test dataset.
    """
    model.eval()  # Ensure the model is in evaluation mode
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)  # Use the passed model
            test_loss += vae_loss(recon, data, mu, logvar).item()

    # Calculate and return average loss
    average_loss = test_loss / len(test_loader.dataset)
    return average_loss
