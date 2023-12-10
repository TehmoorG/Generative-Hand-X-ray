import torch
from vae_model import VAE, vae_loss

def train(model, device, train_loader, optimizer):
    model.train()  # Ensure the model is in training mode
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)  # Use the passed model instead of a global vae
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # Calculate and return average loss
    average_loss = train_loss / len(train_loader.dataset)
    return average_loss

def validate(model, device, test_loader):
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
