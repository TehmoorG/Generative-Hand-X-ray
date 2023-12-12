import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for generating images.

    Implements a standard VAE architecture with convolutional encoder and decoder layers.
    """

    def __init__(self):
        super(VAE, self).__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=4, stride=2, padding=1
        )  # Output: 16x16
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # Output: 8x8
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # Output: 4x4
        self.conv4 = nn.Conv2d(
            128, 256, kernel_size=4, stride=2, padding=1
        )  # Output: 2x2

        self.fc1 = nn.Linear(256 * 2 * 2, 128)  # Output size: mu
        self.fc2 = nn.Linear(256 * 2 * 2, 128)  # Output size: logvar
        # I used logvar isntead of std because it is more numerically stable to compute logvar directly and then exponentiate it later.

        # Decoder layers
        self.fc3 = nn.Linear(128, 256 * 2 * 2)  # Match to the output of the encoder
        self.deconv1 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )  # Output: 4x4
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # Output: 8x8
        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # Output: 16x16
        self.deconv4 = nn.ConvTranspose2d(
            32, 1, kernel_size=4, stride=2, padding=1
        )  # Output: 32x32

    def encode(self, x):
        """
        Encodes input to latent space representation.
        """
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h4_flattened = h4.view(
            -1, 256 * 2 * 2
        )  # Adjust the flattening to match the 256 * 2 * 2 size
        return self.fc1(h4_flattened), self.fc2(h4_flattened)

    def reparameterize(self, mu, logvar):
        """
        Applies reparameterization trick to sample from latent space.
        """
        std = torch.exp(0.5 * logvar)  # Compute std from logvar
        eps = torch.randn_like(std)  # Noise
        return mu + eps * std

    def decode(self, z):
        """
        Decodes latent space representation back to image.
        """
        h3 = F.relu(self.fc3(z))
        h3 = h3.view(-1, 256, 2, 2)  # Adjust this line to match the output size of fc3
        h4 = F.relu(self.deconv1(h3))
        h5 = F.relu(self.deconv2(h4))
        h6 = F.relu(self.deconv3(h5))
        x_recon = torch.tanh(self.deconv4(h6))
        return x_recon

    def forward(self, x):
        """
        Forward pass through the VAE.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """
    Calculates VAE loss combining reconstruction loss and KL divergence.
    """
    mse_loss = F.mse_loss(recon_x, x, reduction="sum")  # Summing instead of averaging
    kld_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )  # Kullback-Leibler divergence loss term
    return mse_loss + kld_loss
