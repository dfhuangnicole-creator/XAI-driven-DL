import torch.nn as nn
import torch.nn.functional as F
import torch

class VAE(nn.Module):
    def __init__(self, input_dim=768, h1=512, h2=256, h3=128, latent_dim=50):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.LeakyReLU(0.2)
        )
        
        self.fc_mu = nn.Linear(h3, latent_dim)
        self.fc_logvar = nn.Linear(h3, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, h3)
        
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(h3),
            nn.LeakyReLU(0.2),
            
            nn.Linear(h3, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.2),
            
            nn.Linear(h1, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        recon = self.decoder(h)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0, kl_anneal=1.0, free_bits=0.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean') 
    
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits) - free_bits
    
    kl_div = kl_per_dim.mean() 
    
    kl_loss = (beta * kl_anneal) * kl_div
    
    return recon_loss + kl_loss, kl_per_dim.detach()