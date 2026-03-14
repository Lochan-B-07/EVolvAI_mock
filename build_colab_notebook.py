import json

code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- 1. DATA LOADER ---
class EVDemandDataset(Dataset):
    def __init__(self, num_samples=1000, num_nodes=50, sequence_length=24):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.seq_len = sequence_length
        base_demand = np.random.uniform(10, 100, (num_samples, sequence_length, num_nodes))
        diurnal = np.array([1 + np.sin((h - 12) * np.pi / 12) for h in range(sequence_length)])
        diurnal = np.clip(diurnal, 0.5, 2.0).reshape(1, sequence_length, 1)
        self.charge_data = (base_demand * diurnal).astype(np.float32)
        self.weather_data = np.random.uniform(-10, 40, (num_samples, sequence_length, 1)).astype(np.float32)
        self.charge_data = (self.charge_data - np.mean(self.charge_data)) / (np.std(self.charge_data) + 1e-5)
        self.weather_data = (self.weather_data - np.mean(self.weather_data)) / (np.std(self.weather_data) + 1e-5)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        charge = torch.tensor(self.charge_data[idx])
        weather = torch.tensor(self.weather_data[idx])
        combined = torch.cat([charge, weather], dim=-1)
        return combined

def get_dataloader(batch_size=32, num_nodes=50):
    dataset = EVDemandDataset(num_nodes=num_nodes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 2. MODELS ---
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, stride=stride, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class GenerativeCounterfactualVAE(nn.Module):
    def __init__(self, num_features, seq_len=24, latent_dim=16, cond_dim=2):
        super(GenerativeCounterfactualVAE, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        tcn_channels = [32, 64]
        self.encoder_tcn = TemporalConvNet(num_inputs=num_features, num_channels=tcn_channels)
        tcn_out_dim = seq_len * tcn_channels[-1] 
        self.fc_mu = nn.Linear(tcn_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(tcn_out_dim, latent_dim)
        dec_in_dim = latent_dim + cond_dim
        self.decoder_fc = nn.Sequential(
            nn.Linear(dec_in_dim, 128), nn.ReLU(), nn.Linear(128, tcn_out_dim), nn.ReLU()
        )
        self.decoder_tcn = TemporalConvNet(num_inputs=tcn_channels[-1], num_channels=[32, num_features])

    def encode(self, x):
        h = self.encoder_tcn(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        zc = torch.cat([z, condition], dim=-1)
        h = self.decoder_fc(zc)
        h = h.view(h.size(0), 64, self.seq_len)
        recon_x = self.decoder_tcn(h)
        return recon_x

    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, condition)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- 3. TRAINING & VALIDATION ---
def train_model():
    print("--- Starting Training Process ---")
    num_nodes = 50
    num_features = num_nodes + 1
    seq_len = 24
    cond_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GenerativeCounterfactualVAE(num_features=num_features, seq_len=seq_len, cond_dim=cond_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataloader = get_dataloader(batch_size=32, num_nodes=num_nodes)
    
    epochs = 5
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            data = data.permute(0, 2, 1).to(device)
            condition = torch.tensor([[0.0, 1.0]] * data.size(0)).to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, condition)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"Epoch: {epoch+1} Average loss: {train_loss / len(dataloader.dataset):.4f}")
    print("Training Complete.\\n")
    return model, device

def apply_interventions(model, device, num_nodes=50):
    print("--- Applying Interventions (Counterfactual Generation) ---")
    model.eval()
    batch_size = 1
    latent_dim = 16
    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim).to(device)
        condition_vector = torch.tensor([[1.0, 2.5]]).to(device)
        counterfactual_output = model.decode(z, condition_vector)
        demand_output = counterfactual_output[:, :-1, :]
        final_tensor = demand_output.squeeze(0).permute(1, 0)
        print(f"Condition Vector Applied: {condition_vector.cpu().numpy()}")
        print(f"Final Output Tensor Shape (24 hours, N_Nodes): {final_tensor.shape}")
        return final_tensor

if __name__ == "__main__":
    trained_model, compute_device = train_model()
    cf_tensor = apply_interventions(trained_model, compute_device)
"""

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# EVolvAI: Generative Counterfactual Framework\n",
            "This notebook combines the PyTorch models, data loader, and training script into a single environment for easy execution on Google Colab GPUs. Because you are training on Colab, you don't have to worry about the RAM/resource limitations of a MacBook Air."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install torch numpy"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("EVolvAI_Training.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Created EVolvAI_Training.ipynb")
