import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import matplotlib.pyplot as plt
import os

class PressureDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class SmoothingLayer(nn.Module):
    """Learnable smoothing layer to reduce artifacts."""
    def __init__(self, channels):
        super().__init__()
        kernel_size = 5
        sigma = 1.5
        self.smooth = nn.Conv2d(channels, channels, kernel_size, 
                               padding=kernel_size//2, groups=channels, bias=False)
        
        # Initialize with Gaussian kernel
        with torch.no_grad():
            kernel = self._create_gaussian_kernel(kernel_size, sigma)
            kernel = kernel.repeat(channels, 1, 1, 1)
            self.smooth.weight.data = kernel
    
    def _create_gaussian_kernel(self, kernel_size, sigma):
        center = kernel_size // 2
        x, y = np.mgrid[0:kernel_size, 0:kernel_size]
        gaussian = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()
        return torch.FloatTensor(gaussian).unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        return self.smooth(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(channels)  # Instance norm for better stability
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(channels)
        self.smooth = SmoothingLayer(channels)
        self.relu = nn.LeakyReLU(0.2)
        
        # SE block for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE attention
        se_weight = self.se(out)
        out = out * se_weight
        
        out = self.smooth(out + residual)
        out = self.relu(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        self.residual = ResidualBlock(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.residual(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        self.residual = ResidualBlock(out_channels)
        self.smooth = SmoothingLayer(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.residual(x)
        x = self.smooth(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        # Input: 64x128
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(1, 32),      # 32x32x64
            EncoderBlock(32, 64),     # 64x16x32
            EncoderBlock(64, 128),    # 128x8x16
            EncoderBlock(128, 256),   # 256x4x8
            EncoderBlock(256, 512)    # 512x2x4
        ])
        
        self.flatten = nn.Flatten()
        self.encoder_linear = nn.Sequential(
            nn.Linear(512 * 2 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        # Skip connections storage
        skips = []
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
        
        x = self.flatten(x)
        x = self.encoder_linear(x)
        
        return x, skips

class Decoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        
        # Linear projection
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512 * 2 * 4)
        )
        
        # Decoder blocks with skip connections
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(512, 256),
            DecoderBlock(512, 128),  # 512 due to skip connection
            DecoderBlock(256, 64),   # 256 due to skip connection
            DecoderBlock(128, 32),   # 128 due to skip connection
            DecoderBlock(64, 16)     # 64 due to skip connection
        ])
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z, skips):
        # Project and reshape
        x = self.decoder_linear(z)
        x = x.view(-1, 512, 2, 4)
        
        # Apply decoder blocks with skip connections
        for i, block in enumerate(self.decoder_blocks):
            if i > 0:  # Skip connection for all but first block
                x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = block(x)
        
        # Final convolution
        x = self.final_conv(x)
        return x.squeeze(1)

class ImprovedAutoencoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        z, skips = self.encoder(x)
        reconstruction = self.decoder(z, skips)
        return reconstruction

def generate_ensemble_predictions(test_data, latent_ensemble, model, device, scaler):
    model.eval()
    predictions = []
    
    # Convert test data to torch tensor
    test_data = torch.FloatTensor(test_data).to(device)    
    latent_codes = torch.FloatTensor(latent_ensemble).to(device)
    
    # Number of samples
    num_samples = test_data.shape[0]
    batch_size = 10
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            test_batch = test_data[start_idx:end_idx]
            latent_batch = latent_codes[start_idx:end_idx]
            
            # Get encoder skips for the test batch
            _, encoder_skips = model.encoder(test_batch)
            
            # For each latent code in the ensemble, generate a prediction using the same encoder skips
            batch_predictions = []
            for i in range(latent_batch.shape[0]):
                reconstruction = model.decoder(latent_batch[i:i+1], encoder_skips)
                batch_predictions.append(reconstruction)
            
            # Stack batch predictions
            batch_predictions = torch.cat(batch_predictions, dim=0)
            predictions.append(batch_predictions.cpu().numpy())
    
    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    
    # Inverse transform if scaler is provided
    if scaler is not None:
        orig_shape = predictions.shape
        predictions = predictions.reshape(-1, predictions.shape[-1])
        predictions = scaler.inverse_transform(predictions)
        predictions = predictions.reshape(orig_shape)
    
    return predictions


def main():
    # Define cases
    cases = [8, 12, 49]

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the saved model checkpoint
    checkpoint_path = 'Results/best_checkpoint_pres.pth'

    # Load the data
    print("Loading data...")
    data_ori = torch.load("Dataset/dP_train_u.pt").numpy()[:, :64, :128, -1]
    data = data_ori.reshape(-1, data_ori.shape[1], data_ori.shape[2])
    print(f"Data shape: {data.shape}")

    train_size = int(0.9 * len(data_ori))

    # Normalize the data
    print("Normalizing data...")
    original_shape = data.shape
    X = data.reshape((-1, data.shape[-1]))
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_norm = scaler.fit_transform(X)

    # Reshape to (N, 64, 128)
    y = X_norm.reshape(-1, 64, 128)

    # Initialize model and load weights
    model = ImprovedAutoencoder(latent_dim=20).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")

    # Load scaler
    scaler = load('Results/scaler.joblib')
    print("Scaler loaded successfully")

    for i, case in enumerate(cases):  # Loop through each case
        print(f"Processing case {case}...")

        # Load ensemble latent codes
        latent_ensemble_path = f'Results/EnsembleForecasts/obs_{case}_2000_ensemble_predictions_case_{case + 1}.npy'
        latent_ensemble = np.load(latent_ensemble_path)
        print(f"Loaded ensemble latent codes shape for case {case}: {latent_ensemble.shape}")

        # Prepare test data
        test_data = y[train_size:][case]
        test_data = np.expand_dims(test_data, axis=0)  # Add batch dimension

        # Generate ensemble predictions
        print(f"Generating ensemble predictions for case {case}...")
        ensemble_predictions = generate_ensemble_predictions(
            test_data,
            latent_ensemble,
            model,
            device,
            scaler
        )

        # Save predictions
        predictions_path = f'Results/Pres_ensemble_predictions_case_{i + 1}.npy'
        np.save(predictions_path, ensemble_predictions)
        print(f"Ensemble predictions for case {case} saved at {predictions_path}.")
        print(f"Ensemble predictions shape: {ensemble_predictions.shape}")

        # Post-processing and plotting
        y_ori = data_ori[train_size:][case]
        y_pred = np.mean(ensemble_predictions, axis=0)
        y_reconstruction = np.load("Results/reconstructed_data.npy")
        y_recons = y_reconstruction[train_size:][case]

        # Plot the results
        print(f"Plotting results for case {case}...")
        plot_samples(y_ori, y_recons, y_pred, case)

    print("Processing completed for all cases.")



def plot_samples(y_ori, y_recons, y_pred, case):
    
    img = y_ori
    reconst = y_recons
    y_pred = y_pred
        
    # Calculate global min and max for consistent colorbar scaling
    vmin = img.min() #min(img.min(), reconst.min())
    vmax = img.max() #max(img.max(), reconst.max())
        
    # Create figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
    # Plot original image
    img1 = axes[0].imshow(img, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Pressure")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")
    fig.colorbar(img1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
        
    # Plot reconstructed image with same scale
    img2 = axes[1].imshow(reconst, cmap='viridis', aspect='auto',vmin=vmin, vmax=vmax)
    axes[1].set_title("AE Reconstructed Pressure")
    axes[1].set_xlabel("X-axis")
    fig.colorbar(img2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        
    # Plot error image with symmetric scale around zero  coolwarm
    img3 = axes[2].imshow(y_pred, cmap='viridis', aspect='auto',vmin=vmin, vmax=vmax)  
    axes[2].set_title("Predicted Pressure")
    axes[2].set_xlabel("X-axis")
    fig.colorbar(img3, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
        
    fig.suptitle(f"Original vs. Reconstructed vs. Predicted", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"Results/Final_pres_pred_{case}.png", dpi=300, bbox_inches='tight')
    plt.close()  



if __name__ == "__main__":
    main()