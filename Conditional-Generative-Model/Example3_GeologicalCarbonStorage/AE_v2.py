import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
# import wandb
from torch.cuda.amp import autocast, GradScaler

def set_seeds(seed=42):
    """Set seeds for reproducibility with additional security."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # for reproducibility

class PressureDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = torch.FloatTensor(data)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

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

    def get_loss(self, x, reconstruction):
        # Combined loss
        mse_loss = F.mse_loss(reconstruction, x)
        
        # Total variation loss for smoothness
        def total_variation_loss(img):
            # Add channel dimension if it's missing
            if len(img.shape) == 3:
                img = img.unsqueeze(1)  # Add channel dimension
                
            # Now img should be [batch, channel, height, width]
            tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
            tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
            return (tv_h + tv_w) / 2.0
        
        tv_loss = total_variation_loss(reconstruction)
        
        # Perceptual loss using feature differences
        def feature_loss(x, y):
            return F.mse_loss(
                F.avg_pool2d(x, kernel_size=4), 
                F.avg_pool2d(y, kernel_size=4)
            )
        
        perceptual_loss = feature_loss(reconstruction, x)
        
        # Combine losses
        total_loss = mse_loss + 0.01 * tv_loss + 0.01 * perceptual_loss
        
        return total_loss

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self.counter = 0

def train_autoencoder(model, train_loader, test_loader, device, num_epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()  # for mixed precision training
    early_stopping = EarlyStopping(patience=10)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                reconstruction = model(data)
                loss = model.get_loss(data, reconstruction)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                reconstruction = model(data)
                val_loss += model.get_loss(data, reconstruction).item()
        
        val_loss /= len(test_loader)
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}')
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'New best validation loss: {best_val_loss:.4f}. Saving checkpoint...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'Results/best_checkpoint_pres.pth')
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(early_stopping.best_state)
            break

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
    return best_val_loss
    
    # wandb.finish()
def generate_latent_and_reconstructions(model, dataset, device, batch_size=32, scaler=None):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    latent_codes = []
    reconstructed_data = []
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            
            # Get latent representation and reconstruction
            latent, skips = model.encoder(data)
            reconstruction = model.decoder(latent, skips)
            
            latent_codes.append(latent.cpu().numpy())
            reconstructed_data.append(reconstruction.cpu().numpy())
    
    # Concatenate all batches
    X_latent = np.concatenate(latent_codes, axis=0)
    reconstructed = np.concatenate(reconstructed_data, axis=0)
    
    # Reshape and inverse transform if scaler is provided
    if scaler is not None:
        reconstructed = reconstructed.reshape(-1, reconstructed.shape[-1])
        reconstructed = scaler.inverse_transform(reconstructed)
        reconstructed = reconstructed.reshape(4500, 64, 128)
    
    return X_latent, reconstructed
    
def main():
    # Set seeds and initialize
    set_seeds()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs("Results", exist_ok=True)
    
    # Load and preprocess data
    # Assuming your data is in the shape (N, 96, 200) where N is the number of samples
    # Dataset location:  https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo
    # https://github.com/gegewen/ufno?tab=readme-ov-file
    # https://www-sciencedirect-com.ornl.idm.oclc.org/science/article/pii/S0309170822000562#d1e8497
    print("Loading data...")
    data_ori = torch.load("Dataset/dP_train_u.pt").numpy()[:,:64,:128,-1]
    data = data_ori.reshape(-1, data_ori.shape[1], data_ori.shape[2])
    print(f"Data shape: {data.shape}")
    
    # Reshape and normalize
    print("Normalizing data...")
    original_shape = data.shape
    X = data.reshape((-1, data.shape[-1]))
    scaler = MinMaxScaler(feature_range=(0,1))
    X_norm = scaler.fit_transform(X)
    dump(scaler, 'Results/scaler.joblib')
    
    # Reshape to (N, 64, 128) 
    y = X_norm.reshape(-1, 64, 128)
    
    train_data = y
    test_data = y  

    # Create datasets and dataloaders
    print("Creating dataloaders...")
    train_dataset = PressureDataset(train_data)
    test_dataset = PressureDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


   # Initialize model
    print("Initializing model...")
    latent_dim = 20
    model = ImprovedAutoencoder(latent_dim).to(device)
    
    # Train the model
    print("Starting training...")
    train_autoencoder(model, train_loader, test_loader, device)
    
    # Generate latent representations and reconstructions
    print("Generating latent representations and reconstructions...")
    X_latent, reconstructed = generate_latent_and_reconstructions(
        model,
        PressureDataset(y),
        device,
        scaler=scaler
    )
    
    # Save results
    print("Saving results...")
    np.save('Dataset/Co2_Pressure/pre_latent.npy', X_latent)
    np.save('Results/reconstructed_data.npy', reconstructed)

    # Extract the values
    indices = np.linspace(0, data_ori.shape[1] - 1, 10, dtype=int)
    extracted = data_ori[:, indices, 0, ].reshape(-1,10)
    np.save('Dataset/Co2_Pressure/pre_observation.npy', extracted)

    print("Training complete!")
    print(f"Latent space shape: {X_latent.shape}")
    print(f"Reconstructed data shape: {reconstructed.shape}")
    print("Extracted shape:", extracted.shape)

    specific_indices = [4058,4062,4099]
    num_samples = len(specific_indices)
    plot_samples(data_ori, reconstructed, num_samples=num_samples, sample_indices=specific_indices)

    plot_summary_statistics(data_ori, reconstructed)


def plot_samples(data_ori, reconstructed, num_samples=5, sample_indices=None, base_dir='Reconstruction_figures'):
    """
    Plot original, reconstructed, and error images for multiple samples with consistent colorbars.
    
    Parameters:
    -----------
    data_ori : numpy array
        Original data
    reconstructed : numpy array
        Reconstructed data
    num_samples : int
        Number of samples to plot (default: 10)
    sample_indices : list or None
        Specific indices to plot. If None, random samples will be chosen
    base_dir : str
        Base directory for saving figures (default: 'pres_figures')
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    
    # Create directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    else:
        print(f"Directory {base_dir} already exists")
    
    # Reshape reconstructed data
    pres_true = data_ori
    recons_20 = reconstructed
    
    # Generate sample indices if not provided
    if sample_indices is None:
        sample_indices = np.random.choice(4500, num_samples, replace=False)
    
    # Create a figure for each sample
    for idx, sample_idx in enumerate(sample_indices):
        # Get the images
        img = pres_true[sample_idx, :, :]
        reconst = recons_20[sample_idx, :, :]
        error = img - reconst
        
        # Calculate global min and max for consistent colorbar scaling
        vmin = img.min() #min(img.min(), reconst.min())
        vmax = img.max() #max(img.max(), reconst.max())
        
        # Calculate error limits symmetrically around zero
        error_limit = max(abs(error.min()), abs(error.max()))
        
        # Create figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        img1 = axes[0].imshow(img, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title("Original Image")
        axes[0].set_xlabel("X-axis")
        axes[0].set_ylabel("Y-axis")
        fig.colorbar(img1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
        
        # Plot reconstructed image with same scale
        img2 = axes[1].imshow(reconst, cmap='viridis', aspect='auto',vmin=vmin, vmax=vmax)
        axes[1].set_title("Reconstructed Image")
        axes[1].set_xlabel("X-axis")
        fig.colorbar(img2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        
        # Plot error image with symmetric scale around zero  coolwarm
        img3 = axes[2].imshow(error, cmap='RdBu', aspect='auto') #,vmin=-error_limit, vmax=error_limit)  
        axes[2].set_title("Error (Original - Reconstructed)")
        axes[2].set_xlabel("X-axis")
        fig.colorbar(img3, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
        
        fig.suptitle(f"Sample {sample_idx}: Original vs. Reconstructed vs. Error", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(base_dir, f'mapping_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  
        
    print(f"Saved {num_samples} plots in {base_dir}")

def plot_summary_statistics(data_ori, reconstructed, base_dir='Reconstruction_figures'):
    """
    Create summary statistics plots for the reconstruction error.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    
    # Reshape data if needed
    pres_true = data_ori
    recons_20 = reconstructed
    
    # Calculate errors for all samples
    errors = np.abs(pres_true[:, :, :] - recons_20[:, :, :])
    mean_error = np.mean(errors, axis=0)
    std_error = np.std(errors, axis=0)
    
    # Create summary plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot mean error
    img1 = axes[0].imshow(mean_error, cmap='coolwarm', aspect='auto')
    axes[0].set_title("Mean Reconstruction Error")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")
    fig.colorbar(img1, ax=axes[0])
    
    # Plot error standard deviation
    img2 = axes[1].imshow(std_error, cmap='coolwarm', aspect='auto')
    axes[1].set_title("Standard Deviation of Error")
    axes[1].set_xlabel("X-axis")
    fig.colorbar(img2, ax=axes[1])
    
    plt.tight_layout()
    
    # Save summary plot
    save_path = os.path.join(base_dir, 'error_statistics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error statistics plot in {base_dir}")

if __name__ == "__main__":
    main()





