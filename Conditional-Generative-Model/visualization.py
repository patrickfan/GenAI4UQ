import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy import stats
from config import Config
from utils import save_data, ODEUtils

def plot_loss(savedir, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 14})
    plt.plot(train_losses, label="Training Loss (MSE)")
    plt.plot(val_losses, label="Validation Loss (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Over Epochs")
    plt.grid(True)  
    plt.tight_layout()  
    plt.savefig(os.path.join(savedir, 'loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close()  


def plot_real_vs_pred(savedir, val_loader, model):

    device = Config.DEVICE
    model.eval()

    all_x_true = []
    all_y_pred = []

    model.eval()

    with torch.no_grad():
        for batch_X, batch_y in val_loader:

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_y)  
            y_pred = outputs.cpu().numpy()[:,-1]
            x_true = batch_X.cpu().numpy()[:,-1]
            all_y_pred.extend(y_pred)
            all_x_true.extend(x_true)

    y_pred = np.array(all_y_pred)
    x_true = np.array(all_x_true)

    val_r2 = r2_score(x_true, y_pred)

    print("Validation performance:")
    print(f"Validation R2 Score of the best model: {val_r2}")

    slope, intercept = np.polyfit(x_true, y_pred, 1)
    line = slope * np.array(x_true) + intercept

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.scatter(x_true, y_pred, s=0.7, label='Data points')
    plt.plot(x_true, line, color='red', label=f'Fit line (R²={val_r2:.2f})')
    plt.xlabel("Real Value")
    plt.ylabel("Predicted Value")
    plt.title("Validation results")
    plt.legend()
    plt.savefig(os.path.join(savedir, 'ValidationResults_R2_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()  

def plot_distribution(savedir, x_test_sample, ensemble_pred, plot_index, inx_obs, case_number=1):

    # Create a Gaussian Kernel Density Estimator (KDE): estiamte the probability density function (PDF)
    x_true = x_test_sample[inx_obs]
    kernel = stats.gaussian_kde(ensemble_pred[:, plot_index])

    # Define the grid for plotting the KDE
    N_interp = 200
    grid_plot = np.linspace(x_test_sample[:, plot_index].min(), x_test_sample[:, plot_index].max(), N_interp)
    grid_pdf = kernel(grid_plot)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the KDE as a filled area
    ax.fill_between(grid_plot, grid_pdf, alpha=0.6, color='green')
    ax.axvline(x_true[plot_index], color='r', linewidth=2, label='"True"')

    # Find the Maximum A Posteriori (MAP) estimate
    map_estimator = grid_plot[np.argmax(grid_pdf)]
    #ax.axvline(map_estimator, linewidth=2, color='blue', label='Maximum A Posterior')

    ax.set_xlabel(f"variable_{plot_index}")
    ax.set_ylim(0, None)
    ax.legend()
    plt.savefig(f"{savedir}/distribution_obs_{inx_obs}_case_{case_number}_dim_{plot_index}.png", dpi=300, bbox_inches='tight')
    plt.close()  



def generate_surface_plot(model, x_sample, y_sample, yTrain_mean, yTrain_std, xTrain_mean, xTrain_std, savedir, y_var_new, IT_SIZE):
    """Generate and plot the surface for the trained model."""
    DEVICE = Config.DEVICE
    Nsample_p = 5000
    SAMPLE_SIZE = len(y_sample)

    ode_utils = ODEUtils(Config())
   
    # Generate test points
    selected_row_indices = np.random.permutation(SAMPLE_SIZE)[:Nsample_p]
    y_test_p = y_sample[selected_row_indices]
    zT_p = np.random.randn(Nsample_p, 1)
    xTrain_p = ode_utils.Gen_labeled_data(
        x_sample, y_sample, zT_p, y_test_p, y_var_new, IT_SIZE
    )
    xTrain_p = xTrain_p.detach().cpu().numpy()
    
    # Create grid for surface plot
    grid_size = 50
    x_grid = np.linspace(np.min(y_test_p), np.max(y_test_p), grid_size)
    y_grid = np.linspace(np.min(zT_p), np.max(zT_p), grid_size)
    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
    
    # Generate surface values
    f_hat = np.zeros((grid_size, grid_size))
    model.eval()
    with torch.no_grad():
        for j in range(grid_size):
            for i in range(grid_size):
                xy_pair = torch.tensor(
                    np.hstack((xx_grid[j,i], yy_grid[j,i]))
                ).to(DEVICE, dtype=torch.float32)
                normalized_input = (xy_pair - yTrain_mean) / yTrain_std
                output = model(normalized_input)
                f_hat[j,i] = (output * xTrain_std + xTrain_mean).cpu().numpy()
    
    # Create and save the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xx_grid, yy_grid, f_hat, cmap='viridis')
    ax.set_xlabel('Y Test')
    ax.set_ylabel('Z')
    ax.set_zlabel('Predicted X')
    plt.colorbar(surf)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'mesh.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the generated data
    save_data(os.path.join(savedir, 'generated_data.npz'), {
        'y_test_p': y_test_p,
        'zT_p': zT_p,
        'xTrain_p': xTrain_p,
        'xx_grid': xx_grid,
        'yy_grid': yy_grid,
        'f_hat': f_hat
    })
    
    return y_test_p, zT_p, xTrain_p, xx_grid, yy_grid, f_hat




    