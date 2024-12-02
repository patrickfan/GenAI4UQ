import os
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.air import session
from ray.tune.stopper import ExperimentPlateauStopper
import numpy as np

from models import FN_Net
from utils import make_folder, save_data, ODEUtils
from data import DataModule
from config import Config
from visualization import plot_real_vs_pred

def train_model(config):

    current_dir = os.path.abspath(os.path.dirname(__file__))
    savedir = os.path.join(current_dir, Config.SAVE_DIRECTORY)
    make_folder(savedir)
    
    # Initialize components
    data_module = DataModule(Config())
    ode_utils = ODEUtils(Config())
    
    # Generate data
    x_sample, y_sample, X_test, y_test = data_module.data_generator.load_data()

    # Calculate mean and standard deviation for all datasets
    x_sample_mean = np.mean(x_sample, axis=0)
    x_sample_std = np.std(x_sample, axis=0)
    y_sample_mean = np.mean(y_sample, axis=0)
    y_sample_std = np.std(y_sample, axis=0)

    # Normalize all datasets
    x_sample_normalized = (x_sample - x_sample_mean) / x_sample_std
    y_sample_normalized = (y_sample - y_sample_mean) / y_sample_std

    # print ("Loaded training sample shape: ", x_sample.shape, y_sample.shape)
    # print ("Loaded test sample shape: ", X_test.shape, y_test.shape)
    # print("First few rows of loaded xdata:")
    # print(x_sample_normalized[:5])
    # print("First few rows of loaded ydata:")
    # print(y_sample_normalized[:5])


    TRAIN_SIZE = max(Config.TRAIN_SIZE, int(x_sample.shape[0] * 1))

    selected_row_indices = np.random.choice(y_sample.shape[0], size=TRAIN_SIZE, replace=True)
    y0_train = y_sample_normalized[selected_row_indices]
    zT = np.random.randn(TRAIN_SIZE, x_sample.shape[1])
    
    # Generate labeled data
    xTrain = ode_utils.Gen_labeled_data( 
        x_sample_normalized, y_sample_normalized, zT, y0_train, Config.Y_VAR_NEW, Config.IT_SIZE
        )

    # Prepare data for training
    yTrain = torch.tensor(np.hstack((y0_train, zT)), dtype=torch.float32).to(Config.DEVICE)
    xTrain = torch.tensor(xTrain, dtype=torch.float32).to(Config.DEVICE)
    # xTrain = xTrain.clone().detach()
    
    # Create data loaders with normalized data
    train_loader, val_loader = data_module.prepare_dataloaders(
        xTrain, yTrain, 
        config["batch_size"]
    )

    # Save data
    all_data = {
        'x_sample': x_sample,
        'y_sample': y_sample,
        'y0_train': y0_train,
        'zT': zT,
        'xTrain': xTrain.cpu().numpy(),
        'yTrain': yTrain.cpu().numpy(),
        'X_test': X_test,
        'y_test': y_test,
    }
    save_data(savedir, all_data)
    
    # Verify the saved data
    data_path = os.path.join(savedir, 'training_data.npz')
    if os.path.exists(data_path):
        print("Data saved successfully!")
        data = np.load(data_path)
        print("Saved data contains:", list(data.keys()))
        print("First item shape:", data['xTrain'].shape)
        print("Second item shape:", data['yTrain'].shape)

    else:
        print("Data not saved successfully. Checking for errors...")
        
        # Check if the savedir folder exists
        if os.path.exists(savedir):
            print(f"Folder '{savedir}' exists.")
        else:
            print(f"Folder '{savedir}' does not exist.")
            
        # Check write permissions
        try:
            test_file = os.path.join(savedir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('Test file')
            os.remove(test_file)
            print("Write permissions are granted.")
        except (OSError, IOError) as e:
            print(f"Error writing to '{savedir}': {e}")
            
        # Check if the file was created but empty
        if os.path.exists(data_path) and os.path.getsize(data_path) == 0:
            print("File was created but is empty.")
        
        print("Please check the above information and try to resolve the issue.")
    
    
    # Model initialization
    current_dir = os.path.abspath(os.path.dirname(__file__))
    checkpoint_dir = os.path.join(current_dir, "checkpoints")
    make_folder(checkpoint_dir)

    device = Config.DEVICE
    x_dim =x_sample.shape[1] # (N, dim)
    y_dim =y_sample.shape[1] # (N, dim)
    model = FN_Net(
        input_dim= y_dim + x_dim,
        output_dim= x_dim,
        n_neurons=config["n_neurons"],
        n_hidden_layers=config["n_hidden_layers"],
        dropout_rate=config["dropout_rate"]
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    best_val_loss = float('inf')
    best_checkpoint_path = None
    
    # Initialize lists to store loss history
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_checkpoint_path = None
    min_delta = 1e-4
    patience = 10

    # Add variables to track overfitting
    best_generalization_gap = float('inf')
    min_epoch_for_stopping = 5  # Minimum epochs before enabling early stopping

    # Training loop
    for epoch in range(config["max_epochs"]):
        pass
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_y)
            loss = criterion(outputs, batch_X)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_y)
                val_loss += criterion(outputs, batch_X).item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        generalization_gap = avg_val_loss - avg_train_loss
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Early stopping logic with multiple conditions
        should_save_checkpoint = False
        
        # Condition 1: Better validation loss
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            should_save_checkpoint = True
            patience_counter = 0
        
        # Condition 2: Better generalization (less overfitting)
        if generalization_gap < best_generalization_gap:
            best_generalization_gap = generalization_gap
            should_save_checkpoint = True
            patience_counter = 0
        
        # If we should save a checkpoint
        if should_save_checkpoint:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history,
            }
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
            torch.save(checkpoint, checkpoint_path)
            best_checkpoint_path = checkpoint_path
        else:
            patience_counter += 1

        
        # Report metrics to Ray Tune
        session.report(
            {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "epoch": epoch,
            "checkpoint_path": best_checkpoint_path,
            "patience_counter": patience_counter
        })

        # Check if early stopping criteria is met
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        if len(train_loss_history) >= 10:  # Need at least 3 epochs to check trend
                recent_train_trend = train_loss_history[-1] - train_loss_history[-10]
                recent_val_trend = val_loss_history[-1] - val_loss_history[-10]
                
                # If training loss is decreasing but validation loss is increasing
                if recent_train_trend < -min_delta and recent_val_trend > min_delta:
                    print(f"Early stopping triggered after {epoch + 1} epochs due to overfitting")
                    break
                


    

     




