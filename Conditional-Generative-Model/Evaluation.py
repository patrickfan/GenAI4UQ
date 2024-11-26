import os
import numpy as np
import json
import torch
from scipy import stats

import ray
from ray import tune
from ray.tune import ExperimentAnalysis

from config import Config
from models import FN_Net
from utils import make_folder, save_data, ODEUtils
from data import DataModule, DataGenerator
from visualization import plot_loss, plot_real_vs_pred, generate_surface_plot, plot_distribution
from train_model import train_model

def load_best_trial_info():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    best_trial_path = os.path.join(current_dir, Config.SAVE_DIRECTORY, 'best_trial_info.json')
    
    if not os.path.exists(best_trial_path):
        raise ValueError(f"Best trial info file not found at {best_trial_path}")
    
    with open(best_trial_path, 'r') as f:
        best_trial_info = json.load(f)
    
    return best_trial_info


def load_best_model_and_evaluate(best_trial, best_checkpoint):
    if best_trial is None or best_checkpoint is None:
        print("Cannot evaluate model: Missing trial or checkpoint information")
        return None
        
    device = Config.DEVICE
    data_module = DataModule(Config())

    current_dir = os.path.abspath(os.path.dirname(__file__))
    savedir = os.path.join(current_dir, Config.SAVE_DIRECTORY)
    file_path = os.path.join(savedir, "training_data.npz") 
    sample_data = np.load(file_path)

    x_test_sample = sample_data['X_test']
    y_test_sample = sample_data['y_test']

    x_test_sample_mean = np.mean(x_test_sample, axis=0, keepdims=True)
    x_test_sample_std = np.std(x_test_sample, axis=0, keepdims=True)
    y_test_sample_mean = np.mean(y_test_sample, axis=0, keepdims=True)
    y_test_sample_std = np.std(y_test_sample, axis=0, keepdims=True)
    
    x_test_sample_normalized = (x_test_sample - x_test_sample_mean) / x_test_sample_std
    y_test_sample_normalized = (y_test_sample - y_test_sample_mean) / y_test_sample_std

    x_dim =sample_data['x_sample'].shape[1] # (N, dim)
    y_dim =sample_data['y_sample'].shape[1] # (N, dim)

    xTrain =sample_data['xTrain']
    yTrain =sample_data['yTrain']

    # Define the model architecture as per the best trial config
    model = FN_Net(
        input_dim= y_dim + x_dim,
        output_dim= x_dim,
        n_neurons=best_trial['n_neurons'],
        n_hidden_layers=best_trial['n_hidden_layers'],
        dropout_rate=best_trial['dropout_rate']
    ).to(device)
    
    # Load the best checkpoint
    if os.path.exists(best_checkpoint):
        try:
            checkpoint = torch.load(best_checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Successfully loaded checkpoint from {best_checkpoint}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    else:
        print(f"Warning: No checkpoint found at {best_checkpoint}")
        return None
    
    # Evaluate on training
    data_module = DataModule(Config())
    train_loader, val_loader = data_module.prepare_dataloaders(
        xTrain, yTrain, 
        best_trial["batch_size"]
    )
    plot_real_vs_pred(savedir,  val_loader, model)

    # Evaluate on test set
    model.eval()
    all_cases = {
        'observation_indices': [],
        'predictions': [],
        'true_values': []
        }

    num_cases =5
    Npath = 2000 

    ensemble_dir = os.path.join(savedir, "EnsembleForecasts")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    for case in range(num_cases):
        # Generate random observation index
        inx_obs = np.random.randint(0, y_test_sample_normalized.shape[0])
        obs_normalized_test = y_test_sample_normalized[inx_obs]
        
        print(f"\nGenerating case {case + 1}/{num_cases}")
        print(f"Observation index: {inx_obs}")
        print(f"Observation shape: {obs_normalized_test.shape}")
        
        # Generate ensemble predictions
        test = torch.tensor(obs_normalized_test*np.ones((Npath,y_dim))).to(device, dtype=torch.float32)
        test_py = torch.hstack((test,torch.randn(Npath,x_dim).to(device, dtype=torch.float32)))
        pred = model(test_py).detach().cpu().numpy()
        Ensemble_pred = pred*x_test_sample_std + x_test_sample_mean
        
        print(f"Ensemble prediction shape: {Ensemble_pred.shape}")
        
        # Save predictions
        case_filename = f"{ensemble_dir}/obs_{inx_obs}_{Npath}_ensemble_predictions_case_{case+1}"
        np.save(case_filename, Ensemble_pred)
        
        # Store case information
        all_cases['observation_indices'].append(inx_obs)
        all_cases['predictions'].append(Ensemble_pred)
        all_cases['true_values'].append(x_test_sample[inx_obs])
        
        # Generate plot for a random dimension
        plot_index = np.random.randint(0, x_dim)
        plot_distribution(ensemble_dir, x_test_sample, Ensemble_pred, plot_index, inx_obs, case_number=case+1)


def evaluate():
    try:
        ray.init(ignore_reinit_error=True)
        best_trial_info = load_best_trial_info()
        
        # Extract the configuration and checkpoint path
        best_config = best_trial_info['config']
        best_checkpoint = best_trial_info['checkpoint_path']
        
        print("Best configuration:", best_config)
        print("Final training loss:", best_trial_info['final_train_loss'])
        print("Final validation loss:", best_trial_info['final_val_loss'])
        
        # Evaluate the model
        load_best_model_and_evaluate(best_config, best_checkpoint)
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    evaluate()


    