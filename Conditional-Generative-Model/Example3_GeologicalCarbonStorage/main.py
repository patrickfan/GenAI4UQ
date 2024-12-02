import os
import numpy as np
import json
import torch
import torch.nn as nn
from torch import optim
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session
from ray.train import ScalingConfig
from sklearn.model_selection import train_test_split

from config import Config
from models import FN_Net
from utils import make_folder, save_data, ODEUtils
from data import DataModule, DataGenerator
from visualization import plot_loss
from train_model import train_model
from Evaluation import evaluate

def main():

    # Initialize Ray with dashboard
    print (f"Using device: {Config.DEVICE}")
    ray.init(dashboard_port=8265, num_gpus=1)
    
    torch.manual_seed(Config.RANDOM_SEED['torch'])
    np.random.seed(Config.RANDOM_SEED['numpy'])
    
    # Create absolute path for storage
    current_dir = os.path.abspath(os.path.dirname(__file__))
    storage_path = os.path.join(current_dir, "ray_results")

    savedir = os.path.join(current_dir, Config.SAVE_DIRECTORY)
    make_folder(savedir)
    
    # Define the early stopping scheduler and search algorithm
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=Config.HP_SEARCH_CONFIG["max_epochs"],
        grace_period=10,
        reduction_factor=2
    )
    
    # Define the search algorithm
    search_alg = HyperOptSearch(
        metric="val_loss",
        mode="min",
        n_initial_points=5
    )

    # Run the hyperparameter optimization
    run_params = {
        "name": "neural_network_tuning",
        "keep_checkpoints_num": 1,
        "checkpoint_score_attr": "min-val_loss",
        "verbose": 1 
    }
    
    if Config.RESOURCES_PER_TRIAL is not None:
        run_params["resources_per_trial"] = Config.RESOURCES_PER_TRIAL

    analysis = tune.run(
        train_model,
        config=Config.HP_SEARCH_CONFIG,
        num_samples=Config.NUM_SAMPLES,
        scheduler=scheduler,
        search_alg=search_alg,
        storage_path=storage_path,
        **run_params
    )
    
    # Get the best trial
    best_trial = analysis.get_best_trial("val_loss", "min", "all")
    if best_trial is None:
        print("No successful trials found")
        return None, None
        
    print("Best Trial Details:")
    print("- Configuration:", best_trial.config)
    print("Best trial config:", best_trial.config)
    print("Best trial final training loss:", best_trial.last_result["train_loss"])
    print("Best trial final validation loss:", best_trial.last_result["val_loss"])
    
    # Plot training history
    train_loss_history = best_trial.last_result.get("train_loss_history")
    val_loss_history = best_trial.last_result.get("val_loss_history")
    
    if train_loss_history and val_loss_history:
        plot_loss(savedir , train_loss_history, val_loss_history)
        print("Loss history plot has been saved as 'loss_history.png'")
    else:
        print("Training/Validation loss history is not available for plotting.")


    # Get the best checkpoint path from the last result
    best_checkpoint = best_trial.last_result.get("checkpoint_path")
    if best_checkpoint is None:
        print("No checkpoint found for the best trial")
        return best_trial, None

    # Save the best trial information
    best_trial_info = {
        'config': best_trial.config,
        'checkpoint_path': best_trial.last_result.get('checkpoint_path'),
        'final_train_loss': best_trial.last_result['train_loss'],
        'final_val_loss': best_trial.last_result['val_loss'],
        'train_loss_history': best_trial.last_result.get('train_loss_history'),
        'val_loss_history': best_trial.last_result.get('val_loss_history')
    }
    
    # Save to a dedicated file
    best_trial_path = os.path.join(savedir, 'best_trial_info.json')
    with open(best_trial_path, 'w') as f:
        json.dump(best_trial_info, f, indent=4)
    
    print(f"Best trial information saved to: {best_trial_path}")

    return best_trial, best_checkpoint

if __name__ == "__main__":

    best_trial, best_checkpoint = main()
    evaluate()



