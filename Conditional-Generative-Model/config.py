import torch
from ray import tune

class Config:

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        RESOURCES_PER_TRIAL = {"gpu": 1} 
    else:
        RESOURCES_PER_TRIAL = {}

    # Generate labled data
    TIME_STEPS = 100  # time step for reverse ODE
    IT_SIZE = 1000    # iteration size of generating labeled data
    Y_VAR_NEW = 1
    SAMPLE_SIZE = 10000   # synthetic case sample size
    TRAIN_SIZE = 20000    # size of reverse ODE generated labeled data
   
    SAVE_DIRECTORY = 'Results'
    RANDOM_SEED = {
        'torch': 12345678,
        'numpy': 12312414
    }

    # Ray Tune hyperparameter search configuration
    NUM_SAMPLES = 10  #  how many trials Ray Tune will run
    HP_SEARCH_CONFIG = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "n_neurons": tune.choice([32, 64, 128]),
        "n_hidden_layers": tune.choice([1, 2]),
        "dropout_rate": tune.uniform(0.01, 0.3),
        "batch_size": tune.choice([32, 64]),
        "max_epochs": 1000
    }

############################################################################################
######## Parameters that require to modify ##########

    # [REQUIRED] Choose data source: syntetic case, y=x^2 
    USE_SYNTHETIC_DATA = True  # Set False to use custom dataset

    # [REQUIRED] Data paths (required if USE_SYNTHETIC_DATA = False) Put your data into the Dataset directory
    X_SAMPLE_PATH = 'Dataset/ELM/InputPara.npy'   # Model parameters (X)
    Y_SAMPLE_PATH = 'Dataset/ELM/Observations.npy'     # Observations (Y)




