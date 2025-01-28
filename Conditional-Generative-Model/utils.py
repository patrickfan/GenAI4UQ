import os
import numpy as np
import torch
from config import Config

def make_folder(folder):
    """Create the folder if it doesn't exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        print(f"Folder '{folder}' already exists.")

def save_data(savedir, data_dict):
    """Save data in numpy format."""
    try:
        os.makedirs(savedir, exist_ok=True)
        np.savez(
            os.path.join(savedir, 'training_data.npz'),
            **data_dict
        )
        print(f"Data saved in {savedir}")
    except Exception as e:
        print(f"Error saving data: {e}")
        raise

class ODEUtils:
    def __init__(self, config):
        self.config = config

    def cond_alpha(self, t, dt):
        return 1 - t + dt
        
    def cond_beta2(self, t, dt):
        return t + dt
        
    def b(self, t, dt):
        return -1.0 / self.cond_alpha(t,dt)
        
    def sigma_sq(self, t, dt):
        return 1.0 - 2 * self.b(t,dt) * self.cond_beta2(t,dt)

    def solve_ode(self, t_vec, zt, y_sample, x_sample, y0_test, time_steps, y_var_new):
        """Solves the ODE to simulate the diffusion process."""
        log_weight_likelihood = torch.sum(-0.5 * ((y0_test[:, None, :] - y_sample[None, :, :]) ** 2) / y_var_new, dim=2)

        dt = torch.abs(t_vec[1]-t_vec[0])
        
        for j in range(time_steps):
            t = t_vec[j + 1]
            dt = t_vec[j] - t_vec[j + 1]
            log_weight_gauss = torch.sum(-0.5 * ((zt[:, None, :] - self.cond_alpha(t,dt) * x_sample[None, :, :]) ** 2) / self.cond_beta2(t,dt), dim=2)
            score_gauss = -1.0 * (zt[:, None, :] - self.cond_alpha(t,dt) * x_sample[None, :, :]) / self.cond_beta2(t,dt)
            weight_temp_log = log_weight_gauss + log_weight_likelihood
            weight_temp_log = weight_temp_log - torch.amax(weight_temp_log, dim=1, keepdim=True)
            weight_temp = torch.exp(weight_temp_log)
            weight = weight_temp / torch.sum(weight_temp, dim=1, keepdims=True)
            score = torch.sum(score_gauss * weight[:, :, None], dim=1)
            zt = zt - (self.b(t,dt) * zt - 0.5 * self.sigma_sq(t,dt) * score) * dt
            
        return zt

    def Gen_labeled_data(self, x_sample, y_sample, zT, y0_train, y_var_new, it_size):

        """ Generate labeled data using a given sample and initial conditions."""

        x_sample = torch.tensor(x_sample).to(Config.DEVICE, dtype=torch.float32)
        y_sample = torch.tensor(y_sample).to(Config.DEVICE, dtype=torch.float32)
        y0_train = torch.tensor(y0_train).to(Config.DEVICE, dtype=torch.float32)
        zT = torch.tensor(zT).to(Config.DEVICE, dtype=torch.float32)
        data_size = y0_train.shape[0]
        xTrain = torch.zeros((data_size, x_sample.shape[1])).to(Config.DEVICE, dtype=torch.float32)
        it_n = int(data_size / it_size)
    
        T = 1.0 
        t_vec = torch.linspace(T, 0, Config.TIME_STEPS+1).to(Config.DEVICE)
        for jj in range(it_n):
            it_zt = zT[jj * it_size: (jj + 1) * it_size, :]
            it_y0 = y0_train[jj * it_size: (jj + 1) * it_size, :]
            x_temp = self.solve_ode(t_vec, it_zt, y_sample, x_sample, it_y0, Config.TIME_STEPS, y_var_new)
            xTrain[jj * it_size: (jj + 1) * it_size, :] = x_temp
            if jj % 10 == 0:
                print('Batch', jj, 'processed')
        return xTrain













        
