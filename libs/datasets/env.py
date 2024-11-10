import unittest
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from torch import multiprocessing
import os

import numpy as np
import random
from collections import defaultdict
import time

import os
import random
import time
import traceback
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from enum import Enum
from scipy.stats import truncnorm 


import atexit
import click
import datetime
import os
import requests
import sys
import yaml
import json
import openai
from functools import partial
from collections import deque
from scipy.optimize import minimize
import math
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def dict_to_array(constants_dict):
    """
    Convert a dictionary of constants to an array.

    Parameters:
    constants_dict (dict): Dictionary of constants.

    Returns:
    list: List of constant values.
    """
    # Ensure consistent ordering of keys
    keys = sorted(constants_dict.keys())
    return [constants_dict[key] for key in keys]

def array_to_dict(constants_array, template_dict):
    """
    Convert an array of constants back to a dictionary.

    Parameters:
    constants_array (list): List of constant values.
    template_dict (dict): A template dictionary to get the keys.

    Returns:
    dict: Dictionary of constants.
    """
    keys = sorted(template_dict.keys())
    return {key: value for key, value in zip(keys, constants_array)}

def generate_bounds(param_dict):
    bounds = []
    keys = sorted(param_dict.keys())
    for key in keys:
        value = param_dict[key]
        # Determine the order of magnitude
        order_of_magnitude = 10 ** (int(math.log10(abs(value))) + 1)

        # # Lower bound is an order of magnitude less, unless the value is 0
        # lower_bound = max(value - order_of_magnitude, 0) if value != 0 else 0

        # # Upper bound is an order of magnitude more
        # upper_bound = value + order_of_magnitude

        # Append the tuple of bounds to the list
        # bounds.append((lower_bound, upper_bound))
        bounds.append((-order_of_magnitude, order_of_magnitude))

    return bounds

# True env
probabilistic = False
device = "cuda:0"

def get_model_parameters(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if '.' not in name:
            # Convert each parameter tensor to a Python number for readability
            # This assumes the parameter tensors contain only a single value (as in your model)
            param_dict[name] = param.item()
    return param_dict

class DatasetEnv:
    def __init__(self):
        pass

    def reset(self, num_patients=1):
        pass
    
    def evaluate_simulator_code_wrapper(self, StateDifferential, train_data, val_data, test_data, config={}, logger=None, env_name=''):
        if config.run.optimizer == 'pytorch':
            train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss = self.evaluate_simulator_code_using_pytorch(StateDifferential, train_data, val_data, test_data, config=config, logger=logger, env_name=env_name)
        elif 'evotorch' in config.run.optimizer:
            train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss = self.evaluate_simulator_code_using_pytorch_with_neuroevolution(StateDifferential, train_data, val_data, test_data, config=config, logger=logger)
        if env_name == 'Dataset-3DLV':
            loss_per_dim_dict = {'prey_population': loss_per_dim[0], 'intermediate_population': loss_per_dim[1], 'top_predators_population': loss_per_dim[2]}
        elif env_name == 'Dataset-HL':
            loss_per_dim_dict = {'hare_population': loss_per_dim[0], 'lynx_population': loss_per_dim[1]}
        return train_loss, val_loss, optimized_parameters, loss_per_dim_dict, test_loss
    
    def evaluate_simulator_code_using_pytorch(self, StateDifferential, train_data, val_data, test_data, config={}, logger=None, env_name=''):
        import torch
        import numpy as np
        device = "cuda:0"
        config.run.pytorch_as_optimizer.batch_size = 1

        # Wrap in try
        f_model = StateDifferential()
        f_model.to(device)

        f_model.train()
        states_train, actions_train = train_data
        if actions_train is not None:
            actions_train = torch.tensor(actions_train, dtype=torch.float32, device=device)
        states_train = torch.tensor(states_train, dtype=torch.float32, device=device)

        states_val, actions_val = val_data
        if actions_val is not None:
            actions_val = torch.tensor(actions_val, dtype=torch.float32, device=device)
        states_val = torch.tensor(states_val, dtype=torch.float32, device=device)

        MSE = torch.nn.MSELoss()
        optimizer = optim.Adam(f_model.parameters(), lr=config.run.pytorch_as_optimizer.learning_rate, weight_decay=config.run.pytorch_as_optimizer.weight_decay)
        # clip_grad_norm = config.run.clip_grad_norm if config.run.clip_grad_norm > 0 else None

        def train(model, states_train_batch_i, actions_train_batch_i):
            optimizer.zero_grad(True)
            pred_states = []
            pred_state = states_train_batch_i[:,0]
            for t in range(states_train_batch_i.shape[1]):
                pred_states.append(pred_state)
                if env_name == 'Dataset-3DLV':
                    prey_population, intermediate_population, top_predators_population = states_train_batch_i[:,t,0], states_train_batch_i[:,t,1], states_train_batch_i[:,t,2]
                    dx_dt = model(prey_population, intermediate_population, top_predators_population)
                elif env_name == 'Dataset-HL':
                    hare, lynx, time = states_train_batch_i[:,t,0], states_train_batch_i[:,t,1], actions_train_batch_i[:,t,0]
                    dx_dt = model(hare, lynx, time)
                dx_dt = torch.stack(dx_dt, dim=-1)
                pred_state = states_train_batch_i[:,t] + dx_dt
                # pred_state[pred_state<=0] = 0
            pred_states = torch.stack(pred_states, dim=1)
            loss = MSE(pred_states, states_train_batch_i)
            loss.backward()
            # if clip_grad_norm:
            #     torch.nn.utils.clip_grad_norm_(f_model.parameters(), clip_grad_norm)
            optimizer.step()
            return loss.item()
        
        # train_opt = torch.compile(train)
        # train_opt = torch.compile(train)
        train_opt = train

        def compute_eval_loss(model, dataset):
            states, actions = dataset
            model.eval()
            with torch.no_grad():
                pred_states = []
                # pred_sates_per_dim_per_bb = []
                pred_state = states[:,0]
                for t in range(states.shape[1]):
                    pred_states.append(pred_state)
                    if env_name == 'Dataset-3DLV':
                        prey_population, intermediate_population, top_predators_population = states[:,t,0], states[:,t,1], states[:,t,2]
                        dx_dt = model(prey_population, intermediate_population, top_predators_population)
                    elif env_name == 'Dataset-HL':
                        hare, lynx, time = states[:,t,0], states[:,t,1], actions[:,t,0]
                        dx_dt = model(hare, lynx, time)
                    dx_dt = torch.stack(dx_dt, dim=-1)
                    pred_state = states[:,t] + dx_dt
                pred_states = torch.stack(pred_states, dim=1)
                val_loss = MSE(pred_states, states).item()
                loss_per_dim = torch.mean(torch.square(pred_states - states), dim=(0,1)).cpu().tolist()
            model.train()
            return val_loss, loss_per_dim
                
        best_model = None
        if config.run.optimize_params:
            best_val_loss = float('inf')  # Initialize with a very high value
            patience_counter = 0  # Counter for tracking patience

            for epoch in range(config.run.pytorch_as_optimizer.epochs):
                iters = 0 
                cum_loss = 0
                t0 = time.perf_counter()
                permutation = torch.randperm(states_train.shape[0])
                for iter_i in range(int(permutation.shape[0]/config.run.pytorch_as_optimizer.batch_size)):
                    indices = permutation[iter_i*config.run.pytorch_as_optimizer.batch_size:iter_i*config.run.pytorch_as_optimizer.batch_size+config.run.pytorch_as_optimizer.batch_size]
                    states_train_batch = states_train[indices]
                    if actions_train is not None:
                        actions_train_batch = actions_train[indices]
                    else:
                        actions_train_batch = None
                    cum_loss += train_opt(f_model, states_train_batch, actions_train_batch)
                    iters += 1
                time_taken = time.perf_counter() - t0
                if epoch % config.run.pytorch_as_optimizer.log_interval == 0:
                    # Collect validation loss
                    val_loss, _ = compute_eval_loss(f_model, (states_val, actions_val))
                    print(f'[EPOCH {epoch} COMPLETE] MSE TRAIN LOSS {cum_loss/iters:.4f} | MSE VAL LOSS {val_loss:.4f} | s/epoch: {time_taken:.2f}s')
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = deepcopy(f_model.state_dict())
                        patience_counter = 0  # Reset counter on improvement
                    else:
                        patience_counter += 1  # Increment counter if no improvement
                    if patience_counter >= config.run.optimization.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break  # Exit the loop if no improvement for 'patience' generations
        else:
            cum_loss, iters = 1, 1

        # Save model after training
        f_model.eval()
        if best_model is not None:
            f_model.load_state_dict(best_model)
            print('Loaded best model')
            
        val_loss, _ = compute_eval_loss(f_model, (states_val, actions_val))
        # torch.save(f_model.state_dict(), f'{folder_path}dynode_model_{env.env_name}_{env.seed}.pt')
        print(f'[Train Run completed successfully] MSE VAL LOSS {val_loss:.4f}')
        print('')

        val_loss, loss_per_dim = compute_eval_loss(f_model, (states_val, actions_val))
        train_loss = cum_loss/iters
        optimized_parameters = get_model_parameters(f_model)

        states_test, actions_test = test_data
        if actions_test is not None:
            actions_test = torch.tensor(actions_test, dtype=torch.float32, device=device)
        states_test = torch.tensor(states_test, dtype=torch.float32, device=device)
        test_data = (states_test, actions_test)

        test_loss, _ = compute_eval_loss(f_model, test_data)

        return train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss
    

def load_data(config={}, seed=0, env_name='', train_ratio=0.7, val_ratio=0.15):
    if env_name == 'Dataset-3DLV':
        pandas_csv_path = './libs/datasets/data/TS_3DLV.csv'
        df = pd.read_csv(pandas_csv_path, sep=';')

        total_time_steps = df.shape[0]
        train_data = (df.iloc[:int(total_time_steps*train_ratio),1:].values[np.newaxis, :, :], None)
        val_data = (df.iloc[int(total_time_steps*train_ratio):int(total_time_steps*(train_ratio+val_ratio)),1:].values[np.newaxis, :, :], None)
        test_data = (df.iloc[int(total_time_steps*(train_ratio+val_ratio)):,1:].values[np.newaxis, :, :], None)
    elif env_name == 'Dataset-HL':
        pandas_csv_path = './libs/datasets/data/TS_HL.csv'
        df = pd.read_csv(pandas_csv_path, sep=';')

        total_time_steps = df.shape[0]
        train_data = (df.iloc[:int(total_time_steps*train_ratio),1:].values[np.newaxis, :, :], df.iloc[:int(total_time_steps*train_ratio),:1].values[np.newaxis, :, :])
        val_data = (df.iloc[int(total_time_steps*train_ratio):int(total_time_steps*(train_ratio+val_ratio)),1:].values[np.newaxis, :, :], df.iloc[int(total_time_steps*train_ratio):int(total_time_steps*(train_ratio+val_ratio)),:1].values[np.newaxis, :, :])
        test_data = (df.iloc[int(total_time_steps*(train_ratio+val_ratio)):,1:].values[np.newaxis, :, :], df.iloc[int(total_time_steps*(train_ratio+val_ratio)):,:1].values[np.newaxis, :, :])
    else:
        raise NotImplementedError
    
    return train_data, val_data, test_data, ''

class TestEnvOptim(unittest.TestCase):
    def setUp(self):
        from hydra import initialize, compose
        initialize(config_path="../../config", version_base=None)  # Point to your actual config directory        
        self.config = compose(config_name="config.yaml")
        self.num_patients = 1000 
        load_data(config=self.config)

    def test_latest_with_pytorch_model(self):

        class StateDifferential(nn.Module):
            def __init__(self):
                super(StateDifferential, self).__init__()
                # Define the parameters for the tumor growth model
                self.alpha = nn.Parameter(torch.tensor(0.1))
                self.beta = nn.Parameter(torch.tensor(0.05))
                # Define the parameters for the chemotherapy effect
                self.gamma = nn.Parameter(torch.tensor(0.01))
                self.delta = nn.Parameter(torch.tensor(0.005))
                # Define the parameters for the radiotherapy effect
                self.epsilon = nn.Parameter(torch.tensor(0.02))
                self.zeta = nn.Parameter(torch.tensor(0.01))
                # Define a neural network for capturing complex interactions and residuals
                self.residual_nn = nn.Sequential(
                    nn.Linear(4, 10),
                    nn.ReLU(),
                    nn.Linear(10, 2)
                )

            def forward(self, tumor_volume: torch.Tensor, chemotherapy_drug_concentration: torch.Tensor, chemotherapy_dosage: torch.Tensor, radiotherapy_dosage: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                # Tumor growth model
                d_tumor_volume__dt = self.alpha * tumor_volume - self.beta * tumor_volume * chemotherapy_drug_concentration - self.epsilon * tumor_volume * radiotherapy_dosage
                # Chemotherapy drug concentration model
                d_chemotherapy_drug_concentration__dt = self.gamma * chemotherapy_dosage - self.delta * chemotherapy_drug_concentration
                # Neural network to model residuals
                residuals = self.residual_nn(torch.cat((tumor_volume.unsqueeze(1), chemotherapy_drug_concentration.unsqueeze(1), chemotherapy_dosage.unsqueeze(1), radiotherapy_dosage.unsqueeze(1)), dim=1))
                # Add residuals to the model
                d_tumor_volume__dt += residuals[:, 0]
                d_chemotherapy_drug_concentration__dt += residuals[:, 1]

                return (d_tumor_volume__dt, d_chemotherapy_drug_concentration__dt)

        train_loss, val_loss, optimized_parameters = self.env.evaluate_simulator_code_using_pytorch(StateDifferential, self.train_data, self.val_data, self.test_data, self.config)
        # train_loss, val_loss, optimized_parameters = self.env.evaluate_simulator_code_using_pytorch_with_neuroevolution(StateDifferential, self.train_data, self.val_data)
        print(f'Optimizer {self.optimizer} : Final Train MSE: {train_loss} | Final Val MSE: {val_loss}') # According to code it is 2694.2922 -- suspect data leakage error
        print(f'Optimized parameters: {optimized_parameters}')
        assert val_loss < 12.3232 * 2.0, "Val loss is too high"
        print('')



if __name__ == "__main__":
    test = TestEnvOptim()
    test.setUp()
    # test.test_latest_with_pytorch_model()
    test.test_latest_with_pytorch_model()
