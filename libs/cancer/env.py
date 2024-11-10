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


def calc_volume(diameter):
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0

def calc_diameter(volume):
    return ((volume / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)) * 2.0

# Tumour constants per
TUMOUR_CELL_DENSITY = 5.8 * 10 ** 8  # cells per cm^3
TUMOUR_DEATH_THRESHOLD = calc_volume(13)  # assume spherical

# Patient cancer stage. (mu, sigma, lower bound, upper bound) - for lognormal dist
tumour_size_distributions = {'I': (1.72, 4.70, 0.3, 5.0),
                             'II': (1.96, 1.63, 0.3, 13.0),
                             'IIIA': (1.91, 9.40, 0.3, 13.0),
                             'IIIB': (2.76, 6.87, 0.3, 13.0),
                             'IV': (3.86, 8.82, 0.3, 13.0)}  # 13.0 is the death condition

# Observations of stage proportions taken from Detterbeck and Gibson 2008
# - URL: http://www.jto.org/article/S1556-0864(15)33353-0/fulltext#cesec50\
cancer_stage_observations = {'I': 1432,
                             "II": 128,
                             "IIIA": 1306,
                             "IIIB": 7248,
                             "IV": 12840}

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

class CancerEnv:
    def __init__(self):
        self.action_max = 2.0
        self.max_chemo_drug = 5.0
        self.max_radio = 2.0
        self.use_e_noise = False

        if probabilistic:
            self.e_noise = np.randn(1).to(device)[0] * 0.01
            self.rho = 7e-5 + np.randn(1).to(device)[0] * 7.23e-3
            self.K = calc_volume(30)
            self.beta_c = 0.028 + np.randn(1).to(device)[0] * 0.0007
            self.alpha_r = 0.0398 + np.randn(1).to(device)[0] * 0.168
            self.beta_r = self.alpha_r /10.0
        else:
            self.e_noise = 0
            # self.rho = 7e-5 + 7.23e-3 * 4 # Good for inducing death
            self.rho = 7e-5 + 7.23e-3 * 2
            self.K = calc_volume(30)
            self.beta_c = 0.028
            self.alpha_r = 0.0398
            self.beta_r = self.alpha_r /10.0
        self.v_death_thres = calc_volume(13)

    # def reset(self):
    #     state = np.random.uniform(low=calc_volume(13)*0.15, high=calc_volume(13)*0.6, size=(2,)).astype('float32')
    #     state[1] = 0.0 # Zero drug concentration starting
    #     v, c = state
    #     self.state = (v, c)
    #     self.time_step = 0
    #     return self.state

    def get_initial_tumor_size(self, num_patients=256, env_name='', variation=''):
        # # INITIAL VOLUMES SAMPLING
        # TOTAL_OBS = sum(cancer_stage_observations.values())
        # cancer_stage_proportions = {k: cancer_stage_observations[k] / TOTAL_OBS for k in cancer_stage_observations}

        # # remove possible entries
        # possible_stages = list(tumour_size_distributions.keys())
        # possible_stages.sort()

        # initial_stages = np.random.choice(possible_stages, num_patients,
        #                                 p=[cancer_stage_proportions[k] for k in possible_stages])

        # # Get info on patient stages and initial volumes
        # output_initial_diam = []
        # # patient_sim_stages = []
        # for stg in possible_stages:
        #     count = np.sum((initial_stages == stg) * 1)

        #     mu, sigma, lower_bound, upper_bound = tumour_size_distributions[stg]

        #     # Convert lognorm bounds in to standard normal bounds
        #     lower_bound = (np.log(lower_bound) - mu) / sigma
        #     upper_bound = (np.log(upper_bound) - mu) / sigma

        #     norm_rvs = truncnorm.rvs(lower_bound, upper_bound,
        #                             size=count)  # truncated normal for realistic clinical outcome

        #     initial_volume_by_stage = np.exp((norm_rvs * sigma) + mu)
        #     output_initial_diam += list(initial_volume_by_stage)
        #     # patient_sim_stages += [stg for i in range(count)]

        # initial_volumes = calc_volume(np.array(output_initial_diam))
        if env_name == 'Cancer-ood' or env_name == 'Cancer-iid':
            if variation == 'train':
                initial_volumes = np.random.rand(num_patients) * 1149.8817159979046 * 0.5 + 0.014165261814709135
            elif variation == 'val':
                initial_volumes = np.random.rand(num_patients) * 1149.8817159979046 * 0.5 + 0.014165261814709135
            elif variation == 'test':
                if env_name == 'Cancer-ood':
                    initial_volumes = np.random.rand(num_patients) * 1149.8817159979046 * 0.3 + 1149.8817159979046 * 0.7
                    # initial_volumes = np.random.rand(num_patients) * 1149.8817159979046 * 0.5 + 1149.8817159979046 * 0.5
                elif env_name == 'Cancer-iid':
                    initial_volumes = np.random.rand(num_patients) * 1149.8817159979046 * 0.5 + 0.014165261814709135
            else:
                raise NotImplementedError
        else:
            initial_volumes = np.random.rand(num_patients) * 1149.8817159979046 + 0.014165261814709135

        return initial_volumes

        # STATIC VARIABLES SAMPLING
        # Fixed params
        K = calc_volume(30)  # carrying capacity given in cm, so convert to volume

    def reset(self, num_patients=1, env_name='', variation=''):
        v = self.get_initial_tumor_size(num_patients=num_patients, env_name=env_name, variation=variation)
        c = np.zeros_like(v)
        self.state = (v, c)
        self.time_step = 0
        return self.state

    def state_diff(self, cancer_volume, chemo_concentration, chemotherapy_dosage, radiotherapy_dosage, env_name=''):
        """
        State inputs are batched. (B, 1)
        """
        if self.use_e_noise:
            e = np.random.rand(1).to(self.d)[0] * 0.1
        else:
            e = self.e_noise
        assert cancer_volume.shape == chemo_concentration.shape == chemotherapy_dosage.shape == radiotherapy_dosage.shape, "Shapes of inputs must be the same"
        assert len(cancer_volume.shape) == len(chemo_concentration.shape) == len(chemotherapy_dosage.shape) == len(radiotherapy_dosage.shape) == 1, "Inputs must be batched"

        rho = self.rho
        K = self.K
        beta_c = self.beta_c
        alpha_r = self.alpha_r
        beta_r = self.beta_r

        v, c = cancer_volume, chemo_concentration
        v[v<=0] = 0
        ca = np.clip(chemotherapy_dosage, a_min=0, a_max=self.max_chemo_drug)
        ra = np.clip(radiotherapy_dosage, a_min=0, a_max=self.max_radio)

        dc_dt = - c / 2 + ca

        if env_name == 'Cancer-random-1':
            dv_dt = (rho * np.log(K / v) - beta_c * c - (alpha_r * ra + beta_r * np.square(ra)) + e + 0.08 * np.sin(v)) * v
        elif env_name == 'Cancer-random-2':
            dv_dt = (rho * np.log(K / v) - beta_c * c - (alpha_r * ra + beta_r * np.square(ra)) + e - 1.25 * (v / K)) * v
        elif env_name == 'Cancer-random-3':
            dv_dt = (rho * np.log(K / (v + c)) - beta_c * c - (alpha_r * ra + beta_r * np.square(ra)) + e) * v
        elif env_name == 'Cancer-random-4':
            dv_dt = (rho * np.log(K / v) - beta_c * c - (alpha_r * ra + beta_r * np.square(ra)) + e + 0.08 * np.cos(v)) * v
        elif env_name == 'Cancer-random-5':
            dv_dt = (rho * np.log(K / v) - beta_c * c - (alpha_r * ra + beta_r * np.square(ra)) + e - 1.25 * (v / K) * (c / 100)) * v
        else:
            dv_dt = (rho * np.log(K / v) - beta_c * c - (alpha_r * ra + beta_r * np.square(ra)) + e) * v
        dv_dt[v==0] = 0
        dv_dt = np.nan_to_num(dv_dt, posinf=0, neginf=0)
        return dv_dt, dc_dt
    
    def evaluate_simulator_code_wrapper(self, StateDifferential, train_data, val_data, test_data, config={}, logger=None, env_name=''):
        if config.run.optimizer == 'pytorch':
            train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss = self.evaluate_simulator_code_using_pytorch(StateDifferential, train_data, val_data, test_data, config=config, logger=logger, env_name=env_name)
        elif 'evotorch' in config.run.optimizer:
            train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss = self.evaluate_simulator_code_using_pytorch_with_neuroevolution(StateDifferential, train_data, val_data, test_data, config=config, logger=logger)
        if env_name == 'Cancer' or env_name == 'Cancer-ood' or env_name == 'Cancer-iid' or 'Cancer-random' in env_name:
            loss_per_dim_dict = {'tumor_volume_val_loss': loss_per_dim[0], 'chemo_drug_concentration_val_loss': loss_per_dim[1]}
        elif env_name == 'Cancer-untreated':
            loss_per_dim_dict = {'tumor_volume_val_loss': loss_per_dim[0]}
        elif env_name == 'Cancer-chemo':
            loss_per_dim_dict = {'tumor_volume_val_loss': loss_per_dim[0], 'chemo_drug_concentration_val_loss': loss_per_dim[1]}
        return train_loss, val_loss, optimized_parameters, loss_per_dim_dict, test_loss
    
    def evaluate_simulator_code_on_dataset(self, env_state_diff, parameters, data, config={}, logger=None):
        keys = sorted(parameters.keys())
        x0 = jnp.array([parameters[key] for key in keys])
        data = (jnp.array(data[0]), jnp.array(data[1]))

        def f_with_data_inner(x0, data_to_evaluate):
            keys = sorted(parameters.keys())
            parameters_in = {key: value for key, value in zip(keys, x0)}
            states_train, actions_train = data_to_evaluate
            # assert not np.any(np.isnan(states_train)), "States array contains NaN"
            # assert not np.any(np.isnan(actions_train)), "Actions array contains NaN"
            v, c = states_train[:,0,0], states_train[:,0,1]
            simulated_states = []
            # simulated_actions = [] # For debugging purposes
            for i in range(states_train.shape[1]):
                simulated_states.append(jnp.stack((v,c),axis=1))
                chemo_dosage, radio_dosage = actions_train[:,i,0], actions_train[:,i,1]
                # simulated_actions.append(np.stack((chemo_dosage, radio_dosage), axis=1))
                dv_dt, dc_dt = env_state_diff(v, c, chemo_dosage, radio_dosage, parameters_in)
                dv_dt = jnp.where(v == 0, 0, dv_dt)
                # dv_dt[v==0] = 0
                v += dv_dt
                c += dc_dt
                v = jnp.where(v <= 0, 0, v)
                c = jnp.where(c <= 0, 0, c)
                # v[v<=0] = 0
                # c[c<=0] = 0
            return jnp.stack(simulated_states, axis=1)


        def f_with_data(x0, data_to_evaluate):
            simulated_states = f_with_data_inner(x0, data_to_evaluate)
            loss = jnp.mean(jnp.square(simulated_states - data_to_evaluate[0]))
            return loss

        loss = f_with_data(x0, data).item()
        return loss


    def evaluate_simulator_code(self, env_state_diff, parameters, train_data, val_data, optimizer = "OptunaCmaEsSampler"):
        def f_with_data(x0, data_to_evaluate):
            parameters_in = array_to_dict(x0, parameters)
            states_train, actions_train = data_to_evaluate
            assert not np.any(np.isnan(states_train)), "States array contains NaN"
            assert not np.any(np.isnan(actions_train)), "Actions array contains NaN"
            v, c = states_train[:,0,0].copy(), states_train[:,0,1].copy()
            simulated_states = []
            # simulated_actions = [] # For debugging purposes
            for i in range(states_train.shape[1]):
                simulated_states.append(np.stack((v,c),axis=1))
                chemo_dosage, radio_dosage = actions_train[:,i,0], actions_train[:,i,1]
                # simulated_actions.append(np.stack((chemo_dosage, radio_dosage), axis=1))
                dv_dt, dc_dt = env_state_diff(v, c, chemo_dosage, radio_dosage, parameters_in)
                # dv_dt[v==0] = 0
                v += dv_dt
                c += dc_dt
                # v[v<=0] = 0
                # c[c<=0] = 0
            simulated_states = np.stack(simulated_states, axis=1)
            # simulated_states[simulated_states > 1e-4] = 1e-4 # Clip to 1e-4 to avoid NaNs
            # np.argwhere(np.isnan(simulated_states))
            # simulated_states = np.nan_to_num(simulated_states, posinf=0, neginf=0)
            # assert not np.any(np.isnan(simulated_states)), "Array contains NaN"
            # simulated_actions = np.stack(simulated_actions, axis=1)
            loss = np.mean(np.square(simulated_states - states_train))
            if np.isnan(loss):
                # loss = 1e10
                loss = 1e6
            # action_loss = np.mean(np.square(simulated_actions - actions_train)) # Should be 0.0 always
            # assert not np.any(np.isnan(loss)), "Loss is NaN"
            return loss

        f_train = partial(f_with_data, data_to_evaluate=train_data)
        f_val = partial(f_with_data, data_to_evaluate=val_data)
        x0 = dict_to_array(parameters)

        keys = sorted(parameters.keys())
        std_multiplier = 1.0
        bounds = [(max(parameters[key][0] - parameters[key][1] * std_multiplier, 1e-6), parameters[key][0] + parameters[key][1] * std_multiplier) for key in keys]

        parameters = {k: v[0] for k, v in parameters.items()}


        # bounds = generate_bounds(parameters)

        

        if optimizer == "BFGS":
            # loss = f_train(x0)
            # print(f"Initial loss: {loss}")
            # res = minimize(f_train, x0, method='BFGS', tol=1e-6, options={'disp': True, 'maxiter': 10000})
            # res = minimize(f_train, x0, method='Nelder-Mead', tol=1e-6, options={'disp': True, 'maxiter': 10000}, bounds=[(-10.0, 10.0) for i in range(len(x0))])
            # res = minimize(f_train, x0, method='L-BFGS-B', tol=1e-6, options={'disp': True, 'maxiter': 10000}, bounds=bounds)
            res = minimize(f_train, x0, method='Nelder-Mead', tol=1e-6, options={'disp': True, 'maxiter': 10000}, bounds=bounds)
            x0_optimized = res.x
            # print(f'BFGS : Final Train MSE: {f_train(res.x)} | Final Val MSE: {f_val(res.x)}')
            # res = minimize(f_train, x0, method='Nelder-Mead', tol=1e-6, options={'disp': True, 'maxiter': 10000})
            # print(f'Nelder-Mead : Final MSE: {f_train(res.x)} | Final Val MSE: {f_val(res.x)}')
            # print('')
        elif optimizer == "OptunaCmaEsSampler":
            import optuna
            def objective(trial):
                keys = sorted(parameters.keys())
                x0_suggested = []
                for i, key in enumerate(keys):
                    # x0_suggested.append(trial.suggest_float(key, 1e-6, bounds[i][1], log=True))
                    x0_suggested.append(trial.suggest_float(key, bounds[i][0], bounds[i][1]))
                    # x0_suggested.append(trial.suggest_float(key, 1e-6, 2000.0, log=True))
                    # x0_suggested.append(trial.suggest_float(key, 1e-6, 2000.0))
                    # if type(parameters[key]) == float:
                    #     # x0_suggested.append(trial.suggest_float(key, 0.0, 5.0))
                    #     # x0_suggested.append(trial.suggest_float(key, -2000.0, 2000.0))
                    #     # x0_suggested.append(trial.suggest_float(key, -10.0, 10.0))
                    #     # x0_suggested.append(trial.suggest_float(key, *bounds[i]))
                    #     x0_suggested.append(trial.suggest_float(key, 1e-6, 2000.0, log=True))
                    # else:
                    #     x0_suggested.append(trial.suggest_int(key, 0, 2000))
                # x0_suggested = [trial.suggest_float(key, 0.0, 1.0) if type(parameters[key]) == 'float' else trial.suggest_int(key, 0, 60) for key in keys]
                # x0_suggested = [trial.suggest_float(f'{i}', 0.0, 1.0) if  for i in range(len(x0))]
                return f_train(x0_suggested)
            study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
            # study = optuna.create_study(sampler=optuna.samplers.TPESampler())
            # study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler())
            study.enqueue_trial(parameters)
            study.optimize(objective, n_trials=5000)
            x0_optimized = list(study.best_params.values())
            print(study.best_trials)
            print('')

        optimized_parameters = array_to_dict(x0_optimized, parameters)
        val_loss = f_val(x0_optimized)
        train_loss = f_train(x0_optimized)
        print(f'Optimizer {optimizer} : Final Train MSE: {f_train(x0_optimized)} | Final Val MSE: {f_val(x0_optimized)}')
        return train_loss, val_loss, optimized_parameters
    
    
    def evaluate_simulator_code_using_pytorch(self, StateDifferential, train_data, val_data, test_data, config={}, logger=None, env_name=''):
        import torch
        import numpy as np
        device = "cuda:0"

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
                if env_name == 'Cancer' or env_name == 'Cancer-ood' or env_name == 'Cancer-iid' or 'Cancer-random' in env_name:
                    tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage = states_train_batch_i[:,t,0], states_train_batch_i[:,t,1], actions_train_batch_i[:,t,0], actions_train_batch_i[:,t,1]
                    dx_dt = model(tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage)
                    dx_dt = torch.stack(dx_dt, dim=-1)
                elif env_name == 'Cancer-untreated':
                    tumor_volume = states_train_batch_i[:,t,0]
                    dx_dt = model(tumor_volume)
                    dx_dt = dx_dt[0].unsqueeze(-1)
                elif env_name == 'Cancer-chemo':
                    tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage = states_train_batch_i[:,t,0], states_train_batch_i[:,t,1], actions_train_batch_i[:,t,0]
                    dx_dt = model(tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage)
                    dx_dt = torch.stack(dx_dt, dim=-1)
                pred_state = states_train_batch_i[:,t] + dx_dt
            pred_states = torch.stack(pred_states, dim=1)
            loss = MSE(pred_states, states_train_batch_i)
            loss.backward()
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
                pred_state = states[:,0]
                for t in range(states.shape[1]):
                    pred_states.append(pred_state)
                    if env_name == 'Cancer' or env_name == 'Cancer-ood' or env_name == 'Cancer-iid' or 'Cancer-random' in env_name:
                        tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage = states[:,t,0], states[:,t,1], actions[:,t,0], actions[:,t,1]
                        dx_dt = model(tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage)
                        dx_dt = torch.stack(dx_dt, dim=-1)
                    elif env_name == 'Cancer-untreated':
                        tumor_volume = states[:,t,0]
                        dx_dt = model(tumor_volume)
                        dx_dt = dx_dt[0].unsqueeze(-1)
                    elif env_name == 'Cancer-chemo':
                        tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage = states[:,t,0], states[:,t,1], actions[:,t,0]
                        dx_dt = model(tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage)
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
    
    
    
    def evaluate_simulator_code_using_pytorch_with_neuroevolution(self, StateDifferential, train_data, val_data, config={}, logger=None):
        import torch
        from evotorch.tools import dtype_of, device_of
        from evotorch.neuroevolution import NEProblem
        # device = "cuda:0"
        device = "cpu"

        batch_size = 1000
        learning_rate = 1e-2
        weight_decay = 0.0
        epochs = 10000
        log_interval = 10

        states_train_np, actions_train_np = train_data
        # states_train, actions_train = torch.tensor(states_train, dtype=torch.float32, device=device), torch.tensor(actions_train, dtype=torch.float32, device=device)

        # states_val, actions_val = val_data
        # states_val, actions_val = torch.tensor(states_val, dtype=torch.float32, device=device), torch.tensor(actions_val, dtype=torch.float32, device=device)
        MSE = torch.nn.MSELoss()

        def evaluate_model(network: torch.nn.Module, data: Tuple[np.ndarray, np.ndarray]):
            states_train_np, actions_train_np = train_data
            states_train, actions_train = torch.tensor(states_train_np, dtype=dtype_of(network), device=device_of(network)), torch.tensor(actions_train_np, dtype=dtype_of(network), device=device_of(network))
            network.eval()
            with torch.no_grad():
                pred_states = []
                pred_state = states_train[:,0]
                for t in range(states_train.shape[1]):
                    pred_states.append(pred_state)
                    tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage = states_train[:,t,0], states_train[:,t,1], actions_train[:,t,0], actions_train[:,t,1]
                    # tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage = pred_state[:,0], pred_state[:,1], actions_val[:,t,0], actions_val[:,t,1]
                    dx_dt = network(tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage)
                    dx_dt = torch.stack(dx_dt, dim=-1)
                    # dx_dt = f_model(states_val[:,t], actions_val[:,t])
                    pred_state = states_train[:,t] + dx_dt
                pred_states = torch.stack(pred_states, dim=1)
                val_loss = MSE(pred_states, states_train)
            return val_loss
        
        evaluate_model_train = partial(evaluate_model, data=train_data)
        evaluate_model_val = partial(evaluate_model, data=val_data)

        ACTORS = 16.0
        min_problem_train_dataset = NEProblem(
            # The objective sense -- we wish to maximize the sign_prediction_score
            objective_sense="min",
            # The network is a Linear layer mapping 3 inputs to 1 output
            network=StateDifferential,
            # Networks will be evaluated according to sign_prediction_score
            network_eval_func=evaluate_model_train,
            num_actors=ACTORS,
            # num_gpus_per_actor=(1 / ACTORS),
        )

        from evotorch.algorithms import PGPE, CMAES 
        from evotorch.logging import PandasLogger

        t0 = time.perf_counter()
        searcher = PGPE(
            min_problem_train_dataset,
            popsize=50,
            radius_init=2.25,
            center_learning_rate=0.2,
            stdev_learning_rate=0.1,
        )
        
        # searcher = PGPE(
        #     problem=min_problem_train_dataset,
        #     popsize=100,  # A larger population size for better exploration
        #     center_learning_rate=0.05,  # Learning rate for the distribution center
        #     stdev_learning_rate=0.1,  # Learning rate for the standard deviations
        #     stdev_init=0.1,  # Initial standard deviation for exploration
        #     optimizer='clipup',  # Using ClipUp as the default optimizer
        #     optimizer_config={'max_speed': 0.2},  # Configuring ClipUp with a maximum speed
        #     ranking_method='centered',  # 0-centered ranking for fitness shaping
        #     stdev_min=0.01,  # Lower bound to prevent standard deviation from becoming too small
        #     stdev_max=1.0,  # Upper bound to prevent standard deviation from becoming too large
        #     stdev_max_change=0.2,  # Limiting the maximum change in standard deviation to 20%
        #     symmetric=True,  # Using symmetric sampling as recommended
        # )

        # searcher = CMAES(
        #     min_problem_train_dataset,
        #     popsize=50,
        #     stdev_init=2.25,
        # )

        # searcher = CMAES(
        #     problem=min_problem_train_dataset,
        #     popsize=4 + int(3 * np.log(78)),  # Following the rule of thumb 4 + floor(3*log(D))
        #     stdev_init=0.5,  # Smaller initial step-size to prevent overshooting
        #     active=True,  # Active CMA-ES for more aggressive updates
        #     c_sigma_ratio=1.5,  # Increase to allow faster adaptation of the step size
        #     damp_sigma_ratio=1.5,  # Same rationale as c_sigma_ratio
        #     c_c_ratio=1.2,  # Slightly increased to allow faster adaptation of the covariance matrix
        #     c_1_ratio=1.2,  # Slightly increased for the same reason as c_c_ratio
        #     c_mu_ratio=1.2,  # Slightly increased to allow more weight on the rank-mu update
        #     separable=False,  # Keep it false for full covariance matrix adaptation
        #     limit_C_decomposition=True,  # Limiting decomposition frequency for computational efficiency
        # )

        logger = PandasLogger(searcher)
        for i in range(10):
            print(f"Generation {i*50}")
            searcher.run(50)
            print(f"Best Loss {searcher.status['best_eval']} | Mean Loss {searcher.status['mean_eval']} | Best Params {searcher.status['best']}")

        print(f"Total time taken: {time.perf_counter() - t0:.2f}s")


        import matplotlib.pyplot as plt
        logger.to_dataframe().mean_eval.plot()
        plt.savefig('test.png')
        print('')

        model_best = min_problem_train_dataset.make_net(searcher.status['best'])
        model_center = min_problem_train_dataset.make_net(searcher.status['center'])

        print(f'Validation Loss Best: {evaluate_model_val(model_best)} | Validation Loss Center: {evaluate_model_val(model_center)}')
        print('')
        print(evaluate_model_val(model_best))
        print(evaluate_model_val(model_center))

        searcher.status['center']
        searcher.status['best']


        def train(model, states_train_batch_i, actions_train_batch_i):
            optimizer.zero_grad(True)
            pred_states = []
            pred_state = states_train_batch_i[:,0]
            for t in range(states_train_batch_i.shape[1]):
                pred_states.append(pred_state)
                # tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage = pred_state[:,0], pred_state[:,1], actions_train_batch[:,t,0], actions_train_batch[:,t,1]
                tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage = states_train_batch_i[:,t,0], states_train_batch_i[:,t,1], actions_train_batch_i[:,t,0], actions_train_batch_i[:,t,1]
                dx_dt = model(tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage)
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
                
        for epoch in range(epochs):
            iters = 0 
            cum_loss = 0
            t0 = time.perf_counter()
            permutation = torch.randperm(states_train.shape[0])
            for iter_i in range(int(permutation.shape[0]/batch_size)):
                indices = permutation[iter_i*batch_size:iter_i*batch_size+batch_size]
                states_train_batch, actions_train_batch = states_train[indices], actions_train[indices]
                cum_loss += train_opt(f_model, states_train_batch, actions_train_batch)
                iters += 1
            time_taken = time.perf_counter() - t0
            if epoch % log_interval == 0:
                # Collect validation loss
                f_model.eval()
                with torch.no_grad():
                    pred_states = []
                    pred_state = states_val[:,0]
                    for t in range(states_val.shape[1]):
                        pred_states.append(pred_state)
                        tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage = states_val[:,t,0], states_val[:,t,1], actions_val[:,t,0], actions_val[:,t,1]
                        # tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage = pred_state[:,0], pred_state[:,1], actions_val[:,t,0], actions_val[:,t,1]
                        dx_dt = f_model(tumor_volume, chemotherapy_drug_concentration, chemotherapy_dosage, radiotherapy_dosage)
                        dx_dt = torch.stack(dx_dt, dim=-1)
                        # dx_dt = f_model(states_val[:,t], actions_val[:,t])
                        pred_state = states_val[:,t] + dx_dt
                    pred_states = torch.stack(pred_states, dim=1)
                    val_loss = MSE(pred_states, states_val)
                f_model.train()
                print(f'[EPOCH {epoch} COMPLETE] MSE TRAIN LOSS {cum_loss/iters:.4f} | MSE VAL LOSS {val_loss:.4f} | s/epoch: {time_taken:.2f}s')

        # Save model after training
        f_model.eval()
        # torch.save(f_model.state_dict(), f'{folder_path}dynode_model_{env.env_name}_{env.seed}.pt')
        print(f'[Train Run completed successfully] MSE VAL LOSS {val_loss:.4f}')
        print('')
    
    
    def evaluate_simulator_code_using_jax(self, env_state_diff, parameters, train_data, val_data, config={}, logger=None, plot_logging=False):

        num_gens = 100
        keys = sorted(parameters.keys())
        std_multiplier = 1.0
        x0_stds = jnp.array([parameters[key][1] for key in keys]) * std_multiplier
        x0_means = jnp.array([parameters[key][0] for key in keys])
        train_data = (jnp.array(train_data[0]), jnp.array(train_data[1]))
        val_data = (jnp.array(val_data[0]), jnp.array(val_data[1]))

        def f_with_data_inner(x0, data_to_evaluate):
            # parameters_in = array_to_dict(x0, parameters)
            keys = sorted(parameters.keys())
            # Unnormalize
            x0 = x0 * x0_stds + x0_means
            x0 = jnp.clip(x0, a_min=0, a_max=None)
            parameters_in = {key: value for key, value in zip(keys, x0)}
            states_train, actions_train = data_to_evaluate
            # assert not np.any(np.isnan(states_train)), "States array contains NaN"
            # assert not np.any(np.isnan(actions_train)), "Actions array contains NaN"
            v, c = states_train[:,0,0], states_train[:,0,1]
            simulated_states = []
            # simulated_actions = [] # For debugging purposes
            for i in range(states_train.shape[1]):
                simulated_states.append(jnp.stack((v,c),axis=1))
                chemo_dosage, radio_dosage = actions_train[:,i,0], actions_train[:,i,1]
                # simulated_actions.append(np.stack((chemo_dosage, radio_dosage), axis=1))
                dv_dt, dc_dt = env_state_diff(v, c, chemo_dosage, radio_dosage, parameters_in)
                dv_dt = jnp.where(v == 0, 0, dv_dt)
                # dv_dt[v==0] = 0
                v += dv_dt
                c += dc_dt
                v = jnp.where(v <= 0, 0, v)
                c = jnp.where(c <= 0, 0, c)
                # v[v<=0] = 0
                # c[c<=0] = 0
            return jnp.stack(simulated_states, axis=1)
            # return simulated_states
            # # simulated_states[simulated_states > 1e-4] = 1e-4 # Clip to 1e-4 to avoid NaNs
            # # np.argwhere(np.isnan(simulated_states))
            # # simulated_states = np.nan_to_num(simulated_states, posinf=0, neginf=0)
            # # assert not np.any(np.isnan(simulated_states)), "Array contains NaN"
            # # simulated_actions = np.stack(simulated_actions, axis=1)
            # loss = jnp.mean(jnp.square(simulated_states - states_train))
            # # if np.isnan(loss):
            # #     # loss = 1e10
            # #     loss = 1e6
            # # action_loss = np.mean(np.square(simulated_actions - actions_train)) # Should be 0.0 always
            # # assert not np.any(np.isnan(loss)), "Loss is NaN"
            # return loss

        def f_with_data(x0, data_to_evaluate):
            simulated_states = f_with_data_inner(x0, data_to_evaluate)
            loss = jnp.mean(jnp.square(simulated_states - data_to_evaluate[0]))
            return loss



        f_train = partial(f_with_data, data_to_evaluate=train_data)
        f_val = partial(f_with_data, data_to_evaluate=val_data)

        # bounds = [(max(parameters[key][0] - parameters[key][1] * std_multiplier, 1e-6), parameters[key][0] + parameters[key][1] * std_multiplier) for key in keys]

        parameters = {k: v[0] for k, v in parameters.items()}
        x0 = dict_to_array(parameters)

        @jit
        def process_batch(x_batched):
            return jax.vmap(f_train)(x_batched)

        # for s_name in ["SimpleES", "SimpleGA", "PSO", "DE", "Sep_CMA_ES", "Full_iAMaLGaM", "Indep_iAMaLGaM", "MA_ES", "LM_MA_ES", "RmES", "GLD", "SimAnneal", "GESMR_GA", "SAMR_GA"]:
        # for s_name in ["DE"]:
        # from evosax import CMA_ES
        from evosax import Strategies
        if plot_logging:
            from evosax.utils import ESLog
            es_logging = ESLog(num_dims=len(x0), num_generations=num_gens, top_k=3, maximize=False)
            log = es_logging.initialize()

        optimizer = config.run.optimizer
        s_name = optimizer.split('-')[1]
        strategy = Strategies[s_name](num_dims=len(x0), popsize=1000, elite_ratio=0.5) # 'CMA_ES'

        es_params = strategy.default_params
        es_params = es_params.replace(init_min=-3, init_max=3) #, clip_min=1e6, clip_max=15000)

        rng = jax.random.PRNGKey(0)
        state = strategy.initialize(rng, es_params)

        best_fitness = float('inf')  # Initialize with a very high value
        patience_counter = 0  # Counter for tracking patience

        for t in range(num_gens):
            rng, rng_ask = jax.random.split(rng)
            # Ask for a set candidates
            x, state = strategy.ask(rng_ask, state, es_params)
            # Evaluate the candidates
            # fitness = jit(jax.vmap(f_train)(x))
            fitness = process_batch(x)
            # fitness = jnp.array([f_train(x_i) for x_i in x])
            # fitness = evaluator.rollout(rng, x)
            # Update the strategy based on fitness
            state = strategy.tell(x, fitness, state, es_params)
            # Update the log with results
            if plot_logging:
                log = es_logging.update(log, x, fitness)
            # if (t + 1) % 5 == 0:
            if config.run.optimization.log_optimization:
                logger.info("{} - # Gen: {}|Fitness: {:.3g}|Params: {} | rng: {}".format(
                    s_name, t+1, state.best_fitness, state.best_member, rng))
                
            # Early stopping check
            if state.best_fitness < best_fitness:
                best_fitness = state.best_fitness
                patience_counter = 0  # Reset counter on improvement
            else:
                patience_counter += 1  # Increment counter if no improvement
            if patience_counter >= config.run.optimization.patience:
                logger.info(f"Early stopping triggered at generation {t+1}")
                break  # Exit the loop if no improvement for 'patience' generations

        if plot_logging:
            es_logging.plot(log, "Function CMA-ES") #, ylims=(0, 30))
            import matplotlib.pyplot as plt
            plt.savefig(f"{s_name}-test.png")

        x0_optimized = state.best_member
        val_loss = f_val(x0_optimized).item()
        train_loss = f_train(x0_optimized).item()
        x0_optimized_unormalized = x0_optimized * x0_stds + x0_means
        x0_optimized_unormalized = jnp.clip(x0_optimized_unormalized, a_min=0, a_max=None)
        optimized_parameters = array_to_dict(x0_optimized_unormalized.tolist(), parameters)
        logger.info(f'Optimizer JAX : Final Train MSE: {f_train(x0_optimized)} | Final Val MSE: {f_val(x0_optimized)}')

        simulated_states = f_with_data_inner(x0_optimized, val_data)
        # Independent losses per dimension
        loss_per_dim = jnp.mean(jnp.square(simulated_states - val_data[0]), axis=(0,1)).tolist()
        return train_loss, val_loss, optimized_parameters, loss_per_dim


    
def simulate(num_patients=256, length=60, chemo_coeff=2.0, radio_coeff=2.0, window_size=15, env_name='', variation='train'):
    env = CancerEnv()
    v, c = env.reset(num_patients=num_patients, env_name=env_name, variation=variation)
    states = []
    actions = []

    cancer_queue = deque(maxlen=window_size)
    D_MAX = calc_diameter(TUMOUR_DEATH_THRESHOLD)
    chemo_sigmoid_intercept = D_MAX / 2.0
    radio_sigmoid_intercept = D_MAX / 2.0
    chemo_sigmoid_beta = chemo_coeff / D_MAX
    radio_sigmoid_beta = radio_coeff / D_MAX
    chemo_application_rvs = np.random.rand(num_patients, length)
    radio_application_rvs = np.random.rand(num_patients, length)

    for t in range(length):
        states.append(np.stack((v,c),axis=1))
        cancer_queue.append(v)
        cancer_diameter_used = calc_diameter(np.stack(cancer_queue, axis=1)).mean(1)
        # 
        chemo_prob = (1.0 / (1.0 + np.exp(-chemo_sigmoid_beta * (cancer_diameter_used - chemo_sigmoid_intercept))))
        radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_beta * (cancer_diameter_used - radio_sigmoid_intercept))))

        if env_name == 'Cancer' or env_name == 'Cancer-ood' or env_name == 'Cancer-iid' or 'Cancer-random' in env_name:
            chemo_dosage = (chemo_application_rvs[:,t] < chemo_prob).astype('float') * env.max_chemo_drug
            radio_dosage = (radio_application_rvs[:,t] < radio_prob).astype('float') * env.max_radio
        elif env_name == 'Cancer-untreated':
            chemo_dosage = np.zeros_like(chemo_prob)
            radio_dosage = np.zeros_like(radio_prob)
        elif env_name == 'Cancer-chemo':
            chemo_dosage = (chemo_application_rvs[:,t] < chemo_prob).astype('float') * env.max_chemo_drug
            radio_dosage = np.zeros_like(radio_prob)

        actions.append(np.stack((chemo_dosage, radio_dosage), axis=1))

        dv_dt, dc_dt = env.state_diff(v, c, chemo_dosage, radio_dosage, env_name=env_name)
        if env_name == 'Cancer-ood' or env_name == 'Cancer-iid':
            v += dv_dt * (1.0/24.0)
            c += dc_dt * (1.0/24.0)
        else:
            v += dv_dt
            c += dc_dt

    states = np.stack(states, axis=1)
    actions = np.stack(actions, axis=1)

    if env_name == 'Cancer-untreated':
        actions = None
        states = states[:,:,0][:,:,np.newaxis]
    elif env_name == 'Cancer-chemo':
        actions = actions[:,:,0][:,:,np.newaxis]

    debug_plot = False
    if debug_plot:
        import matplotlib.pyplot as plt
        patient_idx = -1
        # Create a figure and a set of subplots
        fig, axs = plt.subplots(4, 1, sharex=True)
        # Set the overall title for all subplots
        fig.suptitle(f"{env_name} {variation} {patient_idx}")
        # Plotting each data in a separate subplot
        axs[0].plot(states[patient_idx,:,0], label="Volume")
        axs[0].legend()
        axs[1].plot(states[patient_idx,:,1], label="Concentration")
        axs[1].legend()
        axs[2].plot(actions[patient_idx,:,0], label="Chemo")
        axs[2].legend()
        axs[3].plot(actions[patient_idx,:,1], label="Radio")
        axs[3].legend()
        # Adjust the layout
        plt.tight_layout()
        # Save the figure
        figure_path = f"trajectory_{env_name}_{variation}_{patient_idx}.png"
        plt.savefig(figure_path)
        print(f'./{figure_path}')
    # assert not np.any(np.isnan(states)), "States array contains NaN"
    # assert not np.any(np.isnan(actions)), "Actions array contains NaN"
    return (states, actions)

def load_data(num_patients=1000, config={}, seed=0, env_name=''):
    description = """
Environment Task description:```
A realistic simulator Lung Cancer volume size under the treatment of chemotherapy and radiotherapy.```

You will complete the following function definition of:```
def env_state_diff(cancer_volume, chemo_concentration, chemotherapy_dosage, radiotherapy_dosage, parameters) -> (d_cancer_volume__dt, d_chemo_concentration__dt): 
```
Where `parameters` is a dictionary of parameters that you can use to update the simulation. You can use any parameters you want, but you must use the parameters provided in the function definition. Please recommend good initial parameters to use, at the same time when providing the code. Here you must model the state differential of cancer_volume, and chemo_concentration; with the input actions of chemotherapy_dosage, and radiotherapy_dosage.
"""
    test_set = simulate(num_patients, env_name=env_name, variation='test')
    train_set = simulate(num_patients, env_name=env_name, variation='train')
    val_set = simulate(num_patients, env_name=env_name, variation='val')

    if env_name == 'Cancer-ood' or env_name == 'Cancer-iid':
        print(f'[{env_name}] Training set distribution:\tstart[{train_set[0][:,0,0].min()}, {train_set[0][:,0,0].max()}]\tend[{train_set[0][:,-1,0].min()}, {train_set[0][:,-1,0].max()}]')
        print(f'[{env_name}] Validation set distribution:\tstart[{val_set[0][:,0,0].min()}, {val_set[0][:,0,0].max()}]\tend[{val_set[0][:,-1,0].min()}, {val_set[0][:,-1,0].max()}]')
        print(f'[{env_name}] Test set distribution:\t\tstart[{test_set[0][:,0,0].min()}, {test_set[0][:,0,0].max()}]\tend[{test_set[0][:,-1,0].min()}, {test_set[0][:,-1,0].max()}]')
        # assert train_set[0][:,0,0].max() < test_set[0][:,-1,0].min(), "For OOD test set must be outside the training distribution"

    return train_set, val_set, test_set, description

class TestEnvOptim(unittest.TestCase):
    def setUp(self):
        from hydra import initialize, compose
        self.num_patients = 1000 
        self.train_data = simulate(self.num_patients)
        self.val_data = simulate(self.num_patients)
        self.test_data = simulate(self.num_patients)
        self.env = CancerEnv()
        self.optimizer = "OptunaCmaEsSampler"
        # self.optimizer = "BFGS"
        # Load config from Hydra
        initialize(config_path="../../config", version_base=None)  # Point to your actual config directory        
        self.config = compose(config_name="config.yaml")

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
