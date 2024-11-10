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
from copy import deepcopy
import sciris as sc
import covasim as cv
import shelve



class DotDict(dict):
    def __getattr__(self, name):
        return self.get(name, None)
    
    def __setattr__(self, name, value):
        self[name] = value

def to_dot_dict(d):
    dot_dict = DotDict()
    for key, value in d.items():
        if isinstance(value, dict):
            dot_dict[key] = to_dot_dict(value)
        else:
            dot_dict[key] = value
    return dot_dict

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


class CovidEnv:
    def __init__(self):
        pass

    def reset(self, num_patients=1):
        pass
    
    def evaluate_simulator_code_wrapper(self, StateDifferential, train_data, val_data, test_data, config={}, logger=None, env_name=''):
        if config.run.optimizer == 'pytorch':
            train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss = self.evaluate_simulator_code_using_pytorch(StateDifferential, train_data, val_data, test_data, config=config, logger=logger, env_name=env_name)
        elif 'evotorch' in config.run.optimizer:
            train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss = self.evaluate_simulator_code_using_pytorch_with_neuroevolution(StateDifferential, train_data, val_data, test_data, config=config, logger=logger)
        if env_name == 'COVID':
            loss_per_dim_dict = {'susceptible': loss_per_dim[0], 'exposed': loss_per_dim[1], 'infected': loss_per_dim[2], 'recovered': loss_per_dim[3]}
        return train_loss, val_loss, optimized_parameters, loss_per_dim_dict, test_loss
    
    def evaluate_simulator_code_using_pytorch(self, StateDifferential, train_data, val_data, test_data, config={}, logger=None, env_name=''):
        import torch
        import numpy as np
        device = "cuda:0"
        config.run.pytorch_as_optimizer.batch_size = train_data[0].shape[0] # Will need to change for insight

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
                susceptible, exposed, infected, recovered, total_population = states_train_batch_i[:,t,0], states_train_batch_i[:,t,1], states_train_batch_i[:,t,2], states_train_batch_i[:,t,3], actions_train_batch_i[:,t,0]
                dx_dt = model(susceptible, exposed, infected, recovered, total_population)
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
                    susceptible, exposed, infected, recovered, total_population = states[:,t,0], states[:,t,1], states[:,t,2], states[:,t,3], actions[:,t,0]
                    dx_dt = model(susceptible, exposed, infected, recovered, total_population)
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


        if env_name == 'COVID-insight':
            model_intervened = deepcopy(f_model)
            with torch.no_grad():
                # model_intervened.beta.data = model_intervened.beta.data * 0.25
                model_intervened.beta.data = model_intervened.beta.data * 0.30

            noint_states, noint_actions = states_val, actions_val
            int_states, int_actions = states_test, actions_test

            model_intervened.eval()


            pred_states = []
            # pred_sates_per_dim_per_bb = []
            pred_state = int_states[:,19]
            for t in range(19, int_states.shape[1]):
                pred_states.append(pred_state)
                susceptible, exposed, infected, recovered, total_population = int_states[:,t,0], int_states[:,t,1], int_states[:,t,2], int_states[:,t,3], int_actions[:,t,0]
                dx_dt = model_intervened(susceptible, exposed, infected, recovered, total_population)
                dx_dt = torch.stack(dx_dt, dim=-1)
                # pred_state = int_states[:,t] + dx_dt
                pred_state = pred_state + dx_dt
            pred_states = torch.stack(pred_states, dim=1)

            predicted_int_states = torch.concat((int_states[:,:19,:],pred_states),dim=1).detach()

            pred_states = []
            # pred_sates_per_dim_per_bb = []
            pred_state = int_states[:,19]
            for t in range(19, int_states.shape[1]):
                pred_states.append(pred_state)
                susceptible, exposed, infected, recovered, total_population = int_states[:,t,0], int_states[:,t,1], int_states[:,t,2], int_states[:,t,3], int_actions[:,t,0]
                dx_dt = f_model(susceptible, exposed, infected, recovered, total_population)
                dx_dt = torch.stack(dx_dt, dim=-1)
                # pred_state = int_states[:,t] + dx_dt
                pred_state = pred_state + dx_dt
            pred_states = torch.stack(pred_states, dim=1)

            predicted_full_states = torch.concat((int_states[:,:19,:],pred_states),dim=1).detach()

            noint_states = noint_states.detach().cpu().numpy()
            int_states = int_states.detach().cpu().numpy()
            nsdt_predicted_int_states = predicted_int_states.cpu().numpy()
            nsdt_predicted_full_states = predicted_full_states.cpu().numpy()

            np.savez('COVID-intervention.npz',
                noint_states=noint_states,
                int_states=int_states,
                nsdt_predicted_int_states=nsdt_predicted_int_states,
                nsdt_predicted_full_states=nsdt_predicted_full_states,
            )




            import matplotlib.pyplot as plt
            # 6, 7?, 9!; 20# with rejig
            # traj_idx = 20
            traj_idx = 9
            # Create a figure and a set of subplots
            fig, axs = plt.subplots(4, 1, sharex=True)
            # Set the overall title for all subplots
            fig.suptitle(f"SEIRD for traj {traj_idx}")
            # Plotting each data in a separate subplot
            axs[0].plot(noint_states[traj_idx,:,0].cpu(), label="Not Int")
            axs[0].plot(int_states[traj_idx,:,0].cpu(), label="Int")
            axs[0].plot(predicted_int_states[traj_idx,:,0].cpu(), label="NSDT-intervention")
            # axs[0].set_ylim((0.0, 1.0))
            # axs[0].plot(predicted_full_states[traj_idx,:,0].cpu(), label="NSDT")
            axs[0].legend()
            axs[1].plot(noint_states[traj_idx,:,1].cpu(), label="Not Int")
            axs[1].plot(int_states[traj_idx,:,1].cpu(), label="Int")
            axs[1].plot(predicted_int_states[traj_idx,:,1].cpu(), label="NSDT-intervention")
            # axs[1].plot(predicted_full_states[traj_idx,:,1].cpu(), label="NSDT")
            axs[1].legend()
            axs[2].plot(noint_states[traj_idx,:,2].cpu(), label="Not Int")
            axs[2].plot(int_states[traj_idx,:,2].cpu(), label="Int")
            axs[2].plot(predicted_int_states[traj_idx,:,2].cpu(), label="NSDT-intervention")
            # axs[2].plot(predicted_full_states[traj_idx,:,2].cpu(), label="NSDT")
            axs[2].legend()
            axs[3].plot(noint_states[traj_idx,:,3].cpu(), label="Not Int")
            axs[3].plot(int_states[traj_idx,:,3].cpu(), label="Int")
            axs[3].plot(predicted_int_states[traj_idx,:,3].cpu(), label="NSDT-intervention")
            # axs[3].plot(predicted_full_states[traj_idx,:,3].cpu(), label="NSDT")
            axs[3].legend()
            # axs[4].plot(states[traj_idx,:,4], label="Dead")
            # axs[4].legend()
            # axs[4].plot(actions[traj_idx,:,0], label="Intervention")
            # axs[4].legend()
            # Adjust the layout
            plt.tight_layout()
            # Save the figure
            plt.savefig("test.png")
            # plt.clf()
            print('')

            print('')
            raise NotImplementedError

        return train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss

def run_simulation_intervention_middle(args, sim_config, verbose=0):
    seed, infected = args
    params = deepcopy(sim_config)
    params['rand_seed'] = seed
    params['pop_infected'] = infected
    # intervention_day = int(params['n_days'] / 2) + 10
    intervention_day = 30
    social_distance_beta_change = 0.25
    params['interventions'] = cv.change_beta(days=intervention_day, changes=social_distance_beta_change)
    sim = cv.Sim(pars=params)
    sim.initialize()
    sim.run(verbose=verbose)

    S = sim.results.n_susceptible.values
    E = sim.results.n_exposed.values - sim.results.n_infectious.values
    I = sim.results.n_infectious.values
    R = sim.results.n_recovered.values + sim.results.n_dead.values
    # D = sim.results.n_dead.values

    return np.stack((S, E, I, R), axis=-1)

def run_simulation_intervention_middle_null(args, sim_config, verbose=0):
    seed, infected = args
    params = deepcopy(sim_config)
    params['rand_seed'] = seed
    params['pop_infected'] = infected
    # intervention_day = int(params['n_days'] / 2) + 10
    intervention_day = 100
    social_distance_beta_change = 0.25
    params['interventions'] = cv.change_beta(days=intervention_day, changes=social_distance_beta_change)
    sim = cv.Sim(pars=params)
    sim.initialize()
    sim.run(verbose=verbose)

    S = sim.results.n_susceptible.values
    E = sim.results.n_exposed.values - sim.results.n_infectious.values
    I = sim.results.n_infectious.values
    R = sim.results.n_recovered.values + sim.results.n_dead.values
    # D = sim.results.n_dead.values

    return np.stack((S, E, I, R), axis=-1)

def run_simulation(args, sim_config, verbose=0):
    seed, infected = args
    params = deepcopy(sim_config)
    params['rand_seed'] = seed
    params['pop_infected'] = infected
    sim = cv.Sim(pars=params)
    sim.initialize()
    sim.run(verbose=verbose)

    S = sim.results.n_susceptible.values
    E = sim.results.n_exposed.values - sim.results.n_infectious.values
    I = sim.results.n_infectious.values
    R = sim.results.n_recovered.values + sim.results.n_dead.values
    # D = sim.results.n_dead.values

    return np.stack((S, E, I, R), axis=-1)


def simulate(num_trajectories=20, length=60, pop_size=1e6, num_infected=10000, multi_process=True, name='COVID-social-distancing', length_offset=10):

    sim_config = {
        'pop_size': pop_size,
        # 'pop_infected': num_infected,
        'n_days': length-1 + length_offset,
        'use_waning': False,
        'pop_type': 'random',
        'rescale': False,
    }

    t0 = time.perf_counter()
    seeds = np.random.randint(0, 2**32-1, size=num_trajectories)
    num_infecteds = np.random.randint(10000, 100000, size=num_trajectories)
    args_in = [(seed, infected) for seed, infected in zip(seeds, num_infecteds)]

    # if name == 'COVID-social-distancing':
    #     sampler = partial(run_simulation_with_social_dist_intervention, sim_config=sim_config, verbose=0, social_distance_beta_change=0.25)
    #     intervention_days = np.random.randint(1, length-1, size=num_trajectories)
    #     args_in = [(seed, intervention_day) for seed, intervention_day in zip(seeds, intervention_days)]
    #     actions = np.ones((num_trajectories, length, 1))
    #     for idx, day in tqdm(enumerate(intervention_days)):
    #         actions[idx, :(day-1), :] = 0
    # elif name == 'COVID-schools-and-social-distancing':
    #     sampler = partial(run_simulation_with_school_and_social_dist_intervention, sim_config=sim_config, verbose=0, social_distance_beta_change=0.25, school_beta_change=0.90)
    #     social_dist_intervention_days = np.random.randint(1, length-1, size=num_trajectories)
    #     school_intervention_days = np.random.randint(1, length-1, size=num_trajectories)
    #     args_in = [(seed, social_dist_intervention_day, school_intervention_day) for seed, social_dist_intervention_day, school_intervention_day in zip(seeds, social_dist_intervention_days, school_intervention_days)]

    #     social_dist_actions = np.ones((num_trajectories, length, 1))
    #     for idx, day in tqdm(enumerate(social_dist_intervention_days)):
    #         social_dist_actions[idx, :(day-1), :] = 0
    #     school_actions = np.ones((num_trajectories, length, 1))
    #     for idx, day in tqdm(enumerate(school_intervention_days)):
    #         school_actions[idx, :(day-1), :] = 0
    #     actions = np.concatenate((social_dist_actions, school_actions), axis=-1)
    # else:
    sampler = partial(run_simulation, sim_config=sim_config, verbose=0)
    actions = np.ones((num_trajectories, length, 1)) * pop_size

    if multi_process:
        pool_outer = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    states = []
    if multi_process:
        for i, result in tqdm(enumerate(pool_outer.imap_unordered(sampler, args_in)), total=num_trajectories, smoothing=0):
            states.append(result)
        pool_outer.close()
    else:
        for args in tqdm(args_in, total=seeds.shape[0], smoothing=0):
            states.append(sampler(args))

    states = np.stack(states, axis=0)
    states = states[:, length_offset:, :]
    states = states / pop_size
    print(f"Simulator run time: {time.perf_counter() - t0:.2f}s")
    print('')

    debug_plot = False
    if debug_plot:
        import matplotlib.pyplot as plt
        traj_idx = -1
        # Create a figure and a set of subplots
        fig, axs = plt.subplots(5, 1, sharex=True)
        # Set the overall title for all subplots
        fig.suptitle(f"SEIRD for traj {traj_idx}")
        # Plotting each data in a separate subplot
        axs[0].plot(states[traj_idx,:,0], label="Susceptible")
        axs[0].legend()
        axs[1].plot(states[traj_idx,:,1], label="Exposed")
        axs[1].legend()
        axs[2].plot(states[traj_idx,:,2], label="Infectious")
        axs[2].legend()
        axs[3].plot(states[traj_idx,:,3], label="Recovered")
        axs[3].legend()
        # axs[4].plot(states[traj_idx,:,4], label="Dead")
        # axs[4].legend()
        axs[4].plot(actions[traj_idx,:,0], label="Intervention")
        axs[4].legend()
        # Adjust the layout
        plt.tight_layout()
        # Save the figure
        plt.savefig("test.png")
    assert not np.any(np.isnan(states)), "States array contains NaN"
    # assert not np.any(np.isnan(actions)), "Actions array contains NaN"
    return (states, actions)


def simulate_insight_intervention(num_trajectories=20, length=60, pop_size=1e6, num_infected=10000, multi_process=True, name='COVID-social-distancing', length_offset=10):

    sim_config = {
        'pop_size': pop_size,
        # 'pop_infected': num_infected,
        'n_days': length-1 + length_offset,
        'use_waning': False,
        'pop_type': 'random',
        'rescale': False,
    }

    t0 = time.perf_counter()
    seeds = np.random.randint(0, 2**32-1, size=num_trajectories)
    num_infecteds = np.random.randint(10000, 100000, size=num_trajectories)
    args_in = [(seed, infected) for seed, infected in zip(seeds, num_infecteds)]

    val_seeds = np.random.randint(0, 2**32-1, size=num_trajectories)
    val_num_infecteds = np.random.randint(10000, 100000, size=num_trajectories)
    val_args_in = [(seed, infected) for seed, infected in zip(seeds, num_infecteds)]
    
    def collect_data(args, fn, sim_config):
        sampler = partial(fn, sim_config=sim_config, verbose=0)
        actions = np.ones((num_trajectories, length, 1)) * pop_size

        if multi_process:
            pool_outer = multiprocessing.Pool(multiprocessing.cpu_count() // 2 - 1)
        states = []
        if multi_process:
            for i, result in tqdm(enumerate(pool_outer.imap_unordered(sampler, args_in)), total=num_trajectories, smoothing=0):
                states.append(result)
            pool_outer.close()
        else:
            for args in tqdm(args_in, total=seeds.shape[0], smoothing=0):
                states.append(sampler(args))

        states = np.stack(states, axis=0)
        states = states[:, length_offset:, :]
        states = states / pop_size
        return states, actions
    
    # # # Train
    train_states, train_actions = collect_data(args_in, run_simulation_intervention_middle_null, sim_config)
    # # # Val
    # val_states, val_actions = collect_data(val_args_in, run_simulation_intervention_middle_null, sim_config)
    # Intervention
    noint_states, noint_actions = collect_data(val_args_in, run_simulation_intervention_middle_null, sim_config)
    # No intervention
    int_states, int_actions = collect_data(val_args_in, run_simulation_intervention_middle, sim_config)
    # true_test_states, true_test_actions = collect_data(args_in, run_simulation_intervention_middle_null, sim_config)

    train_set = (train_states, train_actions)
    val_set = (noint_states, noint_actions)
    test_set = (int_states, int_actions)
    
    print(f"Simulator run time: {time.perf_counter() - t0:.2f}s")
    print('')

    debug_plot = True
    if debug_plot:
        import matplotlib.pyplot as plt
        traj_idx = -1
        # Create a figure and a set of subplots
        fig, axs = plt.subplots(5, 1, sharex=True)
        # Set the overall title for all subplots
        fig.suptitle(f"SEIRD for traj {traj_idx}")
        # Plotting each data in a separate subplot
        axs[0].plot(noint_states[traj_idx,:,0], label="Not Int")
        axs[0].plot(int_states[traj_idx,:,0], label="Int")
        axs[0].legend()
        axs[1].plot(noint_states[traj_idx,:,1], label="Not Int")
        axs[1].plot(int_states[traj_idx,:,1], label="Int")
        axs[1].legend()
        axs[2].plot(noint_states[traj_idx,:,2], label="Not Int")
        axs[2].plot(int_states[traj_idx,:,2], label="Int")
        axs[2].legend()
        axs[3].plot(noint_states[traj_idx,:,3], label="Not Int")
        axs[3].plot(int_states[traj_idx,:,3], label="Int")
        axs[3].legend()
        # axs[4].plot(states[traj_idx,:,4], label="Dead")
        # axs[4].legend()
        # axs[4].plot(actions[traj_idx,:,0], label="Intervention")
        # axs[4].legend()
        # Adjust the layout
        plt.tight_layout()
        # Save the figure
        plt.savefig("test.png")
        print('')
    # assert not np.any(np.isnan(states)), "States array contains NaN"
    # assert not np.any(np.isnan(actions)), "Actions array contains NaN"
    return train_set, val_set, test_set

def load_data_inner(num_trajectories=20, name='', config={}, logger=None):
    description = """
TODO
"""
    num_trajectories_input = 24 # Set to be different from rest of envs
    if name == 'COVID':
        train_set = simulate(num_trajectories_input, name=name)
        val_set = simulate(num_trajectories_input, name=name)
        test_set = simulate(num_trajectories_input, name=name)
    elif name == 'COVID-insight':
        train_set, val_set, test_set = simulate_insight_intervention(num_trajectories_input, name=name)
        # method = 'NSDT'
        method = ''
        if method == 'NSDT':
            import torch
            import torch.nn as nn
            from typing import Tuple

            class StateDifferential(nn.Module):
                def __init__(self):
                    super(StateDifferential, self).__init__()
                    # Define the parameters for the SEIR model
                    self.beta = nn.Parameter(torch.tensor(0.261))  # Transmission rate
                    self.sigma = nn.Parameter(torch.tensor(0.237))  # Incubation rate
                    self.gamma = nn.Parameter(torch.tensor(0.105))  # Recovery rate

                    # Black box component for residuals
                    self.residual_mlp = nn.Sequential(
                        nn.Linear(4, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 4)
                    )

                def forward(self, susceptible: torch.Tensor, exposed: torch.Tensor, infected: torch.Tensor, recovered: torch.Tensor, total_population: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    # SEIR model differential equations
                    d_susceptible__dt = -self.beta * susceptible * infected
                    d_exposed__dt = self.beta * susceptible * infected - self.sigma * exposed
                    d_infected__dt = self.sigma * exposed - self.gamma * infected
                    d_recovered__dt = self.gamma * infected

                    # Calculate residuals using the black box MLP
                    residuals = self.residual_mlp(torch.stack((susceptible, exposed, infected, recovered), dim=1))

                    # Add residuals to the white box model predictions
                    d_susceptible__dt += residuals[:, 0]
                    d_exposed__dt += residuals[:, 1]
                    d_infected__dt += residuals[:, 2]
                    d_recovered__dt += residuals[:, 3]

                    return (d_susceptible__dt, d_exposed__dt, d_infected__dt, d_recovered__dt)

            gt_env = CovidEnv()
            train_loss, val_loss, optimized_parameters, loss_per_dim, test_loss = gt_env.evaluate_simulator_code_using_pytorch(StateDifferential, train_set, val_set, test_set, config=config, logger=logger, env_name=name)
            print('')
            raise NotImplementedError

    return train_set, val_set, test_set, description


def load_data(num_trajectories=20, name='', seed=0, force_recache=False, load_from_cache=True, config={}, logger=None):
    num_trajectories = 24
    path = f'datasets_{name}_trajectories_{num_trajectories}_seed_{seed}'
    if force_recache:
        data = load_data_inner(num_trajectories=num_trajectories, name=name, config=config, logger=logger)
        with shelve.open('datasets') as f:
            f[path] = data
    elif load_from_cache:
        try:
            with shelve.open('datasets') as f:
                data = f[path]
        except KeyError:
            data = load_data_inner(num_trajectories=num_trajectories, name=name, config=config, logger=logger)
            with shelve.open('datasets') as f:
                f[path] = data
    else:
        data = load_data_inner(num_trajectories=num_trajectories, name=name, config=config, logger=logger)
    return data



class TestEnvOptim(unittest.TestCase):
    def setUp(self):
        self.num_trajectories = 24
        # self.num_trajectories = 1000
        # self.name = 'COVID-social-distancing'
        # self.name = 'COVID-schools-and-social-distancing'
        self.name = 'COVID'
        self.train_data, self.val_data, self.test_data, self.description = load_data(num_trajectories=self.num_trajectories, name=self.name, seed=99, force_recache=False, load_from_cache=True)
        self.env = CovidEnv()
        self.config = to_dot_dict({'run': {'optimizer': 'evosax-CMA_ES', 'optimization': {'log_optimization': True, 'patience': 10}}})
        self.optimizer = "OptunaCmaEsSampler"
        self.logger = to_dot_dict({'info': print})
        # self.optimizer = "BFGS"

    def test_optim_0_COVID(self):
        if self.name == 'COVID-schools-and-social-distancing':
            assert 'COVID-schools-and-social-distancing' == self.name
            def d_state__dt(susceptible, exposed, infected, recovered, deceased, social_distancing_intervention_active, school_closure_intervention_active, parameters):
                # Unpack parameters
                beta = parameters['beta']
                gamma = parameters['gamma']
                sigma = parameters['sigma']
                mu = parameters['mu']
                intervention_effectiveness = parameters['intervention_effectiveness']
                intervention_school_effectiveness = parameters['intervention_school_effectiveness']

                # Calculate the effective transmission rate
                effective_beta = beta * (1 - social_distancing_intervention_active * intervention_effectiveness) * (1 - school_closure_intervention_active * intervention_school_effectiveness)

                # SEIR model differential equations
                d_susceptible__dt = -effective_beta * susceptible * infected
                d_exposed__dt = effective_beta * susceptible * infected - sigma * exposed
                d_infected__dt = sigma * exposed - gamma * infected - mu * infected
                d_recovered__dt = gamma * infected
                d_deceased__dt = mu * infected

                return d_susceptible__dt, d_exposed__dt, d_infected__dt, d_recovered__dt, d_deceased__dt
            parameters = {'beta': (0.3, 0.1), 'gamma': (0.1, 0.05), 'sigma': (0.2, 0.1), 'mu': (0.01, 0.005), 'intervention_effectiveness': (0.5, 0.2), 'intervention_school_effectiveness': (0.3, 0.1)}
            # Logic goes here
            train_loss, val_loss, optimized_parameters = self.env.evaluate_simulator_code_using_jax(d_state__dt, parameters, self.train_data, self.val_data, env_name=self.name, config=self.config, logger=self.logger)
            print(f'Optimizer {self.optimizer} : Final Train MSE: {train_loss} | Final Val MSE: {val_loss}') # According to code it is 2694.2922 -- suspect data leakage error
            print(f'Optimized parameters: {optimized_parameters}')
            assert val_loss < 12.3232 * 2.0, "Val loss is too high"
            print('')
        elif self.name == 'COVID':
            assert 'COVID' == self.name
            def d_state__dt(susceptible, exposed, infected, recovered, deceased, total_population, parameters):
                beta = parameters['beta']
                gamma = parameters['gamma']
                sigma = parameters['sigma']
                mu = parameters['mu']
                alpha = parameters['alpha']
                
                # New exposures
                d_exposed__dt = beta * susceptible * infected / total_population
                
                # Progression from exposed to infected
                d_infected__dt = sigma * exposed
                
                # Recovery or death
                d_recovered__dt = gamma * infected
                d_deceased__dt = mu * infected
                
                # Update susceptible
                d_susceptible__dt = -d_exposed__dt
                
                # Ensure that the deceased do not recover
                d_recovered__dt -= alpha * d_deceased__dt
                
                return d_susceptible__dt, d_exposed__dt, d_infected__dt, d_recovered__dt, d_deceased__dt
            parameters = {'beta': (0.3, 0.1), 'gamma': (0.1, 0.05), 'sigma': (0.2, 0.1), 'mu': (0.01, 0.005), 'alpha': (0.02, 0.01)}
            # Logic goes here
            train_loss, val_loss, optimized_parameters, loss_per_dim = self.env.evaluate_simulator_code_using_jax(d_state__dt, parameters, self.train_data, self.val_data, env_name=self.name, config=self.config, logger=self.logger)
            print(f'Loss per dim: {loss_per_dim}')
            print(f'Optimizer {self.optimizer} : Final Train MSE: {train_loss} | Final Val MSE: {val_loss}') # According to code it is 2694.2922 -- suspect data leakage error
            print(f'Optimized parameters: {optimized_parameters}')
            assert val_loss < 12.3232 * 2.0, "Val loss is too high"
            print('')


    # def test_optim_0_COVID(self):
    #     assert 'COVID' == self.name
    #     def d_state__dt(susceptible, exposed, infected, recovered, deceased, parameters):
    #         # Unpack parameters
    #         beta = parameters['beta'] # Transmission rate
    #         gamma = parameters['gamma'] # Recovery rate
    #         sigma = parameters['sigma'] # Progression rate from exposed to infected
    #         mu = parameters['mu'] # Death rate
            
    #         # SEIRD model differential equations
    #         d_susceptible__dt = -beta * susceptible * infected
    #         d_exposed__dt = beta * susceptible * infected - sigma * exposed
    #         d_infected__dt = sigma * exposed - gamma * infected - mu * infected
    #         d_recovered__dt = gamma * infected
    #         d_deceased__dt = mu * infected
            
    #         return d_susceptible__dt, d_exposed__dt, d_infected__dt, d_recovered__dt, d_deceased__dt
    #     parameters = {'beta': (0.3, 0.1), 'gamma': (0.1, 0.05), 'sigma': (0.2, 0.1), 'mu': (0.01, 0.005)}
    #     # Logic goes here



if __name__ == "__main__":
    test = TestEnvOptim()
    test.setUp()
    test.test_optim_0_COVID()