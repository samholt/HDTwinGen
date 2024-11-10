import pandas as pds
# import pyro
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from . import data_loader, helper
# import pyro_model.seir_gp
import argparse
import os
from copy import deepcopy
import shelve

def load_data(days=14, regress_only_on_y = False, load_from_cache=True, force_recache=False):
    countries = [
        'United Kingdom',
        'Italy',
        'Germany',
        'Spain',
        'US',
        'France',
        'Belgium',
        'Korea, South',
        'Brazil',
        'Iran',
        'Netherlands',
        'Canada',
        'Turkey',
        'Romania',
        'Portugal',
        'Sweden',
        'Switzerland',
        'Ireland',
        'Hungary',
        'Denmark',
        'Austria',
        'Mexico',
        'India',
        'Ecuador',
        'Russia',
        'Peru',
        'Indonesia',
        'Poland',
        'Philippines',
        'Japan',
        'Pakistan'
    ]

    ensemble = True
    hidden_size = 32
    num_layers = 3

    niter = 2000
    n_sample = 500
    pad = 24
    batch_size = 512
    # num_epochs = 100
    num_epochs = 100
    learning_rate = 1e-3
    window_size = 14
    train_val_split=0.9
    clip_grad_norm=0.1
    # clip_grad_norm=None
    bidirectional=True
    normalize=True
    patience=float('inf')

    path = f'datasets_{"_".join(countries)}_pad_{pad}'
    if force_recache:
        data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad)
        with shelve.open('datasets') as f:
            f[path] = data_dict
    elif load_from_cache:
        try:
            with shelve.open('datasets') as f:
                data_dict = f[path]
        except KeyError:
            data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad)
            with shelve.open('datasets') as f:
                f[path] = data_dict
    else:
        data_dict = data_loader.get_data_pyro(countries, smart_start=False, pad=pad)

    # data_dict = helper.smooth_daily(data_dict)

    train_len = data_dict['cum_death'].shape[0] - days
    n_country = len(countries)

    # covariates_notime = helper.get_covariates_intervention(data_dict, train_len, notime=True)
    train_data = helper.get_covariates_intervention_and_output_tensor(data_dict, train_len, daily=True)
    Y_train = helper.get_Y(data_dict, train_len, daily=True)

    total_len = len(data_dict['date_list'])
    test_data = helper.get_covariates_intervention_and_output_tensor(data_dict, total_len) # Full data

    train_loader, val_loader = helper.create_training_dataset(train_data, window_size=window_size, batch_size=batch_size, train_val_split=train_val_split, regress_only_on_y=regress_only_on_y)

    description = "Time-series of daily COVID-19 fatalities for a specific country, where the data is from the COVID-19 CSSE data repository from Johns Hopkins University."
    attribute_names = []
    if regress_only_on_y:
        for country in countries:
            attribute_names.append("Daily COVID-19 Deaths in " + country)

    return train_loader, val_loader, train_data, test_data, description, attribute_names