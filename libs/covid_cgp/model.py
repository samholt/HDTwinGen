import pandas as pds
# import pyro
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import data_loader, helper
# import pyro_model.seir_gp
import argparse
import os
from copy import deepcopy

from torch import multiprocessing
import logging
from omegaconf import DictConfig

def generate_log_file_path(file, log_folder='logs', config = {}):
    import os, time, logging
    file_name = os.path.basename(os.path.realpath(file)).split('.py')[0]
    from pathlib import Path
    Path(f"./{log_folder}").mkdir(parents=True, exist_ok=True)
    path_run_name = '{}-{}'.format(file_name, time.strftime("%Y%m%d-%H%M%S"))
    # return f"{log_folder}/{path_run_name}_{'-'.join(config.setup.methods_to_evaluate)}_{'-'.join(config.setup.envs_to_evaluate)}_{config.setup.seed_start}_{config.setup.seed_runs}-runs_log.txt"
    return f"{log_folder}/{path_run_name}.txt"

def create_logger_in_process(log_file_path):
    logger = multiprocessing.get_logger()
    if not logger.hasHandlers():
        formatter = logging.Formatter("%(processName)s| %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file_path)
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    return logger


log_path = generate_log_file_path(__file__, log_folder='logs')
logger = create_logger_in_process(log_path)


parser = argparse.ArgumentParser('CGP')
parser.add_argument('--days', type=str, default='14')
args = parser.parse_args()
days = int(args.days)

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

ensemble = False
if ensemble:
    hidden_size = 32
else:
    hidden_size = 128
regress_only_on_y = False
hidden_size = 23
num_layers = 3

niter = 2000
n_sample = 500
pad = 24
batch_size = 512
# num_epochs = 100
num_epochs = 10000
learning_rate = 1e-4
window_size = 14
train_val_split=0.9
clip_grad_norm=0.1
# clip_grad_norm=None
bidirectional=True
normalize=True
patience=float('inf')

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

logger.info('Generate test dataset')
if normalize:
    feature_mean = train_data.covariate.mean(0).float()
    feature_std = train_data.covariate.std(0).float()
    output_mean = train_data.y.mean(0).float()
    output_std = train_data.y.std(0).float()
    if regress_only_on_y:
        feature_mean = output_mean
        feature_std = output_std
    normalization_dict = {'feature_mean': feature_mean, 'feature_std': feature_std, 'output_mean': output_mean, 'output_std': output_std}
else:
    normalization_dict = None

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, bidirectional=True, normalization_dict=None):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        if normalization_dict is not None:
            self.register_buffer('feature_mean', normalization_dict['feature_mean'])
            self.register_buffer('feature_std', normalization_dict['feature_std'])
            self.register_buffer('output_mean', normalization_dict['output_mean'])
            self.register_buffer('output_std', normalization_dict['output_std'])
            self.normalize = True
        else:
            self.normalize = False
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        assert len(x.shape) == 3, "Input must be 3D"
        if self.normalize:
            feature_std = self.feature_std
            feature_std[feature_std == 0] = 1
            x = (x - self.feature_mean) / feature_std
            x[:, :, feature_std == 1] = 0
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        if self.normalize:
            out = out * self.output_std + self.output_mean
        return out
    
# class LayerNormGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, bidirectional=False):
#         super(LayerNormGRU, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
#         self.ln = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)

#     def forward(self, x):
#         x, h = self.gru(x)
#         return self.ln(x), h

# class BiGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=3, bidirectional=True):
#         super(BiGRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional

#         # Layer Normalized GRU layers
#         hidden_size = hidden_size * 2 if bidirectional else hidden_size
#         self.gru_layers = nn.ModuleList([LayerNormGRU(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        
#         # Attention Layer
#         self.attention = nn.Linear(hidden_size, 1)

#         # Dropout for regularization
#         self.dropout = nn.Dropout(0.2)

#         # Advanced output layers
#         self.fc1 = nn.Linear(hidden_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)

#         # Initialize weights
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)

#     def forward(self, x):
#         assert len(x.shape) == 3, "Input must be 3D"
#         for gru_layer in self.gru_layers:
#             x, _ = gru_layer(x)
#             x = self.dropout(x)
        
#         # Apply attention
#         attention_weights = F.softmax(self.attention(x), dim=1)
#         x = torch.sum(attention_weights * x, dim=1)

#         # Advanced output layers
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x
    
class EnsembleOfBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, bidirectional=True, num_models=31, normalization_dict=None):
        super(EnsembleOfBiGRU, self).__init__()
        self.num_models = num_models
        self.num_countries = num_models
        self.models = nn.ModuleList([BiGRU(input_size, hidden_size, output_size, num_layers, bidirectional) for _ in range(num_models)])
        if normalization_dict is not None:
            self.register_buffer('feature_mean', normalization_dict['feature_mean'])
            self.register_buffer('feature_std', normalization_dict['feature_std'])
            self.register_buffer('output_mean', normalization_dict['output_mean'])
            self.register_buffer('output_std', normalization_dict['output_std'])
            self.normalize = True
        else:
            self.normalize = False
    
    def forward(self, x):
        assert len(x.shape) == 3, "Input must be 3D"
        if self.normalize:
            feature_std = self.feature_std
            feature_std[feature_std == 0] = 1
            x = (x - self.feature_mean) / feature_std
            x[:, :, feature_std == 1] = 0
        xi = x.view(x.shape[0], x.shape[1], self.num_countries, -1)
        outs = [model(xi[:,:,i,:]) for i, model in enumerate(self.models)]
        output = torch.concat(outs, dim=1)
        if self.normalize:
            output = output * self.output_std + self.output_mean
        return output


def train(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, test_data, ensemble=False, clip_grad_norm=None, patience = float("inf")):
    model.to(device)
    best_val_loss = float("inf")
    waiting = 0
    best_model = None
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()            
            if ensemble:
                outputs = model(inputs)
                # loss = criterion(outputs, targets)
                # loss.backward()
                losses = [criterion(outputs[:,i].view(-1,1), targets[:,i].view(-1,1)) for i in range (outputs.shape[1])]
                [loss.backward(retain_graph=True) for loss in losses]
                loss = torch.mean(torch.stack(losses))
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_loader)
        
        rmse_mean, mae_mean = helper.evaluate_model_on_test_set(model, test_data, window_size=14, device=device, regress_only_on_y=regress_only_on_y)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f} | [TEST] RMSE: {rmse_mean:.4f}, MAE: {mae_mean:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model.state_dict())
            # torch.save(model.state_dict(), 'best_model.pth')
            waiting = 0
        elif waiting > patience:
            logger.info(f"Early stopping, breaking out...")
            break
        else:
            waiting += 1

    best_model = deepcopy(model.state_dict())
    if best_model is not None:
        model.load_state_dict(best_model)
    torch.save(model.state_dict(), 'best_model.pth')
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = BiGRU(input_size, hidden_size, output_size)

if ensemble:
    input_size = 31 if not regress_only_on_y else 1 # Number of features in covariates_and_y
    # hidden_size = input_size * 3  # Hyperparameter, can be adjusted
    # hidden_size = 128  # Hyperparameter, can be adjusted
    output_size = 1  # Number of output features
    model = EnsembleOfBiGRU(input_size, hidden_size, output_size, num_models=31, normalization_dict=normalization_dict, bidirectional=bidirectional, num_layers=num_layers)
else:
    input_size = 961 if not regress_only_on_y else 31  # Number of features in covariates_and_y
    # hidden_size = input_size * 3  # Hyperparameter, can be adjusted
    # hidden_size = 128  # Hyperparameter, can be adjusted
    output_size = 31  # Number of output features
    model = BiGRU(input_size, hidden_size, output_size, normalization_dict=normalization_dict, bidirectional=bidirectional, num_layers=num_layers)
criterion = nn.MSELoss()  # Example loss function, choose as per your task
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train_opt = torch.compile(train)
model_number_of_parameters = sum(p.numel() for p in model.parameters())
logger.info(f'Number of parameters: {model_number_of_parameters:,}')
logger.info(f'Ensemble: {ensemble}, Regress only on y: {regress_only_on_y}, Hidden size: {hidden_size}, Num layers: {num_layers}, Bidirectional: {bidirectional}, Normalize: {normalize}, Patience: {patience}, Clip grad norm: {clip_grad_norm}, Learning rate: {learning_rate}, Window size: {window_size}, Train val split: {train_val_split}, Batch size: {batch_size}, Num epochs: {num_epochs}, Days: {days}')

model = train(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, test_data, ensemble=ensemble, clip_grad_norm=clip_grad_norm, patience=patience)
# Evaluate model on test set

# Compute 14 day auto regressive forecast
rmse_mean, mae_mean = helper.evaluate_model_on_test_set(model, test_data, window_size=window_size, device=device, regress_only_on_y=regress_only_on_y)
logger.info(f"[TEST] RMSE: {rmse_mean:.4f}, MAE: {mae_mean:.4f}")
logger.info(f'Number of parameters: {model_number_of_parameters:,}')
logger.info(f'Ensemble: {ensemble}, Regress only on y: {regress_only_on_y}, Hidden size: {hidden_size}, Num layers: {num_layers}, Bidirectional: {bidirectional}, Normalize: {normalize}, Patience: {patience}, Clip grad norm: {clip_grad_norm}, Learning rate: {learning_rate}, Window size: {window_size}, Train val split: {train_val_split}, Batch size: {batch_size}, Num epochs: {num_epochs}, Days: {days}')

