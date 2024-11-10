# Import all pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
WEIGHTS_INITIALIZATION_STD=0.5

import torch
import torch.nn as nn

def replace_zeros_with_ones(tensor):
    """
    Replaces all zeros in the input tensor with ones.

    Parameters:
    tensor (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: A new tensor with zeros replaced by ones.
    """
    return torch.where(tensor == 0, torch.ones_like(tensor), tensor)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, feature_dim=250):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        # Linear layer to project input features to the desired dimension
        self.linear_projection = nn.Linear(in_features=self.input_dim, out_features=self.feature_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        # Reshape x to [-1, input_dim] to apply linear projection, then reshape back
        x_reshaped = x.view(-1, self.input_dim)
        x_projected = self.linear_projection(x_reshaped)
        x_projected = x_projected.view(batch_size, seq_len, self.feature_dim)
        return x_projected

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1, output_size=1, input_size=2):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.input_transformer = TimeSeriesTransformer(input_dim=input_size, feature_dim=feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,output_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        src = self.input_transformer(src)
        src = src.transpose(0, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output)
        return output.transpose(0, 1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransformerModel(nn.Module):
    def __init__(self, obs_dim, action_dim, states_actions_train, hidden_dim=128, model_activation='tanh', model_initialization='xavier'):
        super(TransformerModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        if action_dim is None:
            action_dim = 0

        self.model = TransAm(input_size=obs_dim+action_dim, output_size=obs_dim)

        self.register_buffer("in_mean", torch.tensor(states_actions_train.mean((0,1))))
        self.register_buffer("in_std", replace_zeros_with_ones(torch.tensor(states_actions_train.std((0,1)))))
        self.register_buffer("output_mean", torch.tensor(states_actions_train[:, :, :obs_dim].mean((0,1))))
        self.register_buffer("output_std", replace_zeros_with_ones(torch.tensor(states_actions_train[:, :, :obs_dim].std((0,1)))))


    def forward(self, in_xu):
        in_xu = (in_xu - self.in_mean) / self.in_std
        out = self.model(in_xu)
        x = out * self.output_std + self.output_mean
        return x

class RNNModel(nn.Module):
    def __init__(self, obs_dim, action_dim, states_actions_train, hidden_dim=250, model_activation='tanh', model_initialization='xavier'):
        super(RNNModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        if action_dim is None:
            action_dim = 0

        self.gru = nn.GRU(obs_dim + action_dim, hidden_dim, batch_first=True, num_layers=2)
        self.linear_out = nn.Linear(hidden_dim, obs_dim)

        self.register_buffer("in_mean", torch.tensor(states_actions_train.mean((0,1))))
        self.register_buffer("in_std", replace_zeros_with_ones(torch.tensor(states_actions_train.std((0,1)))))
        self.register_buffer("output_mean", torch.tensor(states_actions_train[:, :, :obs_dim].mean((0,1))))
        self.register_buffer("output_std", replace_zeros_with_ones(torch.tensor(states_actions_train[:, :, :obs_dim].std((0,1)))))
        # print('std of output: ', states_actions_train[:, :, :obs_dim].std((0,1)).mean())
        # print('')


    def forward(self, in_xu):
        in_xu = (in_xu - self.in_mean) / self.in_std
        out, _ = self.gru(in_xu)
        x = self.linear_out(out[:, -1, :])
        x = x * self.output_std + self.output_mean
        return x

class DyNODEModel(nn.Module):
    def __init__(self, obs_dim, action_dim, states_actions_train, hidden_dim=128, model_activation='tanh', model_initialization='xavier'):
        super(DyNODEModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        if model_activation == 'tanh':
            activation = nn.Tanh
        elif model_activation == 'silu':
            activation = nn.SiLU
        elif model_activation == 'ELU':
            activation = nn.ELU
        else:
            raise NotImplementedError
        
        if action_dim is None:
            action_dim = 0

        self.stack = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, obs_dim))

        for m in self.stack.modules():
            if isinstance(m, nn.Linear):
                if model_initialization=='xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif model_initialization=='normal':
                    nn.init.normal_(m.weight, mean=0, std=WEIGHTS_INITIALIZATION_STD)

        # states_actions_mean = states_actions_train.mean((0,1))
        # states_actions_std = states_actions_train.std((0,1))
        # states_actions_std[states_actions_std<=1e-6] = 1.0
        # states_mean = states_actions_train[:,:,:obs_dim].mean((0,1))
        # states_std = states_actions_train[:,:,:obs_dim].std((0,1))
        # states_std[states_std<=1e-6] = 1.0

        # self.register_buffer("states_actions_mean", torch.tensor(states_actions_mean, dtype=torch.float32))
        # self.register_buffer("states_actions_std", torch.tensor(states_actions_std, dtype=torch.float32))
        # self.register_buffer("states_mean", torch.tensor(states_mean, dtype=torch.float32))
        # self.register_buffer("states_std", torch.tensor(states_std, dtype=torch.float32))


    def forward(self, in_xu):
        # xu = (in_xu - self.states_actions_mean) / self.states_actions_std
        x = self.stack(in_xu)
        # x = x * self.states_std + self.states_mean
        return x