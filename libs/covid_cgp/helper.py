import torch
import numpy as np
from sklearn.model_selection import train_test_split

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

def tensor_scaled_dist(l, x):
    # l: d, 1 or N, d, 1 => N, d, 1
    if l.dim() == 2:
        l = l.unsqueeze(0)

    # N, d, 1 => N, 1, 1, d
    l = l.unsqueeze(1).transpose(-1, -2)

    # t, d, p => t, p, d
    x = x.transpose(-1, -2)

    # N, t, p, d => N, d, t, p
    x_scaled = (x / l).permute((0, 3, 1, 2))

    # N, d, t, 1
    x2 = (x_scaled ** 2).sum(-1, keepdim=True)

    # N, d, t, t
    xz = torch.einsum('abij,abjk->abik', x_scaled, x_scaled.transpose(-1, -2))

    # N, d, t, t
    r2 = x2 - 2 * xz + x2.transpose(-1, -2)
    return r2.clamp(min=0)


def tensor_RBF(l, v, x):
    # N, d, t, t
    r2 = torch.exp(-0.5 * tensor_scaled_dist(l, x))
    # d, t, t, N
    r2 = r2.permute((1, 2, 3, 0))

    # v: d, 1 or N, d, 1 => N, d, 1
    if v.dim() == 2:
        v = v.unsqueeze(0)

    # N, d, 1, 1
    v = v.unsqueeze(-1)
    # d, 1, 1, N
    v = v.permute((1, 2, 3, 0))

    res = (v * r2).permute(3, 0, 1, 2)

    return res


def eluer_seir_time(s0, e0, i0, r0, f0, beta_t, sigma, alpha, p_fatal, D_death, case_import, t_init):
    # i0, r0, sigma: D, 1; N, D, 1
    # beta_t: D, T; N, D, T
    # t_init: make sure same shape as i0 and r0 (reshape if necessary)

    # i0 = torch.zeros(500, 2, 1)
    # r0 = torch.zeros(500, 2, 1)
    # e0 = torch.zeros(500, 2, 1) + 500 / N
    # f0 = torch.zeros(500, 2, 1)
    # s0 = 1 - i0 - r0 - e0 - f0
    #
    # #  alpha, p_fatal, D_death, case_import
    #
    # # beta_t = torch.randn(500, 2, 37) * 0.001 + 0.5
    # sigma = torch.zeros(500, 2, 1) + 1. / D_infectious
    # alpha = torch.zeros(500, 2, 1) + 1. / D_incubation
    # p_fatal = torch.zeros(500, 2, 1) + 0.012
    # D_death = torch.zeros(500, 2, 1) + Time_to_death - D_infectious
    # case_import = torch.zeros(500, 2, 1) + 500 / N
    # beta_t = (sigma * uk_r0).to(case_import)
    #
    # t_init = torch.zeros((2, 1))
    # t_init[0, 0] = 0
    # t_init[1, 0] = 5
    #
    # t_init = t_init.unsqueeze(0).repeat(500, 1, 1)
    # s, e, i, r, f = eluer_seir_time(s0, e0, i0, r0, f0, beta_t, sigma, alpha, p_fatal, D_death, case_import, t_init)

    T = beta_t.size(-1)

    s_list = []
    e_list = []
    i_list = []
    r_list = []
    f_list = []

    s_t = torch.zeros_like(s0, dtype=torch.float)
    e_t = torch.zeros_like(e0, dtype=torch.float)
    i_t = torch.zeros_like(i0, dtype=torch.float)
    r_t = torch.zeros_like(r0, dtype=torch.float)
    f_t = torch.zeros_like(f0, dtype=torch.float)

    for t in range(T):
        i_t[t_init > t] = 0.
        r_t[t_init > t] = 0.
        s_t[t_init > t] = 0.
        e_t[t_init > t] = 0.
        f_t[t_init > t] = 0.

        i_t[t_init == t] = i0[t_init == t]
        r_t[t_init == t] = r0[t_init == t]
        s_t[t_init == t] = s0[t_init == t]
        e_t[t_init == t] = e0[t_init == t]
        f_t[t_init == t] = f0[t_init == t]

        i_list.append(i_t)
        r_list.append(r_t)
        s_list.append(s_t)
        e_list.append(e_t)
        f_list.append(f_t)

        dSdt = -beta_t[..., t:t + 1] * s_t * i_t
        dEdt = beta_t[..., t:t + 1] * s_t * i_t - alpha * e_t + case_import
        dIdt = alpha * e_t - sigma * i_t
        dRdt = p_fatal * sigma * i_t - (1 / D_death) * r_t
        dFdt = (1 / D_death) * r_t

        i_t = i_t + dIdt
        r_t = r_t + dRdt
        s_t = s_t + dSdt
        e_t = e_t + dEdt
        f_t = f_t + dFdt

    i = torch.cat(i_list, dim=-1)
    r = torch.cat(r_list, dim=-1)
    s = torch.cat(s_list, dim=-1)
    e = torch.cat(e_list, dim=-1)
    f = torch.cat(f_list, dim=-1)
    return s, e, i, r, f


def eluer_sir_time(i0, r0, beta_t, sigma, t_init):
    # i0, r0, sigma: D, 1; N, D, 1
    # beta_t: D, T; N, D, T
    # t_init: make sure same shape as i0 and r0 (reshape if necessary)

    # test 1
    # i0 = torch.randn(500, 2, 1) * 0.001 + 0.01
    # r0 = torch.randn(500, 2, 1) * 0.001 + 0.01
    # beta_t = torch.randn(500, 2, 37) * 0.001 + 0.5
    # sigma = torch.randn(500, 2, 1) * 0.001 + 0.01
    #
    # t_init = torch.zeros((2, 1))
    # t_init[0, 0] = 0
    # t_init[1, 0] = 5
    #
    # t_init = t_init.unsqueeze(0).repeat(500, 1, 1)

    # test 2
    # i0 = torch.randn(2, 1) * 0.001 + 0.01
    # r0 = torch.randn(2, 1) * 0.001 + 0.01
    # beta_t = torch.randn(2, 37) * 0.001 + 0.5
    # sigma = torch.randn(2, 1) * 0.001 + 0.01
    #
    # t_init = torch.zeros_like(i0)
    # t_init[0, 0] = 0
    # t_init[1, 0] = 5

    # i, r = eluer_sir(i0, r0, beta_t, sigma, t_init)

    T = beta_t.size(-1)

    i_list = []
    r_list = []

    #     i_t = i0.clone()
    #     r_t = r0.clone()

    i_t = torch.zeros_like(i0, dtype=torch.float)
    r_t = torch.zeros_like(r0, dtype=torch.float)

    for t in range(T):
        i_t[t_init > t] = 0.
        r_t[t_init > t] = 0.

        i_t[t_init == t] = i0[t_init == t]
        r_t[t_init == t] = r0[t_init == t]

        i_list.append(i_t)
        r_list.append(r_t)

        delta_1 = beta_t[..., t:t + 1] * i_t * (1. - r_t)
        delta_2 = sigma * i_t

        i_t = i_t + delta_1 - delta_2
        r_t = r_t + delta_1

    i = torch.cat(i_list, dim=-1)
    r = torch.cat(r_list, dim=-1)
    return i, r


def block_diag(*arrs):
    bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
    if bad_args:
        raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args )

    shapes = torch.tensor([a.shape for a in arrs])
    out = torch.zeros(torch.sum(shapes, dim=0).tolist(), dtype=arrs[0].dtype, device=arrs[0].device)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out


def get_covariates(data_dict, train_len, notime=False):
    n_country = len(data_dict['countries'])
    s_ind_train = data_dict['s_index'][:train_len, ...]
    country_feat = data_dict['country_feat'][None, :, :]
    country_feat = country_feat.repeat(train_len, 1, 1)

    if not notime:
        time_feat = torch.arange(train_len).to(s_ind_train).unsqueeze(-1).repeat(1, n_country)
        covariate_stack = torch.stack([time_feat, s_ind_train], dim=-1)
    else:
        covariate_stack = s_ind_train[..., None]
    new_shape = covariate_stack.size(-1) * covariate_stack.size(-2)
    covariate = covariate_stack.view(covariate_stack.size(0), new_shape)
    return covariate


def reshape_covariates_pyro(covariate, n_country):
    p_total = covariate.size(-1)
    covariate_unstack = covariate.view(covariate.size(0), n_country, p_total//n_country)
    return covariate_unstack


def get_Y(data_dict, train_len, daily=False):
    if not daily:
        return data_dict['cum_death'][:train_len]
    else:
        return data_dict['daily_death'][:train_len]

def get_covariates_intervention(data_dict, train_len, notime=False):
    n_country = len(data_dict['countries'])
    i_ind_train = data_dict['i_index'][:train_len, ...]
    country_feat = data_dict['country_feat'][None, :, :]
    country_feat = country_feat.repeat(train_len, 1, 1)

    if not notime:
        time_feat = torch.arange(train_len).to(i_ind_train)[:, None, None].repeat(1, n_country, 1) / 100
        covariate_stack = torch.cat([time_feat, i_ind_train, country_feat], dim=-1)
    else:
        covariate_stack = torch.cat([i_ind_train, country_feat], dim=-1)
    new_shape = covariate_stack.size(-1) * covariate_stack.size(-2)
    covariate = covariate_stack.view(covariate_stack.size(0), new_shape)
    return covariate

def get_covariates_intervention_and_output_tensor(data_dict, train_len, notime=False, daily=False):
    n_country = len(data_dict['countries'])
    i_ind_train = data_dict['i_index'][:train_len, ...]
    country_feat = data_dict['country_feat'][None, :, :]
    country_feat = country_feat.repeat(train_len, 1, 1)
    y = get_Y(data_dict, train_len, daily=daily)

    covariate_stack = torch.cat([i_ind_train, country_feat, y.view(y.shape[0], y.shape[1], 1)], dim=-1)
    new_shape = covariate_stack.size(-1) * covariate_stack.size(-2)
    covariate = covariate_stack.view(covariate_stack.size(0), new_shape)
    return DotDict({'covariate': covariate, 'y': y, 'i_ind_train': i_ind_train, 'country_feat': country_feat})


import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def create_training_dataset(train_data, window_size=14, batch_size=256, train_val_split=0.8, regress_only_on_y=False):
    """
    Create a multi-variate time series dataset for PyTorch with a specified window length.

    Args:
    - covariates_and_y (torch.Tensor): Input tensor of shape (time_steps, num_features),
                                       where time_steps is the number of time steps and num_features is the number of features.
    - y (torch.Tensor): Target tensor of shape (time_steps, output_features),
                        where time_steps is the number of time steps and output_features is the number of output features.
    - window_size (int): Length of the sequence window.
    - batch_size (int): Size of each batch.
    - train_val_split (float): Ratio for splitting the dataset into training and validation sets.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - val_loader (DataLoader): DataLoader for the validation set.
    """

    N, H = train_data.covariate.shape
    _, O = train_data.y.shape

    # Convert to numpy
    covariates_and_y = train_data.covariate.cpu().numpy()
    y = train_data.y.cpu().numpy()

    # Check if y and covariates_and_y have the same number of time steps
    if N != len(y):
        raise ValueError("The number of time steps in covariates_and_y and y must be the same.")

    # Create sequences
    inputs = []
    targets = []
    for i in range(N - window_size):
        if not regress_only_on_y:
            inputs.append(covariates_and_y[i:i + window_size])
        else:
            inputs.append(y[i:i + window_size])
        targets.append(y[i + window_size])

    inputs = np.stack(inputs)
    targets = np.stack(targets)

    # Split sequences into training and validation sets
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        inputs, targets, train_size=train_val_split)
    
    # Convert to PyTorch tensors
    train_inputs = torch.tensor(train_inputs, dtype=torch.float)
    val_inputs = torch.tensor(val_inputs, dtype=torch.float)
    train_targets = torch.tensor(train_targets, dtype=torch.float)
    val_targets = torch.tensor(val_targets, dtype=torch.float)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def evaluate_model_on_test_set(model, test_data, window_size=14, device='cpu', regress_only_on_y=False):
    N, H = test_data.covariate.shape

    outputs = None
    outputs_all = []
    targets_all = []

    for i in range(N - window_size*2, N - window_size):
        with torch.no_grad():
            if outputs is None:
                if regress_only_on_y:
                    inputs = test_data.y[i:i + window_size].unsqueeze(0).float().to(device)
                else:
                    inputs = test_data.covariate[i:i + window_size].unsqueeze(0).float().to(device)
            else:
                outputs_all_t = torch.stack(outputs_all).unsqueeze(-1)
                current_y = torch.concat((test_data.y[i:i + window_size - outputs_all_t.shape[0]].unsqueeze(-1).to(device), outputs_all_t), dim=0)
                if regress_only_on_y:
                    inputs = current_y.squeeze(-1).unsqueeze(0).float().to(device)
                else:
                    # targets_all_t = torch.stack(targets_all).unsqueeze(-1)
                    # true_y = test_data.y[i:i + window_size].unsqueeze(-1).to(device)
                    # assert torch.sum(current_y - true_y) == 0.0
                                        

                    inputs = torch.cat([test_data.i_ind_train[i:i + window_size].to(device),
                                        test_data.country_feat[i:i + window_size].to(device),
                                        current_y], dim=-1)
                    new_shape = inputs.size(-1) * inputs.size(-2)
                    inputs = inputs.view(inputs.size(0), new_shape)
                    inputs = inputs.unsqueeze(0).float()
            outputs = model(inputs).view(-1)
            targets = test_data.y[i + window_size].to(device)

            outputs_all.append(outputs)
            targets_all.append(targets)

    outputs_all_t = torch.stack(outputs_all)
    targets_all_t = torch.stack(targets_all)
    # Compute RMSE
    # rmse = torch.sqrt(torch.mean((outputs_all_t - targets_all_t)**2, dim=0))
    rmse = torch.sqrt(torch.mean((outputs_all_t - targets_all_t) ** 2))
    # compute MAE
    mae = torch.mean(torch.abs(outputs_all_t - targets_all_t))
    # mae = torch.abs(torch.sum(outputs_all_t-targets_all_t)) # Aka off
    # mae = torch.abs(torch.sum(outputs_all_t, dim=0) - torch.sum(targets_all_t, dim=0)) # Aka off
    # rmse_mean = torch.mean(rmse)
    # mae_mean = torch.mean(mae)
    return rmse.item(), mae.item()

def smooth_curve_1d(x):
    w = np.ones(7, 'd')
    y = np.convolve(w / w.sum(), x, mode='valid')
    y = np.concatenate([np.zeros(3), y])
    return y


def smooth_daily(data_dict):
    daily = data_dict['daily_death']

    dy_list = list()
    for i in range(daily.size(1)):
        ds = daily[:, i]
        dy = smooth_curve_1d(ds)
        dy_list.append(dy)

    sy = np.stack(dy_list, axis=-1)
    cum_y = np.cumsum(sy, axis=0)
    new_len = min(cum_y.shape[0], data_dict['i_index'].shape[0])

    return {
        'cum_death': torch.tensor(cum_y)[:new_len, :],
        'daily_death': torch.tensor(sy)[:new_len, :],
        'actual_daily_death': data_dict['daily_death'][:new_len, :],
        'actual_cum_death': data_dict['cum_death'][:new_len, :],
        's_index': data_dict['s_index'][:new_len, :],
        'i_index': data_dict['i_index'][:new_len, :],
        'population': data_dict['population'],
        't_init': data_dict['t_init'],
        'date_list': data_dict['date_list'][:new_len],
        'countries': data_dict['countries'],
        'country_feat': data_dict['country_feat']
    }

