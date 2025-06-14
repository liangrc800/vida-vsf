# Copyright 2022 Google LLC
# Copyright (c) 2020 Zonghan Wu

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" Primary utilities """
import pickle
import numpy as np
import os
import math
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from scipy.sparse.linalg import eigs
import torch.nn as nn
import torch.nn.functional as F

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.num_nodes = xs.shape[2]
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj



class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(args, dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    total_num_nodes = None
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

        print("Shape of ", category, " input = ", data['x_' + category].shape)

        total_num_nodes = data['x_' + category].shape[2]
        data['total_num_nodes'] = total_num_nodes

    if args.predefined_S:
        count = math.ceil(total_num_nodes * (args.predefined_S_frac / 100))
        oracle_idxs = np.random.choice( np.arange(total_num_nodes), size=count, replace=False )
        data['oracle_idxs'] = oracle_idxs
        for category in ['train', 'val', 'test']:
            data['x_' + category] = data['x_' + category][:, :, oracle_idxs, :]
            data['y_' + category] = data['y_' + category][:, :, oracle_idxs, :]

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss), torch.mean(loss, dim=1)

def masked_rmse(preds, labels, null_val=np.nan):
    mse_loss, per_instance = masked_mse(preds=preds, labels=labels, null_val=null_val)
    return torch.sqrt(mse_loss), torch.sqrt(per_instance)


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss), torch.mean(loss, dim=1)


def metric(pred, real):
    mae, mae_per_instance = masked_mae(pred,real,0.0)[0].item(), masked_mae(pred,real,0.0)[1]
    rmse, rmse_per_instance = masked_rmse(pred,real,0.0)[0].item(), masked_rmse(pred,real,0.0)[1]
    return mae, rmse, mae_per_instance, rmse_per_instance



def get_node_random_idx_split(args, num_nodes, lb, ub):
    count_percent = np.random.choice( np.arange(lb, ub+1), size=1, replace=False )[0]
    count = math.ceil(num_nodes * (count_percent / 100))

    all_node_idxs = np.arange(num_nodes)
    current_node_idxs = np.random.choice(all_node_idxs, size=count, replace=False )
    # current_node_idxs = np.sort(current_node_idxs)
    return current_node_idxs


def zero_out_remaining_input(testx, idx_current_nodes, device):
    zero_val_mask = torch.ones_like(testx).bool()#.to(device)
    zero_val_mask[:, :, idx_current_nodes, :] = False
    inps = testx.masked_fill_(zero_val_mask, 0.0)
    return inps

def apply_masked_steps(tensor, indices, mask_rate=0.15, mask_type='gaussian'):
    """
    在给定的tensor的指定时间序列中掩码一定比例的时间步。

    参数:
    - tensor: 输入的tensor, 形状为 [batch_size, feat_dim, num_nodes, time_steps]
    - indices: 需要被掩码的时间序列索引列表
    - mask_rate: 需要被掩码的时间步的比例
    - mask_type: 掩码类型，可以是 'gaussian' 或 'zero'

    返回:
    - masked_tensor: 应用了掩码的tensor
    """
    _, _, num_nodes, time_steps = tensor.shape
    # 初始化掩码为全False
    mask = torch.zeros_like(tensor).bool()

    # 根据indices和mask_rate更新掩码
    for index in indices:
        if 0 <= index < num_nodes:  # 确保索引在有效范围内
            # 计算需要掩码的时间步数量
            num_masked_steps = int(time_steps * mask_rate)
            # 随机选择需要被掩码的时间步
            masked_steps = torch.randperm(time_steps)[:num_masked_steps]
            mask[:, :, index, masked_steps] = True

    if mask_type == 'gaussian':
        # 生成高斯掩码
        gaussian_mask = torch.randn_like(tensor)
        masked_tensor = tensor * (~mask) + gaussian_mask * mask
    elif mask_type == 'zero':
        # 生成零值掩码
        masked_tensor = tensor * (~mask)
    else:
        raise ValueError("Unsupported mask type. Choose 'gaussian' or 'zero'.")

    return masked_tensor

def gaussian_noise_input(testx, idx_current_nodes, device='cuda:0'):
    '''
    random mask with gaussian noise
    '''
    mean = torch.mean(testx)
    std_dev = torch.std(testx)
    # mean = 0  # 均值
    # std_dev = 1  # 标准差
    gaussian_noise = torch.normal(mean=mean, std=std_dev, size=testx.size(), device=device)
    # 将指定节点上的值设置为高斯噪声
    gaussian_noise[:, :, idx_current_nodes, :] = testx[:, :, idx_current_nodes, :]
    inps = gaussian_noise
    return inps

def get_subset_from_nodes(idx_all_nodes, idx_subset_nodes):
    """
    :param idx_all_nodes: <class 'numpy.ndarray'>
    :param idx_subset_nodes: <class 'numpy.ndarray'>
    :return idx_subset: <class 'numpy.ndarray'>
    """
    idx_mask_nodes = np.setdiff1d(idx_all_nodes, idx_subset_nodes)
    zero_val_mask = idx_all_nodes.copy()
    zero_val_mask[idx_mask_nodes] = 0.0
    subset = zero_val_mask
    return subset

def get_idx_subset_from_idx_all_nodes_v2(idx_all_nodes, lb, ub):
    """
    :param idx_all_nodes: <class 'numpy.ndarray'>
    :param mask_radio: float
    :return idx_subset: <class 'numpy.ndarray'>
    """
    idx_all_nodes = idx_all_nodes.cpu().detach().numpy()
    
    count_percent = np.random.choice( np.arange(lb, ub+1), size=1, replace=False )[0]
    length_subset = math.ceil(len(idx_all_nodes * (count_percent / 100)))

    idx_subset = np.random.choice(idx_all_nodes, size=length_subset, replace=False)
    return idx_subset

def get_idx_subset_from_idx_all_nodes(idx_all_nodes, mask_radio = 0.15):
    """
    :param idx_all_nodes: <class 'numpy.ndarray'>
    :param mask_radio: float
    :return idx_subset: <class 'numpy.ndarray'>
    """
    idx_all_nodes = idx_all_nodes.cpu().detach().numpy()
    length_subset = math.ceil(len(idx_all_nodes) * mask_radio)
    # idx_subset = idx_all_nodes[:length_subset]
    idx_subset = np.random.choice(idx_all_nodes, size=length_subset, replace=False)
    # idx_subset = np.sort(idx_subset)
    return idx_subset

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    n_iter = W.shape[0]

    # W = W.cpu().detach().numpy()
    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR', maxiter= n_iter * 100000)[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def obtain_instance_prototypes(args, x_train):
    stride = x_train.shape[0] // args.num_prots
    prototypes = []
    for i in range(args.num_prots):
        a = x_train[ i*stride : (i+1)*stride ]
        prot = a[ np.random.randint(0, a.shape[0]) ] # randint will give a single interger here
        prot = np.expand_dims(prot, axis=0)
        prototypes.append(prot)

    prototypes = np.concatenate(tuple(prototypes), axis=0)
    print("\nShape of instance prototypes = ", prototypes.shape, "\n")
    prototypes = torch.FloatTensor(prototypes).to(args.device)
    return prototypes


def obtain_relevant_data_from_prototypes(args, testx, instance_prototypes, idx_current_nodes):
    rem_idx_subset = torch.LongTensor(np.setdiff1d(np.arange(args.num_nodes), idx_current_nodes)).to(args.device)
    idx_current_nodes = torch.LongTensor(idx_current_nodes).to(args.device)

    data_idx_train = instance_prototypes[:, :, idx_current_nodes, :].unsqueeze(0).repeat(testx.shape[0], 1, 1, 1, 1)
    a = testx[:, :, idx_current_nodes, :].transpose(3, 1).unsqueeze(1).repeat(1, instance_prototypes.shape[0], 1, 1, 1)
    assert data_idx_train.shape == a.shape

    raw_diff = data_idx_train - a
    diff = torch.pow( torch.absolute( raw_diff ), args.dist_exp_value ).view(testx.shape[0], instance_prototypes.shape[0], -1)
    diff = torch.mean(diff, dim=-1)

    min_values, topk_idxs = torch.topk(diff, args.num_neighbors_borrow, dim=-1, largest=False)
    b_size = testx.shape[0]
    original_instance = testx.clone()
    testx = testx.repeat(args.num_neighbors_borrow, 1, 1, 1)
    orig_neighs = []

    for j in range(args.num_neighbors_borrow):
        nbs = instance_prototypes[topk_idxs[:, j].view(-1)].transpose(3, 1)
        orig_neighs.append( nbs )
        desired_vals = nbs[:, :, rem_idx_subset, :]
        start, end = j*b_size, (j+1)*b_size
        _local = testx[start:end]
        _local[:, :, rem_idx_subset, :] = desired_vals
        testx[start:end] = _local

    orig_neighs = torch.cat(orig_neighs, dim=0)
    return testx, min_values, orig_neighs, topk_idxs, original_instance



def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


def obtain_discrepancy_from_neighs(preds, orig_neighs_forecasts, args, idx_current_nodes):
    orig_neighs_forecasts = orig_neighs_forecasts.transpose(1, 3)
    orig_neighs_forecasts = orig_neighs_forecasts[:, 0, :, :]
    orig_neighs_forecasts = orig_neighs_forecasts[:, idx_current_nodes, :]

    orig_neighs_forecasts = torch.chunk(orig_neighs_forecasts, args.num_neighbors_borrow)
    orig_neighs_forecasts = [ a.unsqueeze(1) for a in orig_neighs_forecasts ]
    orig_neighs_forecasts = torch.cat(orig_neighs_forecasts, dim=1)

    len_tensor = torch.FloatTensor( np.arange(1, preds.shape[-1]+1) ).to(args.device).view(1, 1, 1, -1).repeat(
                              preds.shape[0], args.num_neighbors_borrow, preds.shape[2], 1) # tensor of time step indexes
    distance = torch.absolute( (preds - orig_neighs_forecasts) / len_tensor ).view(preds.shape[0], args.num_neighbors_borrow, -1)
    distance = torch.mean(distance, dim=-1)
    return distance, orig_neighs_forecasts
