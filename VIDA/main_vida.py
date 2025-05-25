# Copyright 2022 Google LLC
# Copyright (c) 2020 Zonghan Wu

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" The primary training script with our wrapper technique """
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from util import *
from trainer import DATrainer_v2
from forecasters.net import gtnet
from forecasters.ASTGCN import make_ASTGCN
from forecasters.MSTGCN import make_MSTGCN
from forecasters.TGCN import TGCN

import ast
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt

from models.vida import tf_encoder, tf_decoder


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')

parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')


parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')

parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=float, default=5.0, help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')

parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=3407,help='random seed')
parser.add_argument('--path_model_save', type=str, default=None)
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--forecaster_expid',type=int,default=1,help='forecasters experiment id')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

parser.add_argument('--runs',type=int,default=10, help='number of runs')

parser.add_argument('--random_node_idx_split_runs', type=int, default=100, help='number of random node/variable split runs')
parser.add_argument('--lower_limit_random_node_selections', type=int, default=15, help='lower limit percent value for number of nodes in any given split')
parser.add_argument('--upper_limit_random_node_selections', type=int, default=15, help='upper limit percent value for number of nodes in any given split')

parser.add_argument('--model_name', type=str, default='mtgnn')

parser.add_argument('--mask_remaining', type=str_to_bool, default=False, help='the partial setting, subset S')

parser.add_argument('--predefined_S', type=str_to_bool, default=False, help='whether to use subset S selected apriori')
parser.add_argument('--predefined_S_frac', type=int, default=15, help='percent of nodes in subset S selected apriori setting')
parser.add_argument('--adj_identity_train_test', type=str_to_bool, default=False, help='whether to use identity matrix as adjacency during training and testing')

parser.add_argument('--do_full_set_oracle', type=str_to_bool, default=False, help='the oracle setting, where we have entire data for training and \
                            testing, but while computing the error metrics, we do on the subset S')
parser.add_argument('--full_set_oracle_lower_limit', type=int, default=15, help='percent of nodes in this setting')
parser.add_argument('--full_set_oracle_upper_limit', type=int, default=15, help='percent of nodes in this setting')

parser.add_argument('--borrow_from_train_data', type=str_to_bool, default=False, help="the Retrieval solution")
parser.add_argument('--num_neighbors_borrow', type=int, default=5, help="number of neighbors to borrow from, during aggregation")
parser.add_argument('--dist_exp_value', type=float, default=0.5, help="the exponent value")
parser.add_argument('--neighbor_temp', type=float, default=0.1, help="the temperature paramter")
parser.add_argument('--use_ewp', type=str_to_bool, default=False, help="whether to use ensemble weight predictor, ie, FDW")

parser.add_argument('--fraction_prots', type=float, default=1.0, help="fraction of the training data to be used as the Retrieval Set")

# ASTGCN
parser.add_argument('--nb_block', type=int, default=2)
parser.add_argument('--K_A', type=int, default=3)
parser.add_argument('--in_channels_A', type=int, default=2)
parser.add_argument('--nb_chev_filter_A', type=int, default=64)
parser.add_argument('--nb_time_filter_A', type=int, default=64)
parser.add_argument('--time_strides_A', type=int, default=1)

# ========  PDA Experiments ================
parser.add_argument('--input_channels', type=int, default=207, help="input_channels = int_dim * num_nodes")
parser.add_argument('--fourier_modes', type=int, default=6, help="Number of low-frequency modes $\varsigma$: fourier_modes = seq_len // 2")
parser.add_argument('--sequence_len', type=int, default=12, help="sequence_len")
parser.add_argument('--final_out_channels', type=int, default=12, help="sequence_len")
parser.add_argument('--mid_channels', type=int, default=1024, help="TCN temporal embedding size M")
parser.add_argument('--kernel_size', type=int, default=5, help="kernel_size")
parser.add_argument('--features_len', type=int, default=1, help="features_len")
parser.add_argument('--runid', type=int, default=0, help="run id")
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--pre_epochs',type=int,default=70,help='')
parser.add_argument('--align_epochs',type=int,default=30,help='')

parser.add_argument('--w_spfc', type=float, default=0.9, help="weight of source pretraining forecast loss")
parser.add_argument('--w_fafc', type=float, default=0.9, help="weight of feature alignment forecast loss")
parser.add_argument('--w_spc', type=float, default=0.1, help="weight of self-supervised prediction consistency")
parser.add_argument('--w_align', type=float, default=0.1, help="weight of domain alignment loss")

parser.add_argument('--patience',type=int, default=20, help='for early stopping')

parser.add_argument('--useCNN',     type=str_to_bool, default=False,    help="Use CNN for data generation.")
parser.add_argument('--useResNet',  type=str_to_bool, default=False,    help="Use ResNet for data generation.")
parser.add_argument('--useTCN',     type=str_to_bool, default=True,     help="Use TCN for data generation.")

parser.add_argument('--use_lr_scheduler',     type=str_to_bool, default=True,     help="Use lr_scheduler to adjust learning_rate.")
parser.add_argument('--update_nums', type=int, default=30, help='for lr_scheduler')

args = parser.parse_args()
torch.set_num_threads(3)

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(args.seed)

def main(runid):
    if args.predefined_S:
        assert args.random_node_idx_split_runs == 1, "no need for multiple random runs in oracle setting"
        assert args.lower_limit_random_node_selections == args.upper_limit_random_node_selections == 100, "upper and lower limit should be same and equal to 100 percent"

    device = torch.device(args.device)
    dataloader = load_dataset(args, args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    args.num_nodes = dataloader['train_loader'].num_nodes
    print("Number of variables/nodes = ", args.num_nodes)
    args.enc_in = args.num_nodes
    args.dec_in = args.num_nodes
    args.c_out  = args.num_nodes

    if args.node_dim >= args.num_nodes:
        args.node_dim = args.num_nodes
        args.subgraph_size = args.num_nodes

    dataset_name = args.data.strip().split('/')[-1].strip()

    if dataset_name == "METR-LA":
        args.in_dim = 1
    
    args.input_channels = args.num_nodes * args.in_dim
    args.runid = runid

    if args.use_lr_scheduler:
        args.update_size = math.floor(dataloader["x_train"].shape[0] / args.batch_size) * args.update_nums

    if dataset_name == "METR-LA":
        args.adj_data = "../data/sensor_graph/adj_mx.pkl"
        predefined_A = load_adj(args.adj_data)
        predefined_A = torch.tensor(predefined_A)-torch.eye(dataloader['total_num_nodes'])
        predefined_A = predefined_A.to(device)
    elif dataset_name.lower() == "pems-bay":
        args.adj_data = "../data/sensor_graph/adj_mx_bay.pkl"
        predefined_A = load_adj(args.adj_data)
        predefined_A = torch.tensor(predefined_A)-torch.eye(dataloader['total_num_nodes'])
        predefined_A = predefined_A.to(device)
    else:
        predefined_A = None


    if args.adj_identity_train_test:
        if predefined_A is not None:
            print("\nUsing identity matrix during training as well as testing\n")
            predefined_A = torch.eye(predefined_A.shape[0]).to(args.device)

    if args.predefined_S and predefined_A is not None:
        oracle_idxs = dataloader['oracle_idxs']
        oracle_idxs = torch.tensor(oracle_idxs).to(args.device)
        predefined_A = predefined_A[oracle_idxs, :]
        predefined_A = predefined_A[:, oracle_idxs]
        assert predefined_A.shape[0] == predefined_A.shape[1] == oracle_idxs.shape[0]
        print("\nAdjacency matrix corresponding to oracle idxs obtained\n")

    args.path_model_save = "./saved_models/" + args.model_name + "/" + dataset_name + "/" + "seed" + str(args.seed) + "/"

    import os
    if not os.path.exists(args.path_model_save):
        os.makedirs(args.path_model_save)
    
    print(args.model_name)
    if args.model_name.lower() == 'mtgnn':
        forecaster = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                        device, predefined_A=predefined_A,
                        dropout=args.dropout, subgraph_size=args.subgraph_size,
                        node_dim=args.node_dim,
                        dilation_exponential=args.dilation_exponential,
                        conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                        skip_channels=args.skip_channels, end_channels= args.end_channels,
                        seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                        layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True).to(device)
        
        # print_model_size(forecaster, device)
        
        print('The recpetive field size is', forecaster.receptive_field)
        print(args)
        forecaster_params = sum([p.nelement() for p in forecaster.parameters()])
        print('Number of forecaster parameters is', forecaster_params)

    elif args.model_name.lower() in {'astgcn', 'mstgcn', 'tgcn'}:
        print(args)
        if dataset_name.lower() in {"metr-la", "pems-bay"}:
            adj = predefined_A.cpu().detach().numpy()
        else:
            mtgnn = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                    device, predefined_A=predefined_A,
                    dropout=args.dropout, subgraph_size=args.subgraph_size,
                    node_dim=args.node_dim,
                    dilation_exponential=args.dilation_exponential,
                    conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                    skip_channels=args.skip_channels, end_channels= args.end_channels,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True).to(device)
            adj = mtgnn.obtain_adj_matrix(args).cpu().detach().numpy()

        if args.model_name.lower() == 'astgcn':
            forecaster = make_ASTGCN(device, args.nb_block, args.in_dim, args.K_A, args.nb_chev_filter_A, args.nb_time_filter_A, args.time_strides_A, adj,
                                     args.seq_out_len, args.seq_in_len, args.num_nodes).to(device)
        elif args.model_name.lower() == 'mstgcn':
            forecaster = make_MSTGCN(device, args.nb_block, args.in_dim, args.K_A, args.nb_chev_filter_A, args.nb_time_filter_A, args.time_strides_A, adj,
                                     args.seq_out_len, args.seq_in_len).to(device)
        elif args.model_name.lower() == 'tgcn':
            forecaster = TGCN(args.seq_out_len, adj, 128).to(device)
    
    encoder = tf_encoder(args).to(device)
    decoder = tf_decoder(args).to(device)
    # backbone = nn.Sequential(encoder, decoder)

    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    # total_params = encoder_params+decoder_params+forecaster_params
    print('Number of encoder parameters is', encoder_params)
    print('Number of decoder parameters is', decoder_params)
    # print('Number of total parameters is', total_params)

    engine = DATrainer_v2(args, None, encoder, decoder, forecaster, args.model_name, args.learning_rate, args.weight_decay, \
                        args.clip, args.step_size1, args.seq_out_len, scaler, device, args.cl)
    print("=" * 45)
    print("\n Forecaster Training. \n",flush=True)
    print("=" * 45)
    his_loss =[]
    his_rmse =[]
    val_time = []
    train_time = []
    minl = 1e5

    for i in range(1, args.epochs+1):
        train_loss = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            if iter%args.step_size2==0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes/args.num_split)
            for j in range(args.num_split):
                if j != args.num_split-1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                metrics = engine.train_forecaster(args, tx, ty[:,0,:,:], id)
                train_loss.append(metrics[0])
                train_rmse.append(metrics[1])
            if iter % args.print_every == 0 :
                log = 'Forecaster: Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_rmse = []

        s1 = time.time()

        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval_forecaster(args, testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_rmse.append(metrics[1])
        s2 = time.time()
        log = 'Forecaster: Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        his_rmse.append(mvalid_rmse)

        log = 'Forecaster: Runid: {:d} , Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(runid, i, mtrain_loss, mtrain_rmse, mvalid_loss, mvalid_rmse, (t2 - t1)),flush=True)

        if mvalid_loss<minl:
            torch.save(engine.forecaster.state_dict(), args.path_model_save + "forecaster" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) + ".pth")
            minl = mvalid_loss

    if args.epochs > 0:
        print("Forecaster Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Forecaster Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        bestid = np.argmin(his_loss)
        print("Forecaster Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid],4)))
        print("The valid RMSE on best model is", str(round(his_rmse[bestid],4)))
    engine.forecaster.load_state_dict(torch.load(args.path_model_save + "forecaster" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) + ".pth"))
    print("\n Forecaster loaded. \n")

    print("=" * 45)
    print("Pre training stage", flush=True)
    print("=" * 45)
    pre_his_loss =[]
    pre_his_rmse =[]
    pre_val_time = []
    pre_train_time = []
    minl = 1e5
    # early stopping
    early_stop_counter = 0
    for i in range(1, args.pre_epochs+1):
        pre_train_loss = []
        pre_train_rmse = []
        pre_train_fc_loss = []
        pre_train_ssl_loss = []
        pre_train_ssl_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            if iter%args.step_size2==0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes/args.num_split)
            for j in range(args.num_split):
                if j != args.num_split-1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                id_subset = get_idx_subset_from_idx_all_nodes(idx_all_nodes=id)
                id_subset = torch.tensor(id_subset).to(device)
                metrics = engine.pretrain(args, tx, ty[:,0,:,:], id, id_subset)
                pre_train_loss.append(metrics[0])
                pre_train_rmse.append(metrics[1])
                pre_train_fc_loss.append(metrics[2])
                pre_train_ssl_loss.append(metrics[3])
                pre_train_ssl_rmse.append(metrics[4])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, pre train loss: {:.4f}, pre train RMSE: {:.4f}, pre forecast loss: {:.4f}, pre ssl loss: {:.4f}, pre ssl RMSE: {:.4f}'
                print(log.format(iter, pre_train_loss[-1], pre_train_rmse[-1], pre_train_fc_loss[-1], pre_train_ssl_loss[-1], pre_train_ssl_rmse[-1]), flush=True)
                if pre_train_fc_loss[-1] <= 0:
                    break
        t2 = time.time()
        pre_train_time.append(t2-t1)
        # validation
        pre_valid_loss = []
        pre_valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            id = torch.arange(args.num_nodes).to(device)
            id_subset = get_idx_subset_from_idx_all_nodes(idx_all_nodes=id)
            id_subset = torch.tensor(id_subset).to(device)
            metrics = engine.pre_eval(args, testx, testy[:,0,:,:], id, id_subset)
            pre_valid_loss.append(metrics[2])
            pre_valid_rmse.append(metrics[1])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        pre_val_time.append(s2-s1)
        pre_mtrain_loss = np.mean(pre_train_loss)
        pre_mtrain_rmse = np.mean(pre_train_rmse)
        pre_mtrain_fc_loss = np.mean(pre_train_fc_loss)

        pre_mvalid_loss = np.mean(pre_valid_loss)
        pre_mvalid_rmse = np.mean(pre_valid_rmse)
        if pre_mvalid_loss <= 0 or pre_mvalid_rmse <= 0:
            break
        pre_his_loss.append(pre_mvalid_loss)
        pre_his_rmse.append(pre_mvalid_rmse)

        log = 'Runid: {:d} , Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, fc train loss: {:.4f}, fc Valid Loss: {:.4f}, fc Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(runid, i, pre_mtrain_loss, pre_mtrain_rmse, pre_mtrain_fc_loss, pre_mvalid_loss, pre_mvalid_rmse, (t2 - t1)),flush=True)

        if pre_mvalid_loss > 0 and pre_mvalid_loss < minl:
            # torch.save(engine.model.state_dict(), args.path_model_save + "pre_exp" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) + ".pth")
            torch.save(engine.encoder.state_dict(), args.path_model_save + "pre_encoder" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) + ".pth")
            torch.save(engine.decoder.state_dict(), args.path_model_save + "pre_decoder" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) + ".pth")
            minl = pre_mvalid_loss
            early_stop_counter = 0
            print(f"early_stop_counter: {early_stop_counter} / {args.patience}")
        elif pre_mvalid_loss > 0 and pre_mvalid_loss > minl:
            early_stop_counter += 1
            print(f"early_stop_counter: {early_stop_counter} / {args.patience}")
            # if early_stop_counter >= args.patience and i > 99:
            if early_stop_counter >= args.patience:
                print("Early stopping...")
                break

    if args.pre_epochs > 0:
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(pre_train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(pre_val_time)))

        bestid = np.argmin(pre_his_loss)
        print("Pre training finished")
        print("The valid loss on best model is", str(round(pre_his_loss[bestid],4)))
        print("The valid RMSE on best model is", str(round(pre_his_rmse[bestid],4)))

    engine.encoder.load_state_dict(torch.load(args.path_model_save + "pre_encoder" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) +".pth"))
    engine.decoder.load_state_dict(torch.load(args.path_model_save + "pre_decoder" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) +".pth"))
    print("\n Pre Model loaded \n")

    # encoder_src = engine.encoder.load_state_dict(torch.load(args.path_model_save + "pre_encoder" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) +".pth"))
    encoder_src = deepcopy(encoder)
    engine = DATrainer_v2(args, encoder_src, encoder, decoder, forecaster, args.model_name, args.learning_rate, args.weight_decay, \
                        args.clip, args.step_size1, args.seq_out_len, scaler, device, args.cl)
    engine.encoder_src.load_state_dict(torch.load(args.path_model_save + "pre_encoder" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) +".pth"))

    print("=" * 45)
    print("Domain alignment stage",flush=True)
    print("=" * 45)
    align_his_loss =[]
    align_his_rmse =[]
    align_val_time = []
    align_train_time = []
    minl = 1e5
    # early stopping
    early_stop_counter = 0

    for i in range(1, args.align_epochs+1):
        align_train_loss = []
        align_train_rmse = []
        align_train_fc_loss = []
        align_train_sink_loss = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            id = torch.arange(args.num_nodes).to(device)
            tx = trainx[:, :, id, :]
            ty = trainy[:, :, id, :]
            if iter%args.step_size2==0:
                perm = np.random.permutation(range(args.num_nodes))
            num_sub = int(args.num_nodes/args.num_split)
            for j in range(args.num_split):
                if j != args.num_split-1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
            id_subset = get_idx_subset_from_idx_all_nodes(idx_all_nodes=id)
            id_subset = torch.tensor(id_subset).to(device)
            metrics = engine.alignment(args, tx, ty[:,0,:,:], id, id_subset)
            align_train_loss.append(metrics[0])
            align_train_rmse.append(metrics[1])
            align_train_fc_loss.append(metrics[2])
            align_train_sink_loss.append(metrics[3])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, align train loss: {:.4f}, align train RMSE: {:.4f}, align fc loss: {:.4f}, align sink loss: {:.4f}'
                print(log.format(iter, align_train_loss[-1], align_train_rmse[-1], align_train_fc_loss[-1], align_train_sink_loss[-1]),flush=True)
        t2 = time.time()
        align_train_time.append(t2-t1)
        # validation
        align_valid_loss = []
        align_valid_rmse = []

        s1 = time.time()

        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            id = torch.arange(args.num_nodes).to(device)
            id_subset = get_idx_subset_from_idx_all_nodes(idx_all_nodes=id)
            id_subset = torch.tensor(id_subset).to(device)
            metrics = engine.align_eval(args, testx, testy[:,0,:,:], id, id_subset)
            align_valid_loss.append(metrics[2])
            align_valid_rmse.append(metrics[1])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        align_val_time.append(s2-s1)
        align_mtrain_loss = np.mean(align_train_loss)
        align_mtrain_rmse = np.mean(align_train_rmse)
        align_mtrain_fc_loss = np.mean(align_train_fc_loss)

        align_mvalid_loss = np.mean(align_valid_loss)
        align_mvalid_rmse = np.mean(align_valid_rmse)
        align_his_loss.append(align_mvalid_loss)
        align_his_rmse.append(align_mvalid_rmse)

        log = 'Runid: {:d} , Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, align fc train Loss: {:.4f}, align fc valid loss: {:.4f}, align fc valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(runid, i, align_mtrain_loss, align_mtrain_rmse, align_mtrain_fc_loss, align_mvalid_loss, align_mvalid_rmse, (t2 - t1)),flush=True)

        if align_mvalid_loss < minl:
            # torch.save(engine.encoder.state_dict(), args.path_model_save + "align_exp" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) + ".pth")
            torch.save(engine.encoder.state_dict(), args.path_model_save + "align_encoder" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) + ".pth")
            torch.save(engine.decoder.state_dict(), args.path_model_save + "align_decoder" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) + ".pth")
            minl = align_mvalid_loss
            early_stop_counter = 0
            print(f"early_stop_counter: {early_stop_counter} / {args.patience}")
        else:
            early_stop_counter += 1
            print(f"early_stop_counter: {early_stop_counter} / {args.patience}")
            if early_stop_counter >= args.patience:
                print("Early stopping...")
                break

    if args.align_epochs > 0:
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(align_train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(align_val_time)))

        bestid = np.argmin(align_his_loss)
        print("Domain aligment finished")
        print("The valid loss on best model is", str(round(align_his_loss[bestid],4)))
        print("The valid RMSE on best model is", str(round(align_his_rmse[bestid],4)))

    print("\n Alignment Model loaded\n")

    print("=" * 45)
    print("Inference stage",flush=True)
    print("=" * 45)
    engine.encoder.load_state_dict(torch.load(args.path_model_save + "align_encoder" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) +".pth"))
    engine.decoder.load_state_dict(torch.load(args.path_model_save + "align_decoder" + str(args.expid) + "_" + str(runid) + "_" + "seed" + str(args.seed) +".pth"))
    engine.encoder.eval()
    engine.decoder.eval()
    engine.forecaster.eval()
    print("\n Performing test set run. To perform the following inference on validation data, simply adjust 'y_test' to 'y_val' and 'test_loader' to 'val_loader', which\
            has been commented out for faster execution \n")

    random_node_split_avg_mae = []
    random_node_split_avg_rmse = []

    split_run_time=[]
    for split_run in range(args.random_node_idx_split_runs):
        s_run_1 = time.time()
        if args.predefined_S:
            pass
        else:
            print("running on random node idx split ", split_run)

            if args.do_full_set_oracle:
                idx_current_nodes = np.arange( args.num_nodes, dtype=int ).reshape(-1)
                assert idx_current_nodes.shape[0] == args.num_nodes

            else:
                idx_current_nodes = get_node_random_idx_split(args, args.num_nodes, args.lower_limit_random_node_selections, args.upper_limit_random_node_selections)

            print("Number of nodes in current random split run = ", idx_current_nodes.shape)

        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]
        if not args.predefined_S:
            realy = realy[:, idx_current_nodes, :]

        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)

            if not args.predefined_S:
                testx = zero_out_remaining_input(testx, idx_current_nodes, args.device) # Remove the data corresponding to the variables that are not a part of subset "S"

            with torch.no_grad():
                if args.predefined_S:
                    idx_current_nodes = None
                
                # Encode target features via our time-frequency feature encoder
                feat_trg, out_t = engine.encoder(testx)
                # Decode extracted features to time series
                trg_recons = engine.decoder(feat_trg, out_t)
                # trg_recons = zero_out_remaining_input(trg_recons, idx_current_nodes, args.device)
                trg_recons[:, :, idx_current_nodes, :] = testx[:, :, idx_current_nodes, :]
                if args.model_name.lower() == 'mtgnn':
                    preds = engine.forecaster(trg_recons, args=args, mask_remaining=args.mask_remaining, test_idx_subset=idx_current_nodes)
                else: # elif args.model_name.lower() in {'astgcn', 'mstgcn', 'tgcn'}:
                    idx = torch.arange(args.num_nodes).to(device)
                    preds = engine.forecaster(trg_recons, idx=idx, args=args)   # (batch_size, seq_len, num_nodes, input_dim)  
                preds = preds.transpose(1, 3)
                preds = preds[:, 0, :, :]
                if not args.predefined_S:
                    preds = preds[:, idx_current_nodes, :]

            outputs.append(preds)

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        mae = []
        rmse = []

        if args.do_full_set_oracle:
            full_set_oracle_idx = get_node_random_idx_split(args, args.num_nodes, args.full_set_oracle_lower_limit, args.full_set_oracle_upper_limit)

            print("Number of nodes in current oracle random split = ", full_set_oracle_idx.shape)

        for i in range(args.seq_out_len):   # this computes the metrics for multiple horizons lengths, individually, starting from 0 to args.seq_out_len
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]

            if args.do_full_set_oracle:
                pred = pred[:, full_set_oracle_idx]
                real = real[:, full_set_oracle_idx]
                assert pred.shape[1] == real.shape[1] == full_set_oracle_idx.shape[0]

            metrics = metric(pred, real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, metrics[0], metrics[1]))
            mae.append(metrics[0])
            rmse.append(metrics[1])

        random_node_split_avg_mae.append(mae)
        random_node_split_avg_rmse.append(rmse)

        s_run_2 = time.time()
        log = 'Split_Run: {:03d}, Split Run Time: {:.4f} secs'
        print(log.format(split_run, (s_run_2 - s_run_1)))
        split_run_time.append(s_run_2 - s_run_1)
    if args.random_node_idx_split_runs > 0:
        print("\n\nAverage Split Run Time: {:.4f} secs/epoch".format(np.mean(split_run_time)))

    return random_node_split_avg_mae, random_node_split_avg_rmse


if __name__ == "__main__":

    starttime = datetime.datetime.now()
    
    mae = []
    rmse = []

    for i in range(args.runs):
        m1, m2 = main(i)
        mae.extend(m1)
        rmse.extend(m2)

    mae = np.array(mae)
    rmse = np.array(rmse)

    amae = np.mean(mae,0)
    armse = np.mean(rmse,0)
    
    all_runs_avermae = np.mean(amae)
    all_runs_avermse = np.mean(armse)

    smae = np.std(mae,0)
    srmse = np.std(rmse,0)

    all_runs_aver_stdmae = np.mean(smae)
    all_runs_aver_stdrmse = np.mean(srmse)


    print('\n\nResults for multiple runs\n\n')
    for i in range(args.seq_out_len):
        print("runs {:d} ; MAE = {:.4f} +- {:.4f} ; RMSE = {:.4f} +- {:.4f}".format(
              i+1, amae[i], smae[i], armse[i], srmse[i]))
              
    print("\n Final: MAE = {:.4f} +- {:.4f} ; RMSE = {:.4f} +- {:.4f}".format( all_runs_avermae, all_runs_aver_stdmae, all_runs_avermse, all_runs_aver_stdrmse ))
    
    print("\n Final: MAE = {:.2f}({:.2f}) ; RMSE = {:.2f}({:.2f})".format( all_runs_avermae, all_runs_aver_stdmae, all_runs_avermse, all_runs_aver_stdrmse ))
    
    #long running
    endtime = datetime.datetime.now()
    print("\n")
    print((endtime - starttime).seconds)
    print("\n")
