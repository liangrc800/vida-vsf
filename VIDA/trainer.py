# Copyright 2022 Google LLC
# Copyright (c) 2020 Zonghan Wu

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" The code containing Trainer class and optimizer """
import torch.optim as optim
import math
from forecasters.net import *
import util
from util import *

from models.loss import SinkhornDistance

class DATrainer_v2():
    def __init__(self, args, encoder_src, encoder, decoder, forecaster, model_name, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.args = args
        self.scaler = scaler
        self.encoder = encoder
        self.decoder = decoder
        self.forecaster = forecaster
        self.encoder_src = encoder_src
        self.device = device
        self.optimizer = optim.Adam(self.forecaster.parameters(), lr=lrate, weight_decay=wdecay)
        # optimizer
        self.pre_optimizer = torch.optim.Adam(
            list(self.encoder.parameters())+list(self.decoder.parameters()),
            lr=lrate,
            weight_decay=wdecay
        )

        self.pda_optimizer = torch.optim.Adam(
            list(self.encoder.parameters())+list(self.decoder.parameters()),
            lr=lrate,
            weight_decay=wdecay
        )

        self.loss = util.masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl

        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum', device=self.device)

    def pretrain(self, args, input, real_val, idx, idx_subset):
        # Freeze the forecaster
        for k, v in self.forecaster.named_parameters():
            v.requires_grad = False
        
        self.encoder.train()
        self.decoder.train()
        self.pre_optimizer.zero_grad()

        # Encode both source features via our time-frequency feature encoder
        feat_src, out_s = self.encoder(input)
        # Decode extracted features to time series
        src_recons = self.decoder(feat_src, out_s)
        src_recons[:, :, idx_subset, :] = input[:, :, idx_subset, :]
        # src_recons = zero_out_remaining_input(src_recons, idx_subset, args.device)

        # src_recons for forecasting
        pred_result = self.forecaster(src_recons, idx=idx, args=args)
        pred_result = pred_result.transpose(1,3)
        predict_all = self.scaler.inverse_transform(pred_result)
        predict = predict_all[:, :, idx_subset, :]

        # raw data for ssl
        hidden_state = self.forecaster(input, idx=idx, args=args)
        hidden_state = hidden_state.transpose(1,3)
        hidden_state = self.scaler.inverse_transform(hidden_state)
        # hidden_state = hidden_state[:, :, idx_subset, :]

        real = torch.unsqueeze(real_val,dim=1)
        real = real[:, :, idx_subset, :]

        if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
            self.task_level +=1
        if self.cl:
            loss_spfc, _ = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
            loss_spc, _ = self.loss(predict_all[:, :, :, :self.task_level], hidden_state[:, :, :, :self.task_level], 0.0)
        else:
            loss_spfc, _ = self.loss(predict, real, 0.0)
            loss_spc, _ = self.loss(predict_all, hidden_state, 0.0)
                        
        loss = args.w_spfc * loss_spfc + args.w_spc * loss_spc

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        self.pre_optimizer.step()
        # self.lr_scheduler_pre.step()
        rmse = util.masked_rmse(predict,real,0.0)[0].item()
        rmse_spc = util.masked_rmse(predict_all,hidden_state,0.0)[0].item()
        self.iter += 1
        return loss.item(), rmse, loss_spfc.item(), loss_spc.item(), rmse_spc

    def pre_eval(self, args, input, real_val, idx, idx_subset):
        # self.model.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.forecaster.eval()

        # Encode both source features via our time-frequency feature encoder
        feat_src, out_s = self.encoder(input)
        # Decode extracted features to time series
        src_recons = self.decoder(feat_src, out_s)
        src_recons[:, :, idx_subset, :] = input[:, :, idx_subset, :]
        # src_recons = zero_out_remaining_input(src_recons, idx_subset, args.device)

        # src_recons for forecasting
        pred_result = self.forecaster(src_recons, idx=idx, args=args)
        pred_result = pred_result.transpose(1,3)
        predict_all = self.scaler.inverse_transform(pred_result)
        predict = predict_all[:, :, idx_subset, :]

        # raw data for ssl
        hidden_state = self.forecaster(input, idx=idx, args=args)
        hidden_state = hidden_state.transpose(1,3)
        hidden_state = self.scaler.inverse_transform(hidden_state)
        # hidden_state = hidden_state[:, :, idx_subset, :]

        real = torch.unsqueeze(real_val,dim=1)
        real = real[:, :, idx_subset, :]

        loss_spfc, _ = self.loss(predict, real, 0.0)
        loss_spc, _ = self.loss(predict_all, hidden_state, 0.0)

        loss = args.w_spfc * loss_spfc + args.w_spc * loss_spc

        rmse = util.masked_rmse(predict, real, 0.0)[0].item()
        rmse_spc = util.masked_rmse(predict_all, hidden_state, 0.0)[0].item()
        return loss.item(), rmse, loss_spfc.item(), loss_spc.item(), rmse_spc

    def alignment(self, args, input, real_val, idx, idx_subset):
        # Freeze the forecaster
        for k, v in self.forecaster.named_parameters():
            v.requires_grad = False
        for k, v in self.encoder_src.named_parameters():
            v.requires_grad = False
        
        # self.model.train()
        self.encoder.train()
        self.decoder.train()
        self.pda_optimizer.zero_grad()

        src_x = input.clone()
        trg_x = zero_out_remaining_input(input.clone(), idx_subset, args.device)

        # Encode both source and target features via our time-frequency feature encoder
        feat_src, _, = self.encoder_src(src_x)
        feat_trg, out_t = self.encoder(trg_x)

        # Compute alignment loss
        dr, _, _ = self.sink(feat_src, feat_trg)
        loss_align = dr

        # Decode extracted features to time series
        trg_recons = self.decoder(feat_trg, out_t)
        trg_recons[:, :, idx_subset, :] = trg_x[:, :, idx_subset, :]
        # trg_recons = zero_out_remaining_input(trg_recons, idx_subset, args.device)

        output = self.forecaster(trg_recons, idx=idx, args=args)
        output = output.transpose(1, 3)
        predict = self.scaler.inverse_transform(output)
        predict = predict[:, :, idx_subset, :]

        real = torch.unsqueeze(real_val,dim=1)
        real = real[:, :, idx_subset, :]

        if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
            self.task_level +=1
        if self.cl:
            loss_fafc, _ = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss_fafc, _ = self.loss(predict, real, 0.0)
        
        loss = args.w_fafc * loss_fafc + args.w_align * loss_align

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        self.pda_optimizer.step()
        # self.lr_scheduler.step()
        # mae = util.masked_mae(predict,real,0.0)[0].item()
        rmse = util.masked_rmse(predict,real,0.0)[0].item()
        self.iter += 1
        return loss.item(), rmse, loss_fafc.item(), loss_align.item()

    def align_eval(self, args, input, real_val, idx, idx_subset):
        # self.model.eval()
        self.encoder.eval()
        self.encoder_src.eval()
        self.decoder.eval()
        self.forecaster.eval()

        src_x = input.clone()
        trg_x = zero_out_remaining_input(input.clone(), idx_subset, args.device)

        # Encode both source and target features via our time-frequency feature encoder
        feat_src, _, = self.encoder_src(src_x)
        feat_trg, out_t = self.encoder(trg_x)

        # Compute alignment loss
        dr, _, _ = self.sink(feat_src, feat_trg)
        loss_align = dr

        # Decode extracted features to time series
        trg_recons = self.decoder(feat_trg, out_t)
        # trg_recons = zero_out_remaining_input(trg_recons, idx_subset, args.device)
        trg_recons[:, :, idx_subset, :] = trg_x[:, :, idx_subset, :]
        
        output = self.forecaster(trg_recons, idx=idx, args=args)
        output = output.transpose(1,3)
        predict = self.scaler.inverse_transform(output)
        predict = predict[:, :, idx_subset, :]

        real = torch.unsqueeze(real_val,dim=1)
        real = real[:, :, idx_subset, :]

        loss_fafc = self.loss(predict, real, 0.0)[0].item()

        loss = args.w_fafc * loss_fafc + args.w_align * loss_align.item()

        rmse = util.masked_rmse(predict, real, 0.0)[0].item()
        return loss, rmse, loss_fafc

    def train_forecaster(self, args, input, real_val, idx=None):
        self.forecaster.train()
        self.optimizer.zero_grad()
        output = self.forecaster(input, idx=idx, args=args)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
            self.task_level +=1
        if self.cl:
            loss, _ = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss, _ = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.forecaster.parameters(), self.clip)
        self.optimizer.step()
        # mae = util.masked_mae(predict,real,0.0)[0].item()
        rmse = util.masked_rmse(predict,real,0.0)[0].item()
        self.iter += 1
        return loss.item(), rmse

    def eval_forecaster(self, args, input, real_val, idx=None):
        self.forecaster.eval()
        # output = self.model(input, args=args)
        output = self.forecaster(input, idx=idx, args=args)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        rmse = util.masked_rmse(predict,real,0.0)[0].item()
        return loss[0].item(), rmse

class Trainer():
    def __init__(self, args, model, model_name, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.args = args
        self.scaler = scaler
        self.model = model
        self.model_name = model_name
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl

    def train(self, args, input, real_val, epoch_num, batches_per_epoch, current_epoch_batch_num, idx=None):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(input, idx=idx, args=args)

        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
            self.task_level +=1
        if self.cl:
            loss, _ = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss, _ = self.loss(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        # mae = util.masked_mae(predict,real,0.0)[0].item()
        rmse = util.masked_rmse(predict,real,0.0)[0].item()
        self.iter += 1
        return loss.item(), rmse

    def eval(self, args, input, real_val):
        self.model.eval()
        output = self.model(input, args=args)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)

        rmse = util.masked_rmse(predict,real,0.0)[0].item()
        return loss[0].item(), rmse



class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        # for param in self.params:
        #     grad_norm += math.pow(param.grad.data.norm(), 2)
        #
        # grad_norm = math.sqrt(grad_norm)
        # if grad_norm > 0:
        #     shrinkage = self.max_grad_norm / grad_norm
        # else:
        #     shrinkage = 1.
        #
        # for param in self.params:
        #     if shrinkage < 1:
        #         param.grad.data.mul_(shrinkage)
        self.optimizer.step()
        return  grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()
