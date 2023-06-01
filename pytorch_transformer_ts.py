# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class TransformerModel(Model):
    def __init__(
        self,
        d_feat: int = 3,
        d_model: int = 4,
        batch_size: int = 256,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=3,
        lr=0.0001,
        metric="",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=3,
        seed=999,
        **kwargs
    ):

        # set hyper-parameters.
        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        self.logger.info("Naive Transformer:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = Transformer(d_feat,d_model, nhead, num_layers, dropout, self.device)
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2 
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):

        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):

        # model.train()的作用是启用 Batch Normalization 和 Dropout。
        #在训练过程中，我们通常会对数据进行随机扰动（例如随机采样、随机旋转、随机裁剪等），以增加训练集的多样性和泛化能力。
        # 因此，我们在每次训练迭代之前都需要调用 self.model.train() 方法来确保模型处于训练模式，以便进行数据扰动和参数更新。
    
        #  模型可以更新参数，是训练模式。self.model.eval()这个是不更新，就确定了参数的值，不会更新参数，就切换到测试模式了。
        self.model.train() 

        for data in data_loader: # 这里会增加一个batch_size的一个维度。 这里data_loader是个（1601，21）的数据，data的数据变了，是怎么样变的？？
            feature = data[:, :, 0:-1].to(self.device) # 三维数组除了最后一列都选。
            label = data[:, -1, -1].to(self.device) ## 选最后一列最后一个数。

            pred = self.model(feature.float())  # .float()
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward() #根据模型当前的参数和输入数据，计算模型的损失函数对模型参数的导数（梯度），然后将梯度存储在模型的参数中。
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0) #对模型的参数梯度进行截断，防止梯度爆炸问题，这里的 3.0 是截断的阈值。
            self.train_optimizer.step() #根据计算得到的参数梯度，更新模型的参数，使得模型的损失函数尽可能的减小。

    def test_epoch(self, data_loader):

        self.model.eval()

        scores = []
        losses = []

        for data in data_loader:

            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():  # 指定这一段代码的张量不需要梯度，不需要被计算图中，可以提高执行效率，GPU内存消耗减少。
                pred = self.model(feature.float())  # .float(),这里跳进去的函数是”mse“那段/
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)
    
    # def wgn(self,sequence, snr):
    #     sequence = sequence * 1.0e2 
    #     Ps = np.sum(abs(sequence)**2)/len(sequence)
    #     Pn = Ps/(10**((snr/10)))
    #     noise = np.random.randn(len(sequence)) * np.sqrt(Pn)
    #     signal_with_noise = sequence + noise
    #     signal_with_noise = signal_with_noise / 1.0e2  # 将添加了噪声的sequence恢复到原来的数值范围
    #     return signal_with_noise

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        # dl_train中  data_arr.shape 为（1601，21）其中1601最后一个索引是nan,data_index,就是设置的日期。（2020.01.02-2020.01.23）
        # 但是显示的是end =(2020-01-31) 有可能01.23--01.31是没有值的。
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        
        # for i in range(dl_train.data_arr.shape[0]):
        #    dl_train.data_arr[i,:] = self.wgn(dl_train.data_arr[i,:],snr=10)
        
        
        # dl_vaild 里面dl_vaild.shape(3001,21)  同理，3001的索引是nan, data_index的日期是（2020.01.02--2020.02-20）把train里面的数值也加进去了？
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader


        # Dataloader是pytorch的数据处理，并用batch_sampler将指定batch分配给指定worker，worker将它负责的batch加载进RAM。
        # num_worker设置得大，好处是寻batch速度快，因为下一轮迭代的batch很可能在上一轮/上上一轮...迭代时已经加载好了。坏处是内存开销大，也加重了CPU负担

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()

            preds.append(pred)
        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, d_feat, d_model=4, nhead=2, num_layers=2, dropout=0,device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model,dtype=torch.float32).to(device="cuda:3")
        self.pos_encoder = PositionalEncoding(d_model).to(device="cuda:3")
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout).to(device="cuda:3")
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(device="cuda:3")
        self.decoder_layer = nn.Linear(d_model, 1,).to(device="cuda:3")


    def forward(self, src):
        # src [N, T, F], [512, 60, 6]
        src = src.to(dtype=torch.float32,device="cuda:3")
        src = self.feature_layer(src).to(dtype=torch.float32,device="cuda:3")  # [512, 60, 8]
        # src  [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]  (output.transpose(1, 0)[:, -1, :]) 这个变换之后是[512,8] ,在经历decoder_layer变成[512,1]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()
