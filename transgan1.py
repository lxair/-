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
from tqdm import tqdm
import matplotlib.dates as mdates
import torch.nn.functional as F
from torch import Tensor
from absl import app, flags
from easydict import EasyDict
import torchvision

# from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhansmaster.cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhansmaster.cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from torch.utils.tensorboard import SummaryWriter
import logging
import operator
import os
from copy import deepcopy

from imageio import imsave


from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from qlib.contrib.model.pytorch_transformer_ts import Transformer
from qlib.contrib.model.pytorch_transformer_ts import TransformerModel
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class TransGANModel(Model):
    def __init__(
        self,
        d_feat: int = 4,
        d_model: int = 4,
        nhead: int = 2,
        rank: int = -1,
        n_critic : int = 1,
        num_layers: int = 2,
        dropout: float = 0,
        activation: str = "gelu",
        batch_size: int = 256,
        early_stop=10,
        g_accumulated_times = 1,
        iter_idx = 0,
        world_size = 1,
        ema = 0.995,
        n_epochs: int = 100,
        load_path : str= None ,
        learning_rate: float = 0.002,
        lr : float =0.001,
        weight_decay: float = 1e-3,
        evaluation_epoch_num: int = 10,
        n_jobs: int = 10,
        ema_kimg:int = 500,
        ema_warmup = 0,
        global_steps : int = 0,
        accumulated_times: int=1,
        hidden_size:int = 5,
        loss="mse",
        metric="",
        optimizer_betas :float= (0.9,0.999),
        optimizer="adam",
        seed=999,
        GPU=3,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.lr = lr
        self.n_epochs = n_epochs
        self.ema_warmup =ema_warmup
        self.global_steps =global_steps
        self.n_critic =n_critic
        self.rank = rank
        self.world_size = world_size
        self.metric = metric
        self.d_feat = d_feat
        self.ema =ema
        self.num_layers = num_layers
        self.optimizer_betas = optimizer_betas
        self.dropout = dropout
        self.ema_kimg = ema_kimg
        self.iter_idx  = iter_idx  
        self.load_path = load_path
        self.g_accumulated_times =g_accumulated_times
        self.loss = loss
        self.accumulated_times = accumulated_times
        self.activation = activation
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.seed = seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.evaluation_epoch_num = evaluation_epoch_num
        self.n_jobs = n_jobs
        self.optimizer = optimizer.lower()
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.logger = get_module_logger("TransGANModel")
        self.logger.info("Naive TransGAN:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))

        # Create model
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # gen_optimizer = optim.Adam(generator.parameters(), lr=self.lr, betas=self.optimizer_betas) # 优化器，adam优化generator的参数，学习率为lr,动量参数beats
        # dis_optimizer = optim.Adam(discriminator.parameters(), lr=self.lr, betas=self.optimizer_betas)
        
        # Optimizer
        if optimizer.lower() == "adam":
            self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.optimizer_betas)
            self.dis_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.optimizer_betas)
        elif optimizer.lower() == "gd":
            self.gen_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.optimizer_betas)
            self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.optimizer_betas)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

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
    

    def train_epoch_gan(self, data_loader,gen_avg_param,dis_losses,gen_losses,writer,epoch,plot_graph=False,plot_title="Validation Predictions"):
        # train mode
        print("Generator and discriminator are initialized")

        self.generator.train()
        self.discriminator.train()
        # generator_losses = []
        # discriminator_losses = []
        criterion = nn.BCELoss() # 二分类任务的损失函数 ，输出层激活函数为sigmoid，输出代表正例的概率，与真实标签比较。
        real_data_list = []
        predicted_data_list = []
        epsilon = 0.02
        real_label = 0.9
        fake_label = 0.
        gen_step = 0
        col_set=["KMID","KLOW","OPEN0","Ref($close, -2) / Ref($close, -1) - 1"]

        for iter_idx,sequence_batch in enumerate(tqdm(data_loader)):

            # 取数据
            real_sequence = sequence_batch[:, :, 0:-1].to(self.device) # 取前三个时间特征。 [64,30,158]
            batch_size = real_sequence.size(0) # 读取batch_size,后面比较的时候需要用到生成了batch_size的数据。
            # real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device) #111111.....
            # fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=self.device)
            
            real_sequence = real_sequence.reshape(real_sequence.shape[0],real_sequence.shape[2], 1 ,real_sequence.shape[1],) #把[64,30,3]变为[64,3,1,30]
            real_data_list.append(real_sequence)
            
            # 真实的数据放到判别器
            discriminator_output_real = self.discriminator(real_sequence) # 将真实的值传到鉴别器中，先学习真实的数据。discriminator_output_real.shape(64,1)
            discriminator_output_real_scaled = torch.sigmoid(discriminator_output_real) # 这里的值好像不在0-1之间，用sigmod函数，然后与真实的值进行比较。

            z = torch.cuda.FloatTensor(np.random.normal(0, 1,(real_sequence.shape[0],158,30)))
            generator_input_sequence = z  # generator_input_sequence.shape(64,158,1,30)
            _ = self.generator(generator_input_sequence)  #generator_outputs_sequence 是一个三维的数据
            generator_outputs_sequence = self.generator.saved_output
            
        # generator_result_concat.shape(64,158,1,30*2) 正常情况下拼接是时间步拼接,前半部分是真实,后面的生成.这里时间步是第3维.
            # generator_result_concat = torch.cat((generator_input_sequence, generator_outputs_sequence.detach()), 3) #generator_outputs_sequence.shape(64,158,1,30)
            discriminator_output_fake = self.discriminator(generator_outputs_sequence)
            predicted_data_list.append(generator_outputs_sequence)
            discriminator_output_fake = torch.sigmoid(discriminator_output_fake)

            ##---------------------------------------------------------------------这是mse的损失函数
            # discriminator_error_real = nn.MSELoss()(discriminator_output_real_scaled, real_labels)
            # discriminator_error_fake = nn.MSELoss()(discriminator_output_fake, fake_labels)
            # discriminator_error = discriminator_error_real + discriminator_error_fake
            #--------------------------------------------------------------------------------
            
            ##-----------------------------------------------------------------------------------这是'hinge'的损失函数
            discriminator_error = torch.mean(nn.ReLU(inplace=True)(1.0 - discriminator_output_real_scaled)) + \
                    torch.mean(nn.ReLU(inplace=True)(1 + discriminator_output_fake))
            ##-------------------------------------------------------------------------------------
            
            discriminator_error = discriminator_error/float(self.accumulated_times)  # args.accumulated_times = 1
            discriminator_error.backward() 

            if (iter_idx + 1) % self.accumulated_times == 0:   # iter_idx运行初始为0。累积梯度的次数，即将多个batch的梯度加起来，再执行一次参数更新。
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.)
                self.dis_optimizer.step()
                self.dis_optimizer.zero_grad()

            ##----------------------------------------------------------
                                 # 生成器的训练
            ##---------------------------------------------------------
                
            # 生成的数据和真实的数据拼接在一起,反馈生成器的损失函数.

            # generator_result_concat_grad = torch.cat((generator_input_sequence, generator_outputs_sequence), 3) # 拼接
            
            # z用于保持生成的虚假图像与真实图像一致，而gen_z用于生成多样化的虚假图像样本。(真实数据的batch的，生成的虚假的图片第一维有的不一定相同？)
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1,(128,158,30))) # 注意gen_z和gen的区别。
            gen_z.requires_grad = True
            _ = self.generator(gen_z)  # gen_imgs(64,3,1,150)
            gen_imgs = self.generator.saved_output  #gen_imgs(128,158,1,30)

            # #对生成器生成的数据进行干扰，生成对抗样本 ，使用FGSM
            # # generator_outputs_sequence = fast_gradient_method(self.generator, generator_outputs_sequence , 0.4, np.inf)  #FGM对抗样本分类 x_fgm.shape(128,3,32,32)
            # loss = torch.mean(gen_imgs)  # 可以根据具体情况选择其他的损失函数
            # loss.backward()
            # # 计算FGSM扰动
            # perturbation = epsilon * torch.sign(gen_z.grad.data)  #perturbation.shape(128,158,30)
            # # 添加扰动到生成器的输入
            # perturbed_gen_z = gen_z + perturbation  #perturbed_gen_z.shape(128,158,30)
            # perturbed_gen_z = perturbed_gen_z.reshape(perturbed_gen_z.shape[0],perturbed_gen_z.shape[1], 1 ,perturbed_gen_z.shape[2],)# 变成四维（128，158，1，30）
            
            # discriminator_output = self.discriminator(perturbed_gen_z).view(-1) # 作为输入到鉴别器
            discriminator_output = torch.sigmoid(discriminator_output)
            
            #----------------------------------------------------------这也是'hinge'损失函数
            generator_error = -torch.mean(discriminator_output)
            #-----------------------------------------------------------
            generator_error.backward()
            #-------------------------------------------------------------这是mse的损失函数
            # generator_error = nn.MSELoss()(discriminator_output_fake, real_labels)
            #-------------------------------------------------------------------
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5.)
            self.gen_optimizer.step()
            self.gen_optimizer.zero_grad()
            # writer.add_scalar('g_loss', g_loss.item(), self.global_steps)
            # gen_step += 1
        
        #     dis_losses.append(discriminator_error.item())
        #     gen_losses.append(generator_error.item())
        
        # dis_losses.append(np.mean(dis_losses))
        # gen_losses.append(np.mean(gen_losses))

        # writer.add_scalar('Generator Loss', gen_losses[epoch], epoch)
        # writer.add_scalar('Discriminator Loss', dis_losses[epoch], epoch)

        # # 关闭SummaryWriter对象
        # writer.close()
        # real_data = torch.cat(real_data_list, 0)  #real_data.shape(1600,30,3)
        # # real_data = real_data[0,:,0]
        # predicted_data = torch.cat(predicted_data_list, 0)  #predicted_data(1600,30,3)
        # # predicted_data =predicted_data[0,:,0]
        # real_data = real_data.detach().cpu().numpy() # 将张量的值转换为NumPy数组
        # predicted_data = predicted_data.detach().cpu().numpy() # 将张量的值转换为NumPy数组

        # df_pred = pd.DataFrame(predicted_data.reshape(-1,len(col_set)),columns=col_set)
        # df_real = pd.DataFrame(real_data.reshape(-1,len(col_set)),columns=col_set)

        # real_data = real_data[:,-1,:]
        # predicted_data = predicted_data[:,-1,:]
        # df_pred = pd.DataFrame(predicted_data.reshape(-1,len(col_set[0:-1])),columns=col_set[0:-1])
        # df_real = pd.DataFrame(real_data.reshape(-1,len(col_set[0:-1])),columns=col_set[0:-1])  # real_data的数据类型是array

        # if plot_graph:
        #     if not os.path.exists('./plots/'):
        #         os.makedirs('./plots/')
        
        #     # TODO: get x values and plot prediction of multiple columns
        #     fig = plt.figure(figsize=(16,8))
        #     plt.xlabel("datetime")
        #     plt.ylabel("Close Price")
        #     plt.title(plot_title)
        #     plt.plot(df_real['KLOW'],label="Real")
        #     plt.plot(df_pred['KLOW'],label="Predicted")
        #     # plt.ylim(bottom=0)
        #     plt.legend()
        #     fig.savefig('./plots/plt_epoch_{}.png'.format(epoch))
        #     plt.close(fig)


    def test_epoch_train_data(self, data_loader):

        self.generator.eval()

        scores = []
        losses = []

        for data in data_loader:

            feature = data[:, :, 0:-1].to(self.device).to(self.device) #feature.shape (64,30,3)
            
            label = data[:, -1, -1].to(self.device) #label.shape(64)

            feature = feature.to("cuda:3")
            label = label.to("cuda:3") 

            with torch.no_grad():
                feature = feature.permute(0,2,1)
                _ = self.generator(feature.float())
                pred = self.generator.pre_out # .float(),这里跳进去的函数是”mse“那段/ 这里用transgan的是(64,3,1,30)
                pred = pred.squeeze()
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())
            return np.mean(losses), np.mean(scores)
    
    def test_epoch_valid_data(self, data_loader,):
        
            # global generator
            self.generator.eval()

            scores = []
            losses = []

            report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
            
            for iter_idx,data in enumerate(tqdm(data_loader)):
            # for data in data_loader:

                feature = data[:, :, 0:-1].to(self.device).to(self.device) #feature.shape (64,30,158)
                feature = feature.reshape(feature.shape[0],feature.shape[1], 1 ,feature.shape[2],) #把[64,30,158]变为[64,30,1,158]
                feature = feature.permute(0,3,2,1)
                label = data[:, -1, -1].to(self.device) #label.shape(64)

                feature = feature.to("cuda:3")
                label = label.to("cuda:3") 
                # y = label
                # for i in range(y.size(0)):   
                #     if(y[i]>0):
                #         y[i] = 1      #涨，记作1.
                #     else:
                #         y[i] = 0  
                # _, y_pred = self.generator(feature).max(1)  # model prediction on clean examples  y_pred.shape=(128)  y_pred_fgm.shape(128)
                # x_fgm = fast_gradient_method(self.generator, feature , 0.03, np.inf)  #FGM对抗样本分类 x_fgm.shape(128,3,32,32)
                # x_pgd = projected_gradient_descent(self.generator, feature, 0.3, 0.01, 40, np.inf)
                # model prediction on FGM adversarial examples
                # model prediction on PGD adversarial examples
                # _, y_pred_fgm = self.generator(x_fgm).max(1)
                # # _, y_pred_pgd = self.generator(x_pgd).max(1)
                # report.nb_test += y.size(0)  # report.nb_test = 64
                # report.correct += y_pred.eq(y).sum().item()
                # report.correct_fgm += y_pred_fgm.eq(y).sum().item()
                # report.correct_pgd += y_pred_pgd.eq(y).sum().item()  # eq是比较。通过 y_pred.eq(y) 来比较预测标签和真实标签是否相等，得到的是一个布尔值的 Tensor，其中值为 1 的位置表示预测正确，值为 0 的位置表示预测错误。

                with torch.no_grad():
                    _ = self.generator(feature.float())
                    pred = self.generator.pre_out # .float(),这里跳进去的函数是”mse“那段/ 这里用transgan的是(64,3,1,30)
                    pred = pred.squeeze()
                    loss = self.loss_fn(pred, label)
                    losses.append(loss.item())

                    score = self.metric_fn(pred, label)
                    scores.append(score.item())
            # print("test acc on clean examples (%): {:.3f}".format(report.correct / report.nb_test * 100.0))
            # print("test acc on FGM adversarial examples (%): {:.3f}".format(report.correct_fgm / report.nb_test * 100.0))
            # print("test acc on PGD adversarial examples (%): {:.3f}".format(report.correct_pgd / report.nb_test * 100.0)) 
            return np.mean(losses), np.mean(scores)
    
    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        
        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        
        # global generator
        self.generator.eval()
        
        preds = []

        for data in test_loader:

            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                feature = feature.permute(0,2,1)
                _ = self.generator(feature.float())
                pred = self.generator.pre_out
                pred = pred.squeeze()
                pred= pred.detach().cpu().numpy()

            preds.append(pred)
        return pd.Series(np.concatenate(preds), index=dl_test.get_index())
    

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
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
            dl_valid, batch_size=self.batch_size,shuffle=False, num_workers=self.n_jobs, drop_last=True
        )

        # gen_optimizer = optim.Adam(generator.parameters(), lr=self.lr, betas=self.optimizer_betas) # 优化器，adam优化generator的参数，学习率为lr,动量参数beats
        # dis_optimizer = optim.Adam(discriminator.parameters(), lr=self.lr, betas=self.optimizer_betas)

        save_path = get_or_create_path(save_path)
        
        gen_avg_param = copy_params(self.generator, mode='gpu')
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []
        # 定义空列表
        dis_losses = []
        gen_losses = []

        # 定义一个SummaryWriter对象
        writer = SummaryWriter(log_dir='./logs')

        # train
        self.logger.info("training...")
        self.fitted = True

        params_before_train = copy.deepcopy(self.generator.state_dict())
        ##  训练GAN网络的transformer模型
        for epoch in range(self.n_epochs):
            self.logger.info("Epoch%d:", epoch)
            self.logger.info("Training...")
            train_result = self.train_epoch_gan(train_loader,gen_avg_param,dis_losses,gen_losses,writer,epoch,plot_graph=True)

            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch_train_data(train_loader)
            val_loss, val_score = self.test_epoch_valid_data(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = epoch
                best_param = copy.deepcopy(self.generator.state_dict())
                best_param1 = copy.deepcopy(self.discriminator.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.generator.load_state_dict(best_param)
        self.discriminator.load_state_dict(best_param1)
        torch.save(best_param,save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

class Generator(nn.Module):
    def __init__(self, seq_len=30, patch_size=15, channels=158, num_classes=9, embed_dim=158, depth=3,
                 num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(Generator, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        
        # self.l1 = nn.Linear(self.latent_dim, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.blocks = Gen_TransformerEncoder(
                         depth=self.depth,
                         emb_size = self.embed_dim,
                         drop_p = self.attn_drop_rate,
                         forward_drop_p=self.forward_drop_rate
                        )

        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        )

        self.decoder_layer = nn.Linear(158,1)
        self.linear = nn.Linear(158,2) 

    def forward(self, z):
        # x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        z = z.to(dtype=torch.float32,device="cuda:3")  #(64,158,1,30)
        z = z.squeeze()
        z = z.permute(0,2,1)
        x = (z + self.pos_embed).to(dtype=torch.float32,device="cuda:3") #x.shape(64,30,158)
        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, H, W)  #output.shape为[64,158,1,30]
        self.saved_output = output
        
        out = output.squeeze()
        out = out.permute(0,2,1)
        
        self.pre_out = self.decoder_layer(out[:, -1, :]) # 预测的值,是一个一维的.
        fenlei_out = self.linear(out[:, -1, :])  # 二维的值,用作fgm的格式.
        
        return fenlei_out
    
    
class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=2,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

        
class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])       
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

        
        
class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, n_classes=2):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

    
class PatchEmbedding_Linear(nn.Module):
    #what are the proper parameters set here?
    def __init__(self, in_channels = 158, patch_size = 15, emb_size = 15, seq_length = 30):
        # self.patch_size = patch_size
        super().__init__()
        #change the conv2d parameters here
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1 = 1, s2 = patch_size), # 
            nn.Linear(patch_size*in_channels, emb_size,dtype=torch.float32)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn(3, emb_size))
        self.positions = nn.Parameter(torch.randn(1, 1, emb_size))  # 修改 self.positions 的形状

    def forward(self, x: Tensor) -> Tensor:  # x[64,30,3]
        # x = x.reshape(x.shape[0],x.shape[2], 1 ,x.shape[1], ) # x变成四维(64,158,1,60)  原论文的作者是（64，3，1，150）
        b, _, _, _ = x.shape
        x = x.to(dtype=torch.float32,device="cuda:3")
        x = self.projection(x)  # 经过了projection之后，x变为(64,2,15)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)  #cls+tokens(64,1,15)
        #prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)  # x又变成了(64,3,15)
        # position
        x += self.positions.expand_as(x)
        return x         #x.shape(64,3,15)
        
        
class Discriminator(nn.Sequential):
    def __init__(self, 
                 in_channels=158,
                 patch_size=15,
                 emb_size=15, 
                 seq_length = 30,
                 depth=3, 
                 n_classes=1, 
                 **kwargs):
        super().__init__(
            PatchEmbedding_Linear(in_channels, patch_size, emb_size, seq_length),
            Dis_TransformerEncoder(depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
    
def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten