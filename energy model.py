from itertools import batched
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from Sampler import Sampler
from CNN model import CNNModel
import torch


class DeepEnergyModel(pl.LightningModule):
    def __init__(self, img_shape,batch_size, alpha=0.1, lr=1e-4, beta1=0.0,**CNN_args):
        super().__init__()#初始化
        self.save_hyperparameters()

        self.cnn=CNNModel(**CNN_args)#传入CNNmodel的参数调用CNN
        self.sampler=Sampler(self.cnn,img_shape=img)#得到采样结果# 实例化采样器，负责通过 MCMC（马尔可夫链蒙特卡洛）生成“假”样本
        self.example_input_array=torch.zeros(1,*img_shape)

        #前向
        def forward(self, x):
            return self.cnn(x)#本质上能量模型用的网络是CNN,训练得分网络
            return z

        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        #这里处理优化器
        optimizer=optim.Adam(self.parameters(),lr=self.hparams.lr, betas=(self.hparams.beta1,0.999))
        scheduler=optim.lr_scheduler.StepLR(optimizer,1,gamma=0.97)#随着epoch指数衰减学习率, 避免over shooting
        return [optimizer,scheduler]

        #训练
    def training_step(self,batch,batch_idx):#batch_idx可以控制批次号, 学习率等
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs,_=batch#从 batch 中取第一个元素赋给 real_imgs，第二个元素丢弃不用
        small_noise = torch.randn_like(real_imgs)*0.005
        real_imgs=real_imgs+small_noise.add_(small_noise).clamp_(min=-1.0, max=1.0)

        #获取样本,使用60步Langevin采样
        fake_imgs=self.sampler.sample_new_exmps(steps=60,step_size=10)

        #预测所有图像的能量score
        inp_imgs=torch.cat([real_imgs,fake_imgs],dim=0)#在0维度上拼接真假图片
        real_out , fake_out=self.cnn(inp_imgs).chunk(2,dim=0)

        #Calculate Losses
        reg_loss = self.hparams.alpha*(real_out ** 2+ fake_out**2).mean()
        cdiv+loss= fake_out.mean()- real_out.mean()
        loss= reg_loss+cdiv_loss

        #logging
        self.log('loss', loss)
        self.log('loss_regularization', reg_loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('metrics_avg_real', real_out.mean())
        self.log('metrics_avg_fake', fake_out.mean())
        return loss

        #validation步骤
    def validation_step(self,batch,batch_idx):
        #使用对比散度
        # For validating, we calculate the contrastive divergence between purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends on what we are interested in the model
        real_imgs,_=batch
        fake_imgs= torch.rand_like(real_imgs)*2-1

        inp_imgs=torch.cat([real_imgs,fake_imgs],dim=0)
        real_out,fake_out=self.cnn(inp_imgs).chunk(2,dim=0)

        cdiv= fake_out.mean()- real_out.mean()
        self.log('val_constrastive_divergence', cdiv)
        self.log('metrics_avg_fake', fake_out.mean())
        self.log('val_real_out', real_out.mean())


