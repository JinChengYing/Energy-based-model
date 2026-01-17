import torch
import numpy as np
import random

class Sampler:

    def __init__(self,model, img_shape,sample_size,steps,max_len=8192):
        """
               Inputs:
                   model - Neural network to use for modeling E_theta
                   img_shape - Shape of the images to model
                   sample_size - Batch size of the samples
                   max_len - Maximum number of data points to keep in the buffer
               """
        #super().__init__(): 调用父类的初始化方法。虽
        # 然在这个代码片段中 Sampler 看起来没有继承特定类，
        # 但在复杂的框架（如 PyTorch Lightning）中，这确保了父类的属性被正确初始化
        self.model = model
        self.img_shape = img_shape
        self.steps = steps
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples=[(torch.rand((1,)+img_shape)*2-1) for i in range(sample_size)]

    def sample_new_exmps(self, steps=60, step_size=10):
        """
               Function for getting a new batch of "fake" images.
               Inputs:
                   steps - Number of iterations in the MCMC algorithm
                   step_size - Learning rate nu in the algorithm above
               """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05) #确定随机扔掉的样本占5%数量
        rand_imgs = torch.rand((n_new,)+self.img_shape)*2-1 #生成 纯随机噪声图像
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0) #随机扔掉一些旧的样本
        device = next(self.model.parameters()).device
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device) #合并新旧样本为新的batch 扔到gpu里面

        #做MCMC采样,这里是langevin 采样

        inp_imgs=Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)


        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        # 将gpu处理好的采样数据扔到cpu里面, 并且沿着dim0即batch维度方向切割样本, 再放到examples列表的前面, 旧样本按照顺序放到后面
        self.examples = self.examples[:self.max_len]
        #限制列表长度, 避免内存溢出, 放在后面的旧样本会被截断.
        return inp_imgs

    @staticmethod
    def generate_samples(model, inp_imgs, steps=60, step_size=10, return_img_per_step=False):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training#当模型在训练, 会赋值true,反之复制 false
        model.eval()#设置为评估模式, 冻结参数 , 避免drop out
        for p in model.parameters():
            p.requires_grad = False #冻结模型参数梯度
        inp_imgs.requires_grad = True #保留像素梯度, 使得能量降低.

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs
