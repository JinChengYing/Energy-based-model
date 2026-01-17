## Standard libraries
import os
import json
import math
import numpy as np
import random

import torchvision

#from MNIST_pretrained import file_name
from energy_model import DeepEnergyModel
## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib_inline.backend_inline import set_matplotlib_formats

from data_loader import train_loader, test_loader

from Callback import GenerateCallback, SamplerCallback, OutlierCallback

set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## PyTorch
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "C:/EBM/datasets"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "C:/EBM/module"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)



def train_model(**kwargs):
    # 添加CSV logger用于记录训练指标
    from pytorch_lightning.loggers import CSVLogger
    logger = CSVLogger(save_dir=CHECKPOINT_PATH, name="MNIST")
    
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=50,
                         gradient_clip_val=0.1,
                         logger=logger,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                                    GenerateCallback(every_n_epochs=5),
                                    SamplerCallback(every_n_epochs=5),
                                    OutlierCallback(),
                                    LearningRateMonitor("epoch")
                                   ])
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "MNIST.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = DeepEnergyModel(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # No testing as we are more interested in other properties
    return model

if __name__ == '__main__':
    # 创建模型并训练
    model = train_model(
        img_shape=(1, 28, 28),  # MNIST图像形状 (通道, 高度, 宽度)
        batch_size=128,
        hidden_features=32,
        out_dim=1,
        alpha=0.1,
        lr=1e-4,
        beta1=0.0
    )
    print("训练完成！")
    model.to(device)
    pl.seed_everything(43)
    callback = GenerateCallback(batch_size=4, vis_steps=8, num_steps=256)
    imgs_per_step = callback.generate_imgs(model)
    imgs_per_step = imgs_per_step.cpu()
    for i in range(imgs_per_step.shape[1]):
        step_size = callback.num_steps // callback.vis_steps
        imgs_to_plot = imgs_per_step[step_size - 1::step_size, i]
        imgs_to_plot = torch.cat([imgs_per_step[0:1, i], imgs_to_plot], dim=0)
        grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, value_range=(-1, 1),
                                           pad_value=0.5, padding=2)
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.xlabel("Generation iteration")
        plt.xticks([(imgs_per_step.shape[-1] + 2) * (0.5 + j) for j in range(callback.vis_steps + 1)],
                   labels=[1] + list(range(step_size, imgs_per_step.shape[0] + 1, step_size)))
        plt.yticks([])
        folder_path="results"
        file_name=f"example_{i}.png"
        print(f'save{i}')
        full_path = os.path.join(folder_path, file_name)
        plt.savefig(full_path)