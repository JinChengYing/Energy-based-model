"""
训练曲线可视化脚本
从PyTorch Lightning的CSV日志中读取数据并绘制训练曲线
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

CHECKPOINT_PATH = "C:/EBM/module"

def plot_training_curves(logs_dir=None, save_path="training_curves.png"):
    """从CSV日志绘制训练曲线"""
    if logs_dir is None:
        logs_dir = os.path.join(CHECKPOINT_PATH, "MNIST")
    
    if not os.path.exists(logs_dir):
        print(f"未找到日志目录: {logs_dir}")
        return
    
    # 查找最新的日志版本
    versions = [d for d in os.listdir(logs_dir) if d.startswith("version_")]
    if not versions:
        print("未找到训练日志")
        return
    
    latest_version = max(versions, key=lambda x: int(x.split("_")[1]))
    metrics_file = os.path.join(logs_dir, latest_version, "metrics.csv")
    
    if not os.path.exists(metrics_file):
        print(f"未找到metrics文件: {metrics_file}")
        return
    
    print(f"读取训练日志: {metrics_file}")
    df = pd.read_csv(metrics_file)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('训练曲线', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线
    ax = axes[0, 0]
    if 'loss' in df.columns:
        train_loss = df['loss'].dropna()
        if len(train_loss) > 0:
            ax.plot(train_loss.index, train_loss.values, label='training loss', color='blue', linewidth=2)
    if 'val_contrastive_divergence' in df.columns:
        val_cdiv = df['val_contrastive_divergence'].dropna()
        if len(val_cdiv) > 0:
            ax.plot(val_cdiv.index, val_cdiv.values, label='constract divergence validation', color='red', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 对比散度
    ax = axes[0, 1]
    if 'loss_contrastive_divergence' in df.columns:
        cdiv = df['loss_contrastive_divergence'].dropna()
        if len(cdiv) > 0:
            ax.plot(cdiv.index, cdiv.values, label='training divergence', color='green', linewidth=2)
    if 'val_contrastive_divergence' in df.columns:
        val_cdiv = df['val_contrastive_divergence'].dropna()
        if len(val_cdiv) > 0:
            ax.plot(val_cdiv.index, val_cdiv.values, label='validation constra divergence', color='orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('constra divergence')
    ax.set_title('divergence curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 正则化损失
    ax = axes[0, 2]
    if 'loss_regularization' in df.columns:
        reg_loss = df['loss_regularization'].dropna()
        if len(reg_loss) > 0:
            ax.plot(reg_loss.index, reg_loss.values, label='regu loss', color='purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('regu loss')
    ax.set_title('regu loss curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 真实样本平均能量
    ax = axes[1, 0]
    if 'metrics_avg_real' in df.columns:
        real_energy = df['metrics_avg_real'].dropna()
        if len(real_energy) > 0:
            ax.plot(real_energy.index, real_energy.values, label='real sample', color='blue', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('average energy')
    ax.set_title('real sample average energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 生成样本平均能量
    ax = axes[1, 1]
    if 'metrics_avg_fake' in df.columns:
        fake_energy = df['metrics_avg_fake'].dropna()
        if len(fake_energy) > 0:
            ax.plot(fake_energy.index, fake_energy.values, label='generate sample', color='red', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('average energy')
    ax.set_title('average generation energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 能量对比
    ax = axes[1, 2]
    if 'metrics_avg_real' in df.columns and 'metrics_avg_fake' in df.columns:
        real_energy = df['metrics_avg_real'].dropna()
        fake_energy = df['metrics_avg_fake'].dropna()
        common_idx = real_energy.index.intersection(fake_energy.index)
        if len(common_idx) > 0:
            ax.plot(common_idx, real_energy.loc[common_idx].values, label='real sample', color='blue', linewidth=2)
            ax.plot(common_idx, fake_energy.loc[common_idx].values, label='generate sample', color='red', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('average energy')
    ax.set_title('comparison of energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


if __name__ == '__main__':
    plot_training_curves()
