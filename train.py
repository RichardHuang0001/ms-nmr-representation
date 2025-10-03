import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


### 以上为调试器代码部分

# <-- [核心] 启动模型训练的主入口脚本
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from box import Box # 使用box库可以方便地通过点访问符访问字典
import numpy as np
import random

from src.data.dataset import SpectraDataset
from src.models.set_transformer import PretrainSetTransformer
from src.training.trainer import Trainer
import logging

def set_seed(seed):
    """设置随机种子以确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"随机种子设置为: {seed}")

def main(config_path):
    """主函数，负责初始化和启动训练。"""
        # 配置日志记录 ---
    logging.basicConfig(
        level=logging.INFO, # 设置日志级别为INFO，这样INFO及以上级别的日志都会被记录
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("--- 启动预训练任务 ---")

    # 1. 加载配置
    with open(config_path, 'r') as f:
        config = Box(yaml.safe_load(f))
    logging.info("配置加载成功。")

    # 1.1 设置随机种子
    set_seed(config.experiment.seed)
    
    # 2. 初始化数据
    full_dataset = SpectraDataset(
        processed_dir=config.data.processed_dir,
        masking_fraction=config.data.masking_fraction
    )
    
    # 划分训练集和验证集
    val_size = int(config.data.val_split_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    logging.info(f"数据集划分完成: {train_size} 个训练样本, {val_size} 个验证样本。")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, num_workers=config.data.num_workers)
    logging.info("数据加载器准备就绪。")

    # 3. 初始化模型
    model = PretrainSetTransformer(
        dim_input=config.model.dim_input,
        dim_output=config.model.dim_output,
        dim_hidden=config.model.dim_hidden,
        num_heads=config.model.num_heads,
        depth=config.model.depth
    )
    logging.info("模型初始化成功。")

    # 4. 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    # 5. 初始化并启动Trainer
    trainer = Trainer(model, optimizer, train_loader, val_loader, config)
    logging.info("训练器初始化成功，开始训练...")
    trainer.train()

    logging.info("--- 训练结束 ---")


if __name__ == "__main__":
    # 在运行前，确保你已经登录了wandb: `wandb login`
    # 并安装了box: `pip install python-box`
    config_path = "configs/pretrain_set_transformer.yaml"
    main(config_path)