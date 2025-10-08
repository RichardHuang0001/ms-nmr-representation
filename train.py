# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# =============================================================================
# 核心依赖库导入
# =============================================================================
import torch                    # PyTorch深度学习框架
import yaml                     # YAML配置文件解析
from pathlib import Path        # 路径操作工具
from torch.utils.data import DataLoader, random_split  # 数据加载和数据集划分
from box import Box             # 使用box库可以方便地通过点访问符访问字典
import numpy as np              # 数值计算库
import random                   # 随机数生成

# =============================================================================
# 项目自定义模块导入
# =============================================================================
from src.data.dataset import SpectraDataset              # 自定义光谱数据集类
from src.models.set_transformer import PretrainSetTransformer  # Set Transformer预训练模型
from src.training.trainer import Trainer                 # 训练器类，封装训练逻辑
import logging                    # 日志记录模块

# =============================================================================
# 随机种子设置函数
# =============================================================================
def set_seed(seed):
    """
    设置随机种子以确保实验可复现。
    
    参数:
        seed (int): 随机种子值
        
    功能:
        - 设置Python内置random模块的种子
        - 设置NumPy的随机种子
        - 设置PyTorch的CPU随机种子
        - 如果CUDA可用，设置所有GPU的随机种子
    """
    random.seed(seed)              # Python内置随机数生成器
    np.random.seed(seed)           # NumPy随机数生成器
    torch.manual_seed(seed)        # PyTorch CPU随机数生成器
    if torch.cuda.is_available():  # 如果CUDA可用
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    logging.info(f"随机种子设置为: {seed}")

# =============================================================================
# 主训练函数
# =============================================================================
def main(config_path):
    """
    主函数，负责初始化和启动训练。
    
    参数:
        config_path (str): 配置文件路径
        
    功能:
        1. 配置日志系统
        2. 加载YAML配置文件
        3. 设置随机种子
        4. 初始化数据集和数据加载器
        5. 初始化模型和优化器
        6. 启动训练过程
    """
    # =============================================================================
    # 步骤1: 配置日志记录系统
    # =============================================================================
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为INFO，这样INFO及以上级别的日志都会被记录
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("--- 启动预训练任务 ---")

    # =============================================================================
    # 步骤2: 加载配置文件
    # =============================================================================
    with open(config_path, 'r') as f:
        config = Box(yaml.safe_load(f))  # 使用Box将YAML转换为可点访问的对象
    logging.info("配置加载成功。")

    # =============================================================================
    # 步骤3: 设置随机种子以确保实验可复现
    # =============================================================================
    set_seed(config.experiment.seed)
    
    # =============================================================================
    # 步骤4: 初始化数据集
    # =============================================================================
    full_dataset = SpectraDataset(
        processed_dir=config.data.processed_dir,        # 预处理后的数据目录
        masking_fraction=config.data.masking_fraction   # 掩码比例，用于自监督学习
    )
    
    # =============================================================================
    # 步骤5: 划分训练集和验证集
    # =============================================================================
    val_size = int(config.data.val_split_ratio * len(full_dataset))  # 计算验证集大小
    train_size = len(full_dataset) - val_size                        # 计算训练集大小
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])  # 随机划分数据集
    logging.info(f"数据集划分完成: {train_size} 个训练样本, {val_size} 个验证样本。")

    # =============================================================================
    # 步骤6: 创建数据加载器
    # =============================================================================
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size,  # 批量大小
        shuffle=True,                           # 训练时打乱数据顺序
        num_workers=config.data.num_workers,    # 多进程数据加载
        pin_memory=True                         # 将数据固定在内存中，加速GPU传输
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.training.batch_size,  # 验证时使用相同的批量大小
        num_workers=config.data.num_workers     # 多进程数据加载
    )
    logging.info("数据加载器准备就绪。")

    # =============================================================================
    # 步骤7: 初始化Set Transformer模型
    # =============================================================================
    model = PretrainSetTransformer(
        dim_input=config.model.dim_input,    # 输入维度 (12维峰向量)
        dim_output=config.model.dim_output,  # 输出维度 (12维峰向量)
        dim_hidden=config.model.dim_hidden,  # 隐藏层维度 (256)
        num_heads=config.model.num_heads,    # 注意力头数 (8)
        depth=config.model.depth             # Transformer层数 (6)
    )
    logging.info("模型初始化成功。")

    # =============================================================================
    # 步骤8: 初始化优化器
    # =============================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),                    # 模型的所有可训练参数
        lr=config.training.learning_rate      # 学习率 (0.0001)
    )

    # =============================================================================
    # 步骤9: 初始化训练器并启动训练
    # =============================================================================
    trainer = Trainer(
        model,           # Set Transformer模型
        optimizer,       # AdamW优化器
        train_loader,    # 训练数据加载器
        val_loader,      # 验证数据加载器
        config           # 配置对象
    )
    logging.info("训练器初始化成功，开始训练...")
    trainer.train()      # 启动训练循环

    logging.info("--- 训练结束 ---")


# =============================================================================
# 程序入口点
# =============================================================================
if __name__ == "__main__":
    """
    程序主入口点
    
    运行前准备:
    1. 确保已登录wandb: `wandb login`
    2. 确保已安装python-box: `pip install python-box`
    3. 确保已激活conda环境: `conda activate spectra`
    """
    config_path = "configs/pretrain_set_transformer.yaml"  # 配置文件路径
    main(config_path)  # 调用主函数开始训练