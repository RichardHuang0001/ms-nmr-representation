# <-- [核心] 定义PyTorch的Dataset和DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

class SpectraDataset(Dataset):
    """
    用于多模态光谱数据的PyTorch自定义数据集。
    负责加载预处理好的数据，并在运行中动态创建掩码样本以进行自监督预训练。
    """
    def __init__(self, processed_dir: str, masking_fraction: float = 0.15):
        """
        初始化数据集。

        :param processed_dir: 存放预处理好的 .pt 文件的目录。
        :param masking_fraction: 预训练时随机掩码的真实峰的比例。
        """
        self.processed_path = Path(processed_dir)
        self.masking_fraction = masking_fraction
        
        # 加载所有预处理好的样本到内存中
        # 注意：如果数据集非常巨大（> 几十GB），这里可能需要优化为懒加载
        self.samples = []
        pt_files = sorted(list(self.processed_path.glob("*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"在目录 '{processed_dir}' 中未找到任何预处理好的 .pt 文件。")
        
        logging.info(f"正在从 {len(pt_files)} 个文件中加载数据...")
        for pt_file in pt_files:
            self.samples.extend(torch.load(pt_file))
        
        logging.info(f"✅ 数据加载完成，共计 {len(self.samples)} 个样本。")

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本，并为其创建掩码版本以进行预训练。
        这是本类的核心逻辑所在。
        """
        # 1. 获取原始样本
        sample = self.samples[idx]
        input_tensor = sample['input_tensor'].clone() # 使用.clone()避免修改原始数据
        attention_mask = sample['attention_mask']
        
        # 2. 准备标签张量
        # 初始化一个全为-100的标签张量。-100是PyTorch损失函数中默认的忽略索引
        labels_tensor = torch.full_like(input_tensor, -100)
        
        # 3. 确定可以被掩码的真实峰的索引
        # attention_mask中为True的位置代表真实峰
        real_peak_indices = torch.where(attention_mask)[0]
        num_real_peaks = len(real_peak_indices)
        
        # 如果没有真实峰，则直接返回原始数据
        if num_real_peaks == 0:
            return input_tensor, attention_mask, labels_tensor
            
        # 4. 计算需要掩码的峰的数量
        num_to_mask = int(num_real_peaks * self.masking_fraction)
        if num_to_mask == 0:
             # 确保至少掩码一个峰（如果存在的话）
            num_to_mask = 1
            
        # 5. 随机选择要掩码的峰
        # 从真实峰的索引中随机抽取
        perm = torch.randperm(num_real_peaks)
        masked_indices = real_peak_indices[perm[:num_to_mask]]
        
        # 6. 执行掩码操作
        for i in masked_indices:
            # a. 在标签张量中，记录下被掩码的峰的原始向量
            labels_tensor[i] = input_tensor[i]
            
            # b. 在输入张量中，将被选中的峰替换为全零向量（作为[MASK]的占位符）
            # 后续模型可以将这个零向量替换为可学习的[MASK]嵌入
            input_tensor[i] = torch.zeros(input_tensor.shape[1])

        return input_tensor, attention_mask, labels_tensor


# --- 用于演示和调试的示例代码 ---
if __name__ == "__main__":
    # 这是一个示例，展示如何使用我们创建的Dataset类
    
    # 假设你已经在 'data/processed' 目录下生成了.pt文件
    # 如果没有，你可以先运行 preprocess.py
    
    # 检查是否有可用的数据文件
    processed_dir = "data/processed"
    if not any(Path(processed_dir).glob("*.pt")):
        print(f"❌ 警告: 在 '{processed_dir}' 目录下未找到任何.pt文件。")
        print("请先运行 'python src/data/preprocess.py' 来生成数据。")
    else:
        try:
            # 创建数据集实例
            dataset = SpectraDataset(processed_dir=processed_dir)
            
            # 使用DataLoader来创建数据批次
            # num_workers > 0 可以开启多进程加载
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
            
            # 从dataloader中取出一个批次的数据进行检查
            masked_inputs, masks, labels = next(iter(dataloader))
            
            print("\n--- 数据加载器输出检查 ---")
            print(f"批次大小: {masked_inputs.shape[0]}")
            print(f"掩码后输入的形状 (masked_inputs): {masked_inputs.shape}")
            print(f"注意力掩码的形状 (masks): {masks.shape}")
            print(f"标签的形状 (labels): {labels.shape}")
            
            # 检查一个样本
            print("\n--- 单个样本检查 (样本0) ---")
            sample_input = masked_inputs[0]
            sample_labels = labels[0]
            
            # 找到被掩码的位置
            masked_positions = torch.where(torch.all(sample_input == 0, dim=1) & (sample_labels != -100).any(dim=1))[0]
            
            print(f"在样本0中，发现了 {len(masked_positions)} 个被掩码的峰。")
            if len(masked_positions) > 0:
                idx_to_check = masked_positions[0]
                print(f"例如，位置 {idx_to_check} 被掩码了:")
                print(f"  - 输入向量 (input): {sample_input[idx_to_check].tolist()}")
                print(f"  - 标签向量 (label): {sample_labels[idx_to_check].tolist()}")

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"发生未知错误: {e}")