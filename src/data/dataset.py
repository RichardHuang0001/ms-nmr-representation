## dataset.py

# <-- [核心] 定义PyTorch的Dataset和DataLoader，负责数据的加载和预处理
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

class SpectraDataset(Dataset):
    """
    用于多模态光谱数据的PyTorch自定义数据集。
    本数据集的核心任务是为自监督预训练动态地创建掩码样本 (masked samples)。
    其工作方式类似于BERT中的掩码语言模型 (Masked Language Model)。
    """
    def __init__(self, processed_dir: str, masking_fraction: float = 0.15):
        """
        初始化数据集对象。

        :param processed_dir: 字符串，指向存放预处理好的 `.pt` 数据文件的目录。
        :param masking_fraction: 浮点数，定义了在预训练任务中，需要从真实峰中随机掩码的比例。
        """
        self.processed_path = Path(processed_dir)
        self.masking_fraction = masking_fraction
        
        # 将所有预处理好的样本从磁盘加载到内存中。
        # 注意：这种策略适用于中小型数据集。如果数据集规模巨大（例如，超过几十GB），
        # 为了避免内存溢出，可能需要优化为“懒加载”（lazy loading）模式，即在需要时才加载单个文件。
        self.samples = []
        pt_files = sorted(list(self.processed_path.glob("*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"在目录 '{processed_dir}' 中未找到任何预处理好的 .pt 文件。请先运行数据预处理脚本。")
        
        logging.info(f"正在从 {len(pt_files)} 个文件中加载数据...")
        for pt_file in pt_files:
            # torch.load用于反序列化保存的PyTorch对象。
            self.samples.extend(torch.load(pt_file))
        
        logging.info(f"✅ 数据加载完成，共计 {len(self.samples)} 个样本。")

    def __len__(self):
        """返回数据集中样本的总数。DataLoader会使用此方法来确定数据集的大小。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据给定的索引 `idx` 获取单个数据样本，并为其动态创建掩码版本以用于预训练。
        这是该数据集类最核心的逻辑所在。

        :param idx: 整数，请求的样本索引。
        :return: 一个元组，包含三个张量：
                 - input_tensor (掩码后的输入)
                 - attention_mask (原始的注意力掩码)
                 - labels_tensor (用于计算损失的标签)
        """
        # 1. 获取原始样本
        sample = self.samples[idx]
        # 使用 .clone() 创建一个张量的副本，以避免在后续的掩码操作中修改原始缓存数据。
        input_tensor = sample['input_tensor'].clone() 
        attention_mask = sample['attention_mask']
        
        # 2. 准备用于存放标签的张量 (labels_tensor)
        # 我们初始化一个与输入形状相同，但所有值都为-100的张量。
        # 在PyTorch的损失函数（如CrossEntropyLoss）中，-100是一个特殊的“忽略索引”，
        # 意味着在计算损失时，这些位置的预测值不会被考虑。
        labels_tensor = torch.full_like(input_tensor, -100)
        
        # 3. 确定哪些位置是可以被掩码的 "真实峰"
        # attention_mask中值为True（或1）的位置对应于真实的、非填充的光谱峰。
        real_peak_indices = torch.where(attention_mask)[0]
        num_real_peaks = len(real_peak_indices)
        
        # 如果样本中没有任何真实峰（例如，是一个空的或异常的样本），则直接返回原始数据，不进行掩码。
        if num_real_peaks == 0:
            return input_tensor, attention_mask, labels_tensor
            
        # 4. 根据设定的比例，计算需要掩码的峰的数量
        num_to_mask = int(num_real_peaks * self.masking_fraction)
        
        # 确保至少有一个峰被掩码（只要样本中存在真实峰），这对于样本量较小或masking_fraction较低时很重要。
        if num_to_mask == 0 and num_real_peaks > 0:
            num_to_mask = 1
            
        # 5. 随机选择要被掩码的峰的索引
        # `torch.randperm` 会生成一个随机的索引排列，我们从中选取前 `num_to_mask` 个。
        perm = torch.randperm(num_real_peaks)
        masked_indices = real_peak_indices[perm[:num_to_mask]]
        
        # 6. 执行核心的掩码操作
        for i in masked_indices:
            # a. 在标签张量中，将被掩码位置的原始向量记录下来。
            # 这是模型需要学习预测的“正确答案”（ground truth）。
            labels_tensor[i] = input_tensor[i]
            
            # b. 在输入张量中，将被选中的峰的向量替换为一个全零向量。
            # 这个全零向量在功能上充当了[MASK]标记，它告诉模型这个位置的信息丢失了。
            # 在模型内部，这个零向量可以被进一步处理，例如替换为一个可学习的[MASK]嵌入向量。
            input_tensor[i] = torch.zeros(input_tensor.shape[1])

        # 返回处理好的三元组，供DataLoader打包成批次
        return input_tensor, attention_mask, labels_tensor


# --- 用于演示和调试的示例代码 ---
# 当直接运行 `python src/data/dataset.py` 时，以下代码块会被执行。
if __name__ == "__main__":
    # 这是一个示例，展示了如何实例化并使用上面定义的SpectraDataset类。
    
    # 假设预处理好的数据存放在 'data/processed' 目录下。
    # 如果该目录不存在或为空，需要先运行 `src/data/preprocess.py` 脚本。
    
    # 检查是否有可用的数据文件
    processed_dir = "data/processed"
    if not any(Path(processed_dir).glob("*.pt")):
        print(f"❌ 警告: 在 '{processed_dir}' 目录下未找到任何.pt文件。")
        print("请先运行 'python src/data/preprocess.py' 来生成数据。")
    else:
        try:
            # 1. 创建数据集实例
            dataset = SpectraDataset(processed_dir=processed_dir, masking_fraction=0.2) # 使用稍高的掩码率进行测试
            
            # 2. 使用DataLoader来封装数据集，以便进行批次化、多进程加载等操作
            # num_workers > 0 可以开启多进程并行加载数据，加快速度
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
            
            # 3. 从dataloader中迭代获取一个批次的数据进行检查
            masked_inputs, masks, labels = next(iter(dataloader))
            
            # 4. 打印输出张量的形状，以验证数据加载是否正确
            print("\n--- 数据加载器输出检查 ---")
            print(f"批次大小 (Batch Size): {masked_inputs.shape[0]}")
            print(f"掩码后输入的形状 (masked_inputs): {masked_inputs.shape} -> [批次大小, 序列长度, 特征维度]")
            print(f"注意力掩码的形状 (masks): {masks.shape} -> [批次大小, 序列长度]")
            print(f"标签的形状 (labels): {labels.shape} -> [批次大小, 序列长度, 特征维度]")
            
            # 5. 深入检查批次中的第一个样本，验证掩码逻辑
            print("\n--- 单个样本检查 (样本0) ---")
            sample_input = masked_inputs[0]
            sample_labels = labels[0]
            
            # 找到那些在输入中是零向量（可能被掩码），并且在标签中不是-100（确实被掩码）的位置
            # `torch.all(sample_input == 0, dim=1)` 检查特征维度上是否全为0
            # `(sample_labels != -100).any(dim=1)` 检查特征维度上是否有任何一个值不是-100
            masked_positions = torch.where(torch.all(sample_input == 0, dim=1) & (sample_labels != -100).any(dim=1))[0]
            
            print(f"在样本0中，共发现了 {len(masked_positions)} 个被掩码的峰。")
            if len(masked_positions) > 0:
                # 随机挑一个被掩码的位置进行展示
                idx_to_check = masked_positions[0]
                print(f"例如，位置 {idx_to_check} 被掩码了:")
                print(f"  - 输入向量 (input):  {sample_input[idx_to_check].tolist()}  <-- 被置为全零")
                print(f"  - 标签向量 (label): {sample_labels[idx_to_check].tolist()} <-- 保存了原始值")

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"在测试过程中发生未知错误: {e}")