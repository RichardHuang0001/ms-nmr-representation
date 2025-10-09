## preprocess.py
"""
多模态光谱数据预处理脚本

本脚本是MS-NMR表示学习项目的核心数据预处理模块，负责将原始的多模态光谱数据
（质谱MS和核磁共振NMR）转换为统一的Peak Vector表示，用于后续的Set Transformer模型训练。

项目背景：
- 目标：学习分子结构的多模态光谱表示，用于分子结构解析
- 数据来源：包含79万个分子的多模态光谱数据集，包括¹H-NMR、¹³C-NMR、MS/MS等
- 模型架构：基于Set Transformer的自监督预训练模型，类似BERT的掩码语言模型

数据流程：
原始Parquet文件 → Peak Vector序列 → PyTorch张量 → Set Transformer模型
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from multiprocessing import Pool, cpu_count
import functools

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================================================================================================
# Peak Vector 数据表示设计
# ================================================================================================
"""
Peak Vector是本项目的核心数据表示，将不同模态的光谱峰统一编码为12维向量。
这种设计允许Set Transformer模型同时处理NMR和MS峰，学习跨模态的分子表示。

Peak Vector结构（12维）：
索引 0-1:  模态标识符 (Modality Indicators)
  [0] is_NMR: 1.0表示NMR峰，0.0表示MS峰
  [1] is_MS:  1.0表示MS峰，0.0表示NMR峰

索引 2-5:  数值特征 (Numerical Features)
  [2] position_normalized:    归一化的化学位移/质荷比 (0-1范围)
  [3] ms_position_normalized: MS特有的归一化m/z值 (仅MS峰使用)
  [4] intensity_normalized:   归一化的强度值 (0-1范围)
  [5] integration_value:      积分值/氢原子数 (主要用于¹H-NMR)

索引 6-11: 多重性编码 (Multiplicity Encoding) - 仅用于¹H-NMR峰
  [6] is_singlet:    单重峰 (s)
  [7] is_doublet:    双重峰 (d)  
  [8] is_triplet:    三重峰 (t)
  [9] is_quartet:    四重峰 (q)
  [10] is_multiplet: 多重峰 (m)
  [11] is_other:     其他类型

设计原理：
1. 统一表示：不同模态的峰都用相同维度的向量表示，便于Set Transformer处理
2. 模态区分：通过前两位的one-hot编码区分NMR和MS峰
3. 特征归一化：所有数值特征都归一化到[0,1]范围，提高训练稳定性
4. 稀疏编码：每个峰只使用部分维度，其余维度为0，符合实际物理意义
"""

PEAK_VECTOR_DIM = 12  # Peak Vector的总维度

# ¹H-NMR多重性类型映射表
# 多重性反映了氢原子的化学环境和耦合关系，是重要的结构信息
MULTIPLICITY_MAP = {
    's': 0,   # 单重峰 (singlet) - 无耦合
    'd': 1,   # 双重峰 (doublet) - 与1个氢耦合
    't': 2,   # 三重峰 (triplet) - 与2个氢耦合
    'q': 3,   # 四重峰 (quartet) - 与3个氢耦合
    'm': 4    # 多重峰 (multiplet) - 复杂耦合
    # 索引5将用于'other'类型
}

def normalize_values(values):
    """
    对数值列表进行最小-最大归一化 (Min-Max Normalization)
    
    将输入值缩放到[0, 1]范围内，这是机器学习中常用的特征缩放技术。
    归一化的目的是：
    1. 消除不同特征之间的量纲差异（如ppm vs m/z）
    2. 提高神经网络训练的稳定性和收敛速度
    3. 防止某些特征因数值范围大而主导模型学习
    
    参数:
        values (list): 待归一化的数值列表
        
    返回:
        list: 归一化后的数值列表，范围在[0, 1]之间
        
    边界情况处理:
        - 空列表：返回空列表
        - 单个值：返回[0.0]
        - 所有值相同：返回全零列表（避免除零错误）
    """
    if not values or len(values) < 2 or max(values) == min(values):
        return [0.0] * len(values)
    
    min_val, max_val = min(values), max(values)
    range_val = max_val - min_val
    
    # Min-Max归一化公式: (x - min) / (max - min)
    return [(v - min_val) / range_val for v in values]

def process_single_sample(row_tuple, ms_col_name):
    """
    处理单个分子样本，将其多模态光谱峰转换为统一的Peak Vector表示
    
    这是数据预处理的核心函数，负责将原始的NMR和MS峰数据转换为模型可以处理的
    统一向量表示。每个分子的所有峰（无论来自哪种光谱技术）都被编码为相同维度的向量。
    
    输入数据结构：
    - ¹H-NMR峰：包含化学位移(delta)、氢原子数(nH)、多重性(type)等信息
    - ¹³C-NMR峰：包含化学位移(delta)、强度(intensity)等信息  
    - MS/MS峰：包含质荷比(m/z)和相对强度的二元组列表
    
    输出：
    - Peak Vector列表：每个峰对应一个12维向量，包含模态标识、位置、强度等信息
    
    参数:
        row_tuple (tuple): (index, row)元组，其中row包含单个分子的所有光谱数据
        ms_col_name (str): 要处理的MS数据列名（如'msms_positive_20ev'）
        
    返回:
        list: Peak Vector列表，每个元素是长度为12的浮点数列表
        None: 如果样本中没有任何有效峰
    """
    index, row = row_tuple
    all_peak_vectors = []  # 存储该分子的所有Peak Vector
    
    # ================================================================================================
    # 第一步：提取和预处理NMR峰数据
    # ================================================================================================
    """
    NMR（核磁共振）数据包含两种类型：
    1. ¹H-NMR：氢原子的化学环境信息，包含化学位移、多重性、积分值等
    2. ¹³C-NMR：碳原子的化学环境信息，包含化学位移和强度
    
    这些信息对于确定分子结构至关重要，不同的化学环境会产生不同的化学位移。
    """
    h_nmr_peaks = row.get('h_nmr_peaks', [])  # ¹H-NMR峰列表
    c_nmr_peaks = row.get('c_nmr_peaks', [])  # ¹³C-NMR峰列表
    
    # 数据类型转换：处理从Parquet文件加载的numpy数组
    if isinstance(h_nmr_peaks, np.ndarray):
        h_nmr_peaks = h_nmr_peaks.tolist()
    if isinstance(c_nmr_peaks, np.ndarray):
        c_nmr_peaks = c_nmr_peaks.tolist()

    # 数据完整性检查：确保数据格式正确
    if not isinstance(h_nmr_peaks, list):
        logging.warning(f"样本 {index} 的 'h_nmr_peaks' 类型不正确 (实际为 {type(h_nmr_peaks)})，已置为空列表。")
        h_nmr_peaks = []
    if not isinstance(c_nmr_peaks, list):
        logging.warning(f"样本 {index} 的 'c_nmr_peaks' 类型不正确 (实际为 {type(c_nmr_peaks)})，已置为空列表。")
        c_nmr_peaks = []

    # 收集所有NMR峰的位置和强度信息，用于后续的归一化
    # 注意：¹H-NMR和¹³C-NMR使用不同的键名来存储化学位移
    all_nmr_positions = [p.get('delta', 0.0) for p in h_nmr_peaks] + [p.get('delta (ppm)', 0.0) for p in c_nmr_peaks]
    all_nmr_intensities = [p.get('intensity', 0.0) for p in h_nmr_peaks] + [p.get('intensity', 0.0) for p in c_nmr_peaks]
    
    # 计算归一化参数：在样本内进行归一化，确保不同分子的峰值范围一致
    min_ppm, max_ppm = (min(all_nmr_positions), max(all_nmr_positions)) if all_nmr_positions else (0, 0)
    min_int_nmr, max_int_nmr = (min(all_nmr_intensities), max(all_nmr_intensities)) if all_nmr_intensities else (0, 0)

    # ================================================================================================
    # 第二步：处理¹H-NMR峰，转换为Peak Vector
    # ================================================================================================
    """
    ¹H-NMR峰包含丰富的结构信息：
    - 化学位移(delta)：反映氢原子的化学环境，单位为ppm
    - 氢原子数(nH)：该峰对应的氢原子数量，用于定量分析
    - 多重性(type)：反映氢原子与相邻氢原子的耦合关系，如单重峰(s)、双重峰(d)等
    
    Peak Vector编码方式：
    - [0] = 1.0：标识为NMR峰
    - [2] = 归一化化学位移
    - [5] = 氢原子数（积分值）
    - [6-11] = 多重性的one-hot编码
    """
    for peak in h_nmr_peaks:
        vec = [0.0] * PEAK_VECTOR_DIM  # 初始化12维零向量
        
        # 设置模态标识：这是一个NMR峰
        vec[0] = 1.0  # is_NMR = True
        
        # 提取并归一化化学位移
        position = peak.get('delta')  # 化学位移，单位ppm
        if position is None:
            logging.debug(f"样本 {index}, H-NMR 峰缺少 'delta' 键。")
            position = 0.0
        
        # 在样本内归一化化学位移到[0,1]范围
        norm_pos = (position - min_ppm) / (max_ppm - min_ppm) if max_ppm > min_ppm else 0.0
        vec[2] = norm_pos  # position_normalized
        
        # 提取氢原子数（积分值）
        # 注意：¹H-NMR中通常不直接提供强度，而是提供氢原子数
        integration = peak.get('nH')  # 氢原子数
        if integration is None:
            logging.debug(f"样本 {index}, H-NMR 峰缺少 'nH' 键。")
            integration = 0.0
        vec[5] = integration  # integration_value
        
        # 编码多重性信息（耦合模式）
        mult_type = peak.get('type', 'other')  # 多重性类型
        mult_idx = MULTIPLICITY_MAP.get(mult_type, 5)  # 获取对应索引，默认为5(other)
        vec[6 + mult_idx] = 1.0  # 在对应位置设置为1.0，形成one-hot编码
        
        all_peak_vectors.append(vec)
            
    # ================================================================================================
    # 第三步：处理¹³C-NMR峰，转换为Peak Vector
    # ================================================================================================
    """
    ¹³C-NMR峰提供碳原子的化学环境信息：
    - 化学位移(delta)：反映碳原子的化学环境，范围通常在0-230 ppm
    - 强度(intensity)：峰的相对强度，反映信号的强弱
    
    与¹H-NMR不同，¹³C-NMR通常不提供多重性信息，因为碳原子通常进行质子去耦。
    
    Peak Vector编码方式：
    - [0] = 1.0：标识为NMR峰
    - [2] = 归一化化学位移
    - [4] = 归一化强度
    - [6-11] = 保持为0（无多重性信息）
    """
    for peak in c_nmr_peaks:
        vec = [0.0] * PEAK_VECTOR_DIM  # 初始化12维零向量
        
        # 设置模态标识：这是一个NMR峰
        vec[0] = 1.0  # is_NMR = True
        
        # 提取并归一化化学位移
        # 注意：¹³C-NMR使用不同的键名 'delta (ppm)'
        position = peak.get('delta (ppm)')  # 化学位移，单位ppm
        if position is None:
            logging.debug(f"样本 {index}, C-NMR 峰缺少 'delta (ppm)' 键。")
            position = 0.0
            
        # 提取强度信息
        intensity = peak.get('intensity')  # 峰强度
        if intensity is None:
            logging.debug(f"样本 {index}, C-NMR 峰缺少 'intensity' 键。")
            intensity = 0.0
            
        # 归一化处理
        norm_pos = (position - min_ppm) / (max_ppm - min_ppm) if max_ppm > min_ppm else 0.0
        norm_int = (intensity - min_int_nmr) / (max_int_nmr - min_int_nmr) if max_int_nmr > min_int_nmr else 0.0
        
        # 填充Peak Vector
        vec[2] = norm_pos   # position_normalized
        vec[4] = norm_int   # intensity_normalized
        # 注意：¹³C-NMR峰不设置多重性信息（索引6-11保持为0）
        
        all_peak_vectors.append(vec)

    # ================================================================================================
    # 第四步：处理MS/MS峰，转换为Peak Vector
    # ================================================================================================
    """
    MS/MS（质谱）峰提供分子的质量信息：
    - m/z值：质荷比，反映离子的质量与电荷的比值
    - 强度(intensity)：峰的相对强度，反映该质量离子的丰度
    
    MS/MS数据对于分子识别和结构解析至关重要，特别是在代谢组学和蛋白质组学中。
    
    Peak Vector编码方式：
    - [1] = 1.0：标识为MS峰（与NMR峰区分）
    - [3] = 归一化m/z值
    - [4] = 归一化强度
    - [0,2,5,6-11] = 保持为0（MS峰不使用这些特征）
    """
    ms_peaks = row.get(ms_col_name)
    
    # [FIX] 从Parquet加载的数据可能是numpy.ndarray，需要转换为list
    if isinstance(ms_peaks, np.ndarray):
        ms_peaks = ms_peaks.tolist()

    if isinstance(ms_peaks, list) and len(ms_peaks) > 0:
        # 确保内部元素也是列表，而不是numpy数组
        if isinstance(ms_peaks[0], np.ndarray):
            ms_peaks = [p.tolist() for p in ms_peaks]

        # [鲁棒性] 过滤掉格式不正确的峰 (例如长度不为2)
        ms_peaks = [p for p in ms_peaks if isinstance(p, list) and len(p) == 2]

        if len(ms_peaks) > 0:
            ms_positions = [p[0] for p in ms_peaks]  # m/z值列表
            ms_intensities = [p[1] for p in ms_peaks]  # 强度列表
            
            # 在样本内归一化MS峰的m/z和强度值
            norm_ms_positions = normalize_values(ms_positions)
            norm_ms_intensities = normalize_values(ms_intensities)
            
            # 为每个MS峰创建Peak Vector
            for i in range(len(ms_peaks)):
                vec = [0.0] * PEAK_VECTOR_DIM  # 初始化12维零向量
                
                # 设置模态标识：这是一个MS峰
                vec[1] = 1.0  # is_MS = True
                
                # 填充归一化的m/z和强度值
                vec[3] = norm_ms_positions[i]   # ms_position_normalized
                vec[4] = norm_ms_intensities[i]  # intensity_normalized
                # 注意：MS峰不使用position、integration和multiplicity信息（索引0,2,5,6-11保持为0）
                
                all_peak_vectors.append(vec)
            
    if not all_peak_vectors:
        logging.debug(f"样本 {index} 因未提取到任何有效峰而被标记为空。")
        return None # 如果没有任何峰被提取，返回None

    # ================================================================================================
    # 第五步：返回最终的Peak Vector表示
    # ================================================================================================
    """
    函数返回值说明：
    - 返回一个PyTorch张量，形状为 [N, 12]，其中N是该样本的总峰数
    - 每一行代表一个峰的12维向量表示
    - 这种统一的表示方式使得不同模态的光谱数据可以在同一个模型中处理
    - 支持Set Transformer架构，因为峰的顺序不影响分子的表示
    
    数据流向：
    原始光谱数据 → Peak Vector → SpectraDataset → Set Transformer → 学习到的分子表示
    """
    return all_peak_vectors

def process_file(file_path, args, file_idx, total_files):
    """
    处理单个parquet文件并保存结果
    
    这个函数是数据预处理流水线的核心，负责：
    1. 读取原始parquet文件
    2. 并行处理每个样本，转换为Peak Vector表示
    3. 应用填充和截断策略，确保所有样本具有相同的峰数量
    4. 生成attention mask，用于区分真实峰和填充峰
    5. 保存为PyTorch格式，供训练使用
    
    参数:
        file_path: 原始parquet文件路径
        args: 命令行参数，包含处理配置
        file_idx: 当前文件索引（用于进度显示）
        total_files: 总文件数量
    """
    processed_data_dir = Path(args.processed_dir)
    output_filename = processed_data_dir / f"processed_{file_path.stem}.pt"

    # ================================================================================================
    # 优化策略1：断点续传/故障恢复
    # ================================================================================================
    # 如果输出文件已存在，跳过处理，避免重复计算
    if output_filename.exists():
        logging.info(f"⏭️  [{file_idx}/{total_files}] 文件 '{output_filename.name}' 已存在，跳过处理。")
        return

    logging.info(f"⚙️  [{file_idx}/{total_files}] 正在处理文件: {file_path.name}")
    df = pd.read_parquet(file_path)
    all_samples_data = []

    # ================================================================================================
    # 优化策略2：CPU多进程并行处理
    # ================================================================================================
    # 使用多进程池并行处理样本，显著提升处理速度
    # functools.partial 用于固定 process_single_sample 的 ms_col_name 参数
    process_func = functools.partial(process_single_sample, ms_col_name=args.ms_col)
    
    with Pool(processes=args.num_workers) as pool:
        # 使用 imap 以便可以和 tqdm 结合显示进度条
        results = list(tqdm(pool.imap(process_func, df.iterrows()), total=df.shape[0], desc=f"处理 {file_path.name}"))

    # ================================================================================================
    # 数据后处理：填充、截断和attention mask生成
    # ================================================================================================
    """
    为了支持批量训练，所有样本必须具有相同的峰数量。
    我们采用以下策略：
    1. 截断：如果峰数量超过max_peaks，保留前max_peaks个峰
    2. 填充：如果峰数量不足max_peaks，用零向量填充
    3. Attention Mask：标记哪些位置是真实峰(1)，哪些是填充(0)
    
    这种设计使得Set Transformer能够：
    - 忽略填充位置的影响
    - 专注于真实峰的表示学习
    - 支持变长序列的高效批量处理
    """
    total_processed = 0
    empty_skipped = 0
    for peak_vectors in results:
        # 如果 process_single_sample 返回 None，说明是空样本，直接跳过
        if peak_vectors is None:
            empty_skipped += 1
            continue

        num_peaks = len(peak_vectors)
        
        # 处理峰数量超过限制的情况：截断
        if num_peaks > args.max_peaks:
            peak_vectors = peak_vectors[:args.max_peaks]
            num_real_peaks = args.max_peaks
        else:
            # 处理峰数量不足的情况：填充零向量
            padding_needed = args.max_peaks - num_peaks
            peak_vectors.extend([[0.0] * PEAK_VECTOR_DIM for _ in range(padding_needed)])
            num_real_peaks = num_peaks
        
        # 生成attention mask：1表示真实峰，0表示填充
        attention_mask = [1] * num_real_peaks + [0] * (args.max_peaks - num_real_peaks)
        
        # 构建最终的样本数据结构
        all_samples_data.append({
            "input_tensor": torch.tensor(peak_vectors, dtype=torch.float32),      # [max_peaks, 12]
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool)      # [max_peaks]
        })
        total_processed += 1
    
    # ================================================================================================
    # 保存处理结果
    # ================================================================================================
    if total_processed > 0:
        torch.save(all_samples_data, output_filename)
        logging.info(f"✅ 处理完成 (处理 {total_processed} 个, 跳过 {empty_skipped} 个空样本), 数据已保存到: {output_filename}")
    else:
        logging.warning(f"⚠️ 在文件 {file_path.name} 中未找到任何有效的、非空的样本。未生成输出文件。")

def main(args):
    """
    主执行函数 - 多模态光谱数据预处理流水线
    
    整体数据流程：
    原始Parquet文件 → Peak Vector转换 → 填充/截断 → Attention Mask → PyTorch张量 → 保存
    
    这个预处理流水线的目标是：
    1. 将异构的多模态光谱数据（¹H-NMR, ¹³C-NMR, MS/MS）统一为Peak Vector表示
    2. 支持Set Transformer架构的无序集合处理
    3. 为自监督预训练任务准备数据
    4. 优化处理性能，支持大规模数据集
    
    输出数据格式：
    - 每个样本：{"input_tensor": [max_peaks, 12], "attention_mask": [max_peaks]}
    - 适用于SpectraDataset和PretrainSetTransformer
    """
    logging.info("--- 开始数据预处理 (优化版) ---")
    
    # ================================================================================================
    # 初始化和验证
    # ================================================================================================
    raw_data_dir = Path(args.raw_dir)
    processed_data_dir = Path(args.processed_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有原始数据文件
    raw_files = sorted(list(raw_data_dir.glob("*.parquet")))
    if not raw_files:
        logging.error(f"在 '{raw_data_dir}' 中未找到任何 .parquet 文件。")
        return

    logging.info(f"发现 {len(raw_files)} 个原始数据文件。将使用 {args.num_workers} 个CPU核心进行处理。")

    # ================================================================================================
    # 批量处理所有文件
    # ================================================================================================
    """
    逐个处理parquet文件，每个文件包含多个分子样本。
    处理策略：
    - 文件级并行：逐个处理文件（避免内存溢出）
    - 样本级并行：在单个文件内使用多进程处理样本
    - 断点续传：跳过已处理的文件
    """
    for i, file_path in enumerate(raw_files):
        process_file(file_path, args, i+1, len(raw_files))

    logging.info("--- 数据预处理完成 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态光谱数据预处理脚本 (优化版)")
    parser.add_argument('--raw_dir', type=str, default="data/raw/nips_2024_dataset/multimodal_spectroscopic_dataset", help="原始数据目录路径")
    parser.add_argument('--processed_dir', type=str, default="data/processed", help="处理后数据的保存目录路径")
    parser.add_argument('--max_peaks', type=int, default=512, help="每个样本填充到的最大峰数量")
    # [FIX] 修正默认的MS列名，以匹配真实数据文件
    parser.add_argument('--ms_col', type=str, default='msms_positive_20ev', help="选择要使用的MS/MS数据列")
    parser.add_argument('--num_workers', type=int, default=max(1, cpu_count() // 2), help="用于并行处理的CPU核心数")
    
    args = parser.parse_args()
    main(args)