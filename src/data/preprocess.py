# <-- [核心] 实现数据预处理流程的主脚本
# src/data/preprocess.py

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 我们最终确定的Peak Vector设计参数 ---
PEAK_VECTOR_DIM = 12
MULTIPLICITY_MAP = {
    's': 0, 'd': 1, 't': 2, 'q': 3, 'm': 4
    # 第5个索引将留给 'other'
}

def normalize_values(values):
    """对一个列表进行最小-最大归一化"""
    if not values or max(values) == min(values):
        return [0.0] * len(values)
    min_val, max_val = min(values), max(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

def process_single_sample(row, ms_col_name):
    """
    处理单个样本（DataFrame的一行），将其所有MS和NMR峰转换为Peak Vector列表。
    """
    all_peak_vectors = []
    
    # --- 提取和处理NMR峰 ---
    h_nmr_peaks = row.get('h_nmr_peaks', [])
    c_nmr_peaks = row.get('c_nmr_peaks', [])
    
    # --- [修复] 增加鲁棒性，并使用正确的 'position' 键 ---
    all_nmr_positions = []
    
    # 安全地提取 h_nmr 峰位置
    if isinstance(h_nmr_peaks, list):
        for p in h_nmr_peaks:
            try:
                all_nmr_positions.append(p['position'])
            except KeyError:
                logging.warning(f"样本中一个 ¹H-NMR 峰缺少 'position' 键。该峰将被忽略。")
    
    # 安全地提取 c_nmr 峰位置
    if isinstance(c_nmr_peaks, list):
        for p in c_nmr_peaks:
            try:
                all_nmr_positions.append(p['position'])
            except KeyError:
                logging.warning(f"样本中一个 ¹³C-NMR 峰缺少 'position' 键。该峰将被忽略。")

    all_nmr_intensities = [p.get('intensity', 0) for p in h_nmr_peaks if isinstance(p, dict)] + \
                          [p.get('intensity', 0) for p in c_nmr_peaks if isinstance(p, dict)]
    
    if not all_nmr_positions:
        norm_nmr_positions = []
    else:
        min_ppm, max_ppm = min(all_nmr_positions), max(all_nmr_positions)
        norm_nmr_positions = [(p - min_ppm) / (max_ppm - min_ppm) if max_ppm > min_ppm else 0.0 for p in all_nmr_positions]

    if not all_nmr_intensities:
        norm_nmr_intensities = []
    else:
        min_int, max_int = min(all_nmr_intensities), max(all_nmr_intensities)
        norm_nmr_intensities = [(i - min_int) / (max_int - min_int) if max_int > min_int else 0.0 for i in all_nmr_intensities]
        
    nmr_pos_iter = iter(norm_nmr_positions)
    nmr_int_iter = iter(norm_nmr_intensities)

    # 转换¹H-NMR峰
    if isinstance(h_nmr_peaks, list):
        for peak in h_nmr_peaks:
            # 确保峰是字典且包含'position'，否则跳过
            if not isinstance(peak, dict) or 'position' not in peak:
                continue
            vec = [0.0] * PEAK_VECTOR_DIM
            vec[0] = 1.0  # is_NMR
            vec[2] = next(nmr_pos_iter, 0.0) # norm_position_NMR
            vec[4] = next(nmr_int_iter, 0.0) # norm_intensity
            vec[5] = peak.get('integration', 0.0)
            
            mult_type = peak.get('type', 'other')
            mult_idx = MULTIPLICITY_MAP.get(mult_type, 5) # 5是'other'的索引
            vec[6 + mult_idx] = 1.0
            
            all_peak_vectors.append(vec)
            
    # 转换¹³C-NMR峰
    if isinstance(c_nmr_peaks, list):
        for peak in c_nmr_peaks:
            # 确保峰是字典且包含'position'，否则跳过
            if not isinstance(peak, dict) or 'position' not in peak:
                continue
            vec = [0.0] * PEAK_VECTOR_DIM
            vec[0] = 1.0  # is_NMR
            vec[2] = next(nmr_pos_iter, 0.0) # norm_position_NMR
            vec[4] = next(nmr_int_iter, 0.0) # norm_intensity
            # 其他NMR专属特征为0
            all_peak_vectors.append(vec)

    # --- 提取和处理MS峰 ---
    ms_peaks = row[ms_col_name]
    if isinstance(ms_peaks, list) and len(ms_peaks) > 0:
        ms_positions = [p[0] for p in ms_peaks]
        ms_intensities = [p[1] for p in ms_peaks]
        
        norm_ms_positions = normalize_values(ms_positions)
        norm_ms_intensities = normalize_values(ms_intensities)
        
        for i in range(len(ms_peaks)):
            vec = [0.0] * PEAK_VECTOR_DIM
            vec[1] = 1.0 # is_MS
            vec[3] = norm_ms_positions[i] # norm_position_MS
            vec[4] = norm_ms_intensities[i] # norm_intensity
            # 其他NMR专属特征为0
            all_peak_vectors.append(vec)
            
    return all_peak_vectors


def main(args):
    """主执行函数"""
    logging.info("--- 开始数据预处理 ---")
    
    raw_data_dir = Path(args.raw_dir)
    processed_data_dir = Path(args.processed_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有原始数据文件
    raw_files = list(raw_data_dir.glob("*.parquet"))
    if not raw_files:
        logging.error(f"在 '{raw_data_dir}' 中未找到任何 .parquet 文件。")
        return

    logging.info(f"发现 {len(raw_files)} 个原始数据文件。")

    # 遍历每个文件进行处理
    for file_path in raw_files:
        logging.info(f"正在处理文件: {file_path.name}")
        df = pd.read_parquet(file_path)
        
        all_samples_data = []
        
        # tqdm用于显示进度条
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="处理样本"):
            # 选择一种MS谱进行处理，这里我们以CFMID 20eV为例
            peak_vectors = process_single_sample(row, ms_col_name='msms_cfmid_positive_20ev')
            
            # 截断或填充
            num_peaks = len(peak_vectors)
            if num_peaks > args.max_peaks:
                peak_vectors = peak_vectors[:args.max_peaks]
                num_real_peaks = args.max_peaks
            else:
                padding_needed = args.max_peaks - num_peaks
                peak_vectors.extend([[0.0] * PEAK_VECTOR_DIM for _ in range(padding_needed)])
                num_real_peaks = num_peaks
            
            # 创建注意力掩码
            attention_mask = [1] * num_real_peaks + [0] * (args.max_peaks - num_real_peaks)
            
            all_samples_data.append({
                "input_tensor": torch.tensor(peak_vectors, dtype=torch.float32),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.bool)
            })

        # 将处理好的数据保存为PyTorch的.pt文件
        output_filename = processed_data_dir / f"processed_{file_path.stem}.pt"
        torch.save(all_samples_data, output_filename)
        logging.info(f"✅ 处理完成，数据已保存到: {output_filename}")

    logging.info("--- 所有数据预处理完成 ---")


if __name__ == "__main__":
    # 使用argparse来接收命令行参数，增加脚本的灵活性
    parser = argparse.ArgumentParser(description="多模态光谱数据预处理脚本")
    parser.add_argument('--raw_dir', type=str, default="data/raw/nips_2024_dataset", help="原始数据目录路径")
    parser.add_argument('--processed_dir', type=str, default="data/processed", help="处理后数据的保存目录路径")
    parser.add_argument('--max_peaks', type=int, default=512, help="每个样本填充到的最大峰数量")
    
    args = parser.parse_args()
    main(args)