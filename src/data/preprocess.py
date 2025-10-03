## preprocess.py

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

# --- 我们最终确定的Peak Vector设计参数 ---
PEAK_VECTOR_DIM = 12
MULTIPLICITY_MAP = {
    's': 0, 'd': 1, 't': 2, 'q': 3, 'm': 4
    # 第5个索引将留给 'other'
}

def normalize_values(values):
    """对一个列表进行最小-最大归一化"""
    if not values or len(values) < 2 or max(values) == min(values):
        return [0.0] * len(values)
    min_val, max_val = min(values), max(values)
    range_val = max_val - min_val
    return [(v - min_val) / range_val for v in values]

def process_single_sample(row_tuple, ms_col_name):
    """
    处理单个样本（DataFrame的一行元组），将其所有MS和NMR峰转换为Peak Vector列表。
    为适配多进程map，输入从DataFrame行改为元组。
    """
    index, row = row_tuple
    all_peak_vectors = []
    
    # --- 提取和处理NMR峰 ---
    h_nmr_peaks = row.get('h_nmr_peaks', [])
    c_nmr_peaks = row.get('c_nmr_peaks', [])
    
    # [鲁棒性修复] 确保数据是列表格式
    h_nmr_peaks = h_nmr_peaks if isinstance(h_nmr_peaks, list) else []
    c_nmr_peaks = c_nmr_peaks if isinstance(c_nmr_peaks, list) else []

    all_nmr_positions = [p.get('position', 0.0) for p in h_nmr_peaks] + [p.get('position', 0.0) for p in c_nmr_peaks]
    all_nmr_intensities = [p.get('intensity', 0.0) for p in h_nmr_peaks] + [p.get('intensity', 0.0) for p in c_nmr_peaks]
    
    # 归一化参数计算
    min_ppm, max_ppm = (min(all_nmr_positions), max(all_nmr_positions)) if all_nmr_positions else (0, 0)
    min_int_nmr, max_int_nmr = (min(all_nmr_intensities), max(all_nmr_intensities)) if all_nmr_intensities else (0, 0)

    # 转换¹H-NMR峰
    for peak in h_nmr_peaks:
        vec = [0.0] * PEAK_VECTOR_DIM
        vec[0] = 1.0  # is_NMR
        norm_pos = (peak.get('position', 0.0) - min_ppm) / (max_ppm - min_ppm) if max_ppm > min_ppm else 0.0
        norm_int = (peak.get('intensity', 0.0) - min_int_nmr) / (max_int_nmr - min_int_nmr) if max_int_nmr > min_int_nmr else 0.0
        vec[2] = norm_pos
        vec[4] = norm_int
        vec[5] = peak.get('integration', 0.0)
        
        mult_type = peak.get('type', 'other')
        mult_idx = MULTIPLICITY_MAP.get(mult_type, 5)
        vec[6 + mult_idx] = 1.0
        
        all_peak_vectors.append(vec)
            
    # 转换¹³C-NMR峰
    for peak in c_nmr_peaks:
        vec = [0.0] * PEAK_VECTOR_DIM
        vec[0] = 1.0  # is_NMR
        norm_pos = (peak.get('position', 0.0) - min_ppm) / (max_ppm - min_ppm) if max_ppm > min_ppm else 0.0
        norm_int = (peak.get('intensity', 0.0) - min_int_nmr) / (max_int_nmr - min_int_nmr) if max_int_nmr > min_int_nmr else 0.0
        vec[2] = norm_pos
        vec[4] = norm_int
        all_peak_vectors.append(vec)

    # --- 提取和处理MS峰 ---
    ms_peaks = row.get(ms_col_name)
    if isinstance(ms_peaks, list) and len(ms_peaks) > 0:
        ms_positions = [p[0] for p in ms_peaks]
        ms_intensities = [p[1] for p in ms_peaks]
        
        norm_ms_positions = normalize_values(ms_positions)
        norm_ms_intensities = normalize_values(ms_intensities)
        
        for i in range(len(ms_peaks)):
            vec = [0.0] * PEAK_VECTOR_DIM
            vec[1] = 1.0 # is_MS
            vec[3] = norm_ms_positions[i]
            vec[4] = norm_ms_intensities[i]
            all_peak_vectors.append(vec)
            
    return all_peak_vectors

def process_file(file_path, args):
    """处理单个parquet文件并保存结果"""
    processed_data_dir = Path(args.processed_dir)
    output_filename = processed_data_dir / f"processed_{file_path.stem}.pt"

    # --- [优化2] 断点续传/故障恢复 ---
    if output_filename.exists():
        logging.info(f"⏭️  文件 '{output_filename.name}' 已存在，跳过处理。")
        return

    logging.info(f"⚙️  正在处理文件: {file_path.name}")
    df = pd.read_parquet(file_path)
    all_samples_data = []

    # --- [优化1] CPU多进程并行处理 ---
    # functools.partial 用于固定 process_single_sample 的 ms_col_name 参数
    # df.iterrows() 性能较差, 改为 df.itertuples()
    process_func = functools.partial(process_single_sample, ms_col_name=args.ms_col)
    
    with Pool(processes=args.num_workers) as pool:
        # 使用 imap 以便可以和 tqdm 结合显示进度条
        results = list(tqdm(pool.imap(process_func, df.iterrows()), total=df.shape[0], desc=f"处理 {file_path.name}"))

    for peak_vectors in results:
        num_peaks = len(peak_vectors)
        if num_peaks > args.max_peaks:
            peak_vectors = peak_vectors[:args.max_peaks]
            num_real_peaks = args.max_peaks
        else:
            padding_needed = args.max_peaks - num_peaks
            peak_vectors.extend([[0.0] * PEAK_VECTOR_DIM for _ in range(padding_needed)])
            num_real_peaks = num_peaks
        
        attention_mask = [1] * num_real_peaks + [0] * (args.max_peaks - num_real_peaks)
        
        all_samples_data.append({
            "input_tensor": torch.tensor(peak_vectors, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool)
        })

    torch.save(all_samples_data, output_filename)
    logging.info(f"✅ 处理完成，数据已保存到: {output_filename}")

def main(args):
    """主执行函数"""
    logging.info("--- 开始数据预处理 (优化版) ---")
    
    raw_data_dir = Path(args.raw_dir)
    processed_data_dir = Path(args.processed_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    raw_files = sorted(list(raw_data_dir.glob("*.parquet")))
    if not raw_files:
        logging.error(f"在 '{raw_data_dir}' 中未找到任何 .parquet 文件。")
        return

    logging.info(f"发现 {len(raw_files)} 个原始数据文件。将使用 {args.num_workers} 个CPU核心进行处理。")

    for file_path in raw_files:
        process_file(file_path, args)

    logging.info("--- 所有数据预处理完成 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态光谱数据预处理脚本 (优化版)")
    parser.add_argument('--raw_dir', type=str, default="data/raw/nips_2024_dataset/multimodal_spectroscopic_dataset", help="原始数据目录路径")
    parser.add_argument('--processed_dir', type=str, default="data/processed", help="处理后数据的保存目录路径")
    parser.add_argument('--max_peaks', type=int, default=512, help="每个样本填充到的最大峰数量")
    parser.add_argument('--ms_col', type=str, default='msms_cfmid_positive_20ev', help="选择要使用的MS/MS数据列")
    parser.add_argument('--num_workers', type=int, default=max(1, cpu_count() // 2), help="用于并行处理的CPU核心数")
    
    args = parser.parse_args()
    main(args)