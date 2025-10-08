#!/usr/bin/env python3
"""
数据嗅探工具 (修改版)
用于精确探查特定 Parquet 文件的数据结构和内容，以诊断预处理问题。
"""

import pandas as pd
import argparse
from pathlib import Path

def print_section(title, char="="):
    """打印分节标题"""
    print(f"\n{char * 80}")
    print(f" {title}")
    print(f"{char * 80}")

def sniff_parquet_file(file_path: Path, ms_col_name: str):
    """
    嗅探单个 Parquet 文件，打印其 schema 和第一个样本的关键列内容。
    """
    print_section(f"🕵️  开始嗅探文件: {file_path.name}")

    if not file_path.exists():
        print(f"❌ 错误: 文件不存在 -> {file_path}")
        return

    try:
        df = pd.read_parquet(file_path)
        print(f"✅ 文件加载成功。")
        print(f"   - 数据形状: {df.shape[0]} 个样本, {df.shape[1]} 个特征")

        # 1. 打印所有列名
        print("\n--- 📋 可用列名 (Schema) ---")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")

        # 2. 详细检查第一个样本 (sample_idx=0) 的关键列
        print("\n--- 🔬 第一个样本 (索引0) 的关键列内容检查 ---")
        if df.empty:
            print("   - 文件为空，没有样本可供检查。")
            return
            
        sample = df.iloc[0]

        # 关键列列表
        key_columns = ['h_nmr_peaks', 'c_nmr_peaks', ms_col_name]
        
        for col in key_columns:
            print(f"\n   --- 列: '{col}' ---")
            if col in df.columns:
                data = sample[col]
                print(f"   - (L1) 数据类型: {type(data)}")

                # 对MS列进行深度嗅探
                if col == ms_col_name and hasattr(data, '__len__') and len(data) > 0:
                    first_element = data[0]
                    print(f"   - (L2) 第一个元素的数据类型: {type(first_element)}")
                    
                    if hasattr(first_element, '__len__'):
                        print(f"   - (L3) 第一个元素的长度: {len(first_element)}")
                        print(f"   - (L3) 第一个元素的内容: {first_element}")

                print(f"   - 数据内容:")
                # 为了更清晰的显示，我们对内容进行格式化
                if hasattr(data, '__len__') and len(data) > 5:
                    print(f"     (列表长度为 {len(data)}, 仅显示前5个元素)")
                    print(f"     {data[:5]}")
                else:
                    print(f"     {data}")
            else:
                print(f"   - ❗️ 警告: 在文件中未找到该列。")

    except Exception as e:
        print(f"\n❌ 处理文件时发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Parquet 文件数据嗅探工具。")
    parser.add_argument(
        'filenames', 
        nargs='+', 
        help="要嗅探的一个或多个 Parquet 文件名 (例如: aligned_chunk_0.parquet)"
    )
    parser.add_argument(
        '--base_dir', 
        type=str, 
        default="data/raw/nips_2024_dataset/multimodal_spectroscopic_dataset",
        help="存放 Parquet 文件的基础目录。"
    )
    parser.add_argument(
        '--ms_col', 
        type=str, 
        default='msms_cfmid_positive_20ev', 
        help="要检查的质谱列名。"
    )
    args = parser.parse_args()

    for filename in args.filenames:
        file_path = Path(args.base_dir) / filename
        sniff_parquet_file(file_path, args.ms_col)

if __name__ == "__main__":
    main()