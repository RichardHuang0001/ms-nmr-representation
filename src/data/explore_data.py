#!/usr/bin/env python3
"""
多模态光谱数据集内容展示工具
用于直观展示和分析仓库中的示例数据
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def print_section(title, char="="):
    """打印分节标题"""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")

def analyze_basic_info(df):
    """分析基本数据信息"""
    print_section("📊 数据集基本信息")
    print(f"数据形状: {df.shape[0]} 个样本, {df.shape[1]} 个特征")
    print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    print(f"\n📋 所有列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n📈 数据类型分布:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} 列")

def show_molecular_info(df):
    """展示分子信息"""
    print_section("🧪 分子结构信息", "-")
    
    print("前5个分子的基本信息:")
    for i in range(min(5, len(df))):
        print(f"\n样本 {i+1}:")
        print(f"  分子式: {df.iloc[i]['molecular_formula']}")
        print(f"  SMILES: {df.iloc[i]['smiles']}")
        
def show_spectrum_data(df, spectrum_type, sample_idx=0):
    """展示光谱数据"""
    print_section(f"📊 {spectrum_type} 光谱数据分析", "-")
    
    if spectrum_type not in df.columns:
        print(f"❌ 未找到 {spectrum_type} 数据")
        return
        
    spectrum = df.iloc[sample_idx][spectrum_type]
    
    if isinstance(spectrum, np.ndarray):
        print(f"光谱数据类型: numpy数组")
        print(f"数据形状: {spectrum.shape}")
        print(f"数值范围: [{spectrum.min():.4f}, {spectrum.max():.4f}]")
        print(f"数据类型: {spectrum.dtype}")
        print(f"前10个数据点: {spectrum[:10].tolist()}")
        
        # 统计信息
        print(f"\n统计信息:")
        print(f"  均值: {np.mean(spectrum):.4f}")
        print(f"  标准差: {np.std(spectrum):.4f}")
        print(f"  中位数: {np.median(spectrum):.4f}")
        print(f"  非零元素: {np.count_nonzero(spectrum)}/{len(spectrum)}")
        
    else:
        print(f"光谱数据类型: {type(spectrum)}")
        print(f"内容: {spectrum}")

def show_peaks_data(df, peaks_type, sample_idx=0):
    """展示峰数据"""
    print_section(f"🏔️ {peaks_type} 峰数据分析", "-")
    
    if peaks_type not in df.columns:
        print(f"❌ 未找到 {peaks_type} 数据")
        return
        
    peaks = df.iloc[sample_idx][peaks_type]
    print(f"峰数据类型: {type(peaks)}")
    
    if isinstance(peaks, list) and peaks:
        print(f"峰的数量: {len(peaks)}")
        print(f"前3个峰的详细信息:")
        
        for i, peak in enumerate(peaks[:3]):
            print(f"\n  峰 {i+1}:")
            if isinstance(peak, dict):
                for key, value in peak.items():
                    print(f"    {key}: {value}")
            else:
                print(f"    数据: {peak}")
    else:
        print(f"峰数据内容: {peaks}")

def show_msms_data(df, sample_idx=0):
    """展示质谱数据"""
    print_section("🎯 MS/MS 质谱数据分析", "-")
    
    msms_columns = [col for col in df.columns if 'msms' in col.lower()]
    print(f"发现 {len(msms_columns)} 个质谱相关列:")
    
    for col in msms_columns[:3]:  # 只展示前3个
        print(f"\n📊 {col}:")
        msms_data = df.iloc[sample_idx][col]
        
        if isinstance(msms_data, list) and msms_data:
            print(f"  数据点数量: {len(msms_data)}")
            print(f"  数据类型: {type(msms_data[0]) if msms_data else 'Empty'}")
            
            if len(msms_data) > 0:
                if isinstance(msms_data[0], list) and len(msms_data[0]) >= 2:
                    print(f"  m/z 范围: [{min(peak[0] for peak in msms_data):.1f}, {max(peak[0] for peak in msms_data):.1f}]")
                    print(f"  强度范围: [{min(peak[1] for peak in msms_data):.1f}, {max(peak[1] for peak in msms_data):.1f}]")
                    print(f"  前5个峰 (m/z, intensity): {msms_data[:5]}")
        else:
            print(f"  数据: {msms_data}")

def compare_samples(df, n_samples=3):
    """比较不同样本"""
    print_section(f"🔍 样本对比分析 (前{n_samples}个样本)", "-")
    
    for i in range(min(n_samples, len(df))):
        print(f"\n样本 {i+1}:")
        print(f"  分子式: {df.iloc[i]['molecular_formula']}")
        print(f"  SMILES: {df.iloc[i]['smiles'][:50]}...")
        
        # 统计各种光谱的大小
        if 'h_nmr_spectra' in df.columns:
            h_nmr = df.iloc[i]['h_nmr_spectra']
            print(f"  1H-NMR 数据点: {len(h_nmr) if hasattr(h_nmr, '__len__') else 'N/A'}")
            
        if 'c_nmr_spectra' in df.columns:
            c_nmr = df.iloc[i]['c_nmr_spectra']
            print(f"  13C-NMR 数据点: {len(c_nmr) if hasattr(c_nmr, '__len__') else 'N/A'}")
            
        if 'ir_spectra' in df.columns:
            ir = df.iloc[i]['ir_spectra']
            print(f"  IR 数据点: {len(ir) if hasattr(ir, '__len__') else 'N/A'}")

def save_sample_data(df, output_file="sample_analysis.json"):
    """保存样本分析结果"""
    sample_data = {
        "dataset_info": {
            "total_samples": len(df),
            "total_features": len(df.columns),
            "columns": list(df.columns)
        },
        "sample_molecules": []
    }
    
    for i in range(min(3, len(df))):
        sample = {
            "sample_id": i + 1,
            "molecular_formula": df.iloc[i]['molecular_formula'],
            "smiles": df.iloc[i]['smiles']
        }
        
        # 添加光谱数据的基本统计
        for col in ['h_nmr_spectra', 'c_nmr_spectra', 'ir_spectra']:
            if col in df.columns:
                spectrum = df.iloc[i][col]
                if isinstance(spectrum, np.ndarray):
                    sample[f"{col}_stats"] = {
                        "length": len(spectrum),
                        "mean": float(np.mean(spectrum)),
                        "std": float(np.std(spectrum)),
                        "min": float(spectrum.min()),
                        "max": float(spectrum.max())
                    }
        
        sample_data["sample_molecules"].append(sample)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 样本分析结果已保存到: {output_file}")

def main():
    """主函数"""
    print_section("🔬 多模态光谱数据集内容分析工具")
    
    # 加载数据 - 分析实际下载的数据
    data_path = Path("data/raw/nips_2024_dataset/multimodal_spectroscopic_dataset")
    print(f"🔍 正在加载数据: {data_path}")
    
    try:
        # 加载第一个实际数据文件
        parquet_files = list(data_path.glob("*.parquet"))
        if not parquet_files:
            print("❌ 未找到 parquet 文件")
            return
        
        print(f"📁 找到 {len(parquet_files)} 个 parquet 文件")
        print(f"📄 正在分析文件: {parquet_files[0].name}")
        df = pd.read_parquet(parquet_files[0])
        print(f"✅ 成功加载数据文件")
        
        # 基本信息分析
        analyze_basic_info(df)
        
        # 分子信息展示
        show_molecular_info(df)
        
        # 光谱数据展示
        spectrum_types = ['h_nmr_spectra', 'c_nmr_spectra', 'ir_spectra']
        for spectrum_type in spectrum_types:
            if spectrum_type in df.columns:
                show_spectrum_data(df, spectrum_type)
        
        # 峰数据展示
        peaks_types = ['h_nmr_peaks', 'c_nmr_peaks']
        for peaks_type in peaks_types:
            if peaks_type in df.columns:
                show_peaks_data(df, peaks_type)
        
        # 质谱数据展示
        show_msms_data(df)
        
        # 样本对比
        compare_samples(df)
        
        # 保存分析结果
        save_sample_data(df)
        
        print_section("🎉 数据分析完成！")
        print("💡 建议: 查看生成的 sample_analysis.json 文件获取详细的数据统计")
        
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()