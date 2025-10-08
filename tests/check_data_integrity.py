import torch
from pathlib import Path
from tqdm import tqdm
import argparse

def check_data(processed_dir: str, max_files: int):
    """
    扫描预处理好的 .pt 文件，检查 attention_mask 的完整性。
    """
    processed_path = Path(processed_dir)
    pt_files = sorted(list(processed_path.glob("*.pt")))

    if not pt_files:
        print(f"❌ 错误: 在 '{processed_dir}' 目录中未找到 .pt 文件。")
        print("请先运行预处理脚本。")
        return

    if max_files > 0:
        files_to_check = pt_files[:min(max_files, len(pt_files))]
        print(f"🔍 正在检查前 {len(files_to_check)} 个文件 (共 {len(pt_files)} 个)...")
    else:
        files_to_check = pt_files
        print(f"🔍 正在检查全部 {len(pt_files)} 个文件...")

    total_samples = 0
    empty_samples = 0
    non_empty_peak_counts = []

    for pt_file in tqdm(files_to_check, desc="扫描文件"):
        try:
            samples = torch.load(pt_file)
            for sample in samples:
                total_samples += 1
                attention_mask = sample.get('attention_mask')

                if attention_mask is None:
                    empty_samples += 1
                    continue

                num_real_peaks = attention_mask.sum().item()
                if num_real_peaks == 0:
                    empty_samples += 1
                else:
                    non_empty_peak_counts.append(num_real_peaks)

        except Exception as e:
            print(f"\n🚨 加载或处理 {pt_file} 时出错: {e}")
            continue
    
    print("\n--- 📊 数据完整性检查报告 ---")
    if total_samples == 0:
        print("未能处理任何样本，请检查文件内容。")
        return
        
    empty_percentage = (empty_samples / total_samples) * 100
    
    print(f"总共扫描文件数: {len(files_to_check)}")
    print(f"总共分析样本数: {total_samples}")
    print(f"不包含任何真实峰的样本数: {empty_samples}")
    print(f"空样本比例: {empty_percentage:.2f}%")
    
    if non_empty_peak_counts:
        avg_peaks = sum(non_empty_peak_counts) / len(non_empty_peak_counts)
        min_peaks = min(non_empty_peak_counts)
        max_peaks = max(non_empty_peak_counts)
        print("\n--- 对于非空样本的统计 ---")
        print(f"发现的非空样本数量: {len(non_empty_peak_counts)}")
        print(f"每个非空样本的平均峰数: {avg_peaks:.2f}")
        print(f"非空样本的最小/最大峰数: {min_peaks} / {max_peaks}")
    else:
        print("\n在扫描的文件中未能找到任何非空样本。")
        
    print("\n--- 💡 结论 ---")
    if empty_percentage > 95:
        print("❗️ 检测到极高比例的空样本。")
        print("这很可能就是导致 'loss=0' 问题的根本原因。")
        print("👉 建议: 请重点审查 'src/data/preprocess.py' 中生成 'attention_mask' 的逻辑。")
    elif empty_percentage > 10:
        print("⚠️ 发现了较多空样本。这可能符合数据集的特性，但也可能是一个预处理问题。")
    else:
        print("✅ 空样本比例似乎较低。'loss=0' 的问题可能由其他原因引起。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查预处理后文件的完整性。")
    parser.add_argument(
        "--dir", 
        type=str, 
        default="data/processed",
        help="包含预处理好的 .pt 文件的目录。"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=5,
        help="检查的最大文件数量。设置为 0 则检查所有文件。"
    )
    args = parser.parse_args()
    
    check_data(args.dir, args.max_files)

