import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_data(processed_dir: str, max_files: int):
    """
    扫描预处理好的 .pt 文件，检查 attention_mask 的完整性，验证数据质量。
    """
    processed_path = Path(processed_dir)
    pt_files = sorted(list(processed_path.glob("*.pt")))

    if not pt_files:
        logging.error(f"在 '{processed_dir}' 目录中未找到任何 .pt 文件。")
        logging.error("请确认预处理脚本是否已成功运行并生成了输出。")
        return

    files_to_check = pt_files
    if max_files > 0:
        if len(pt_files) > max_files:
            files_to_check = pt_files[:max_files]
            logging.info(f"🔍 发现 {len(pt_files)} 个文件，将抽样检查前 {len(files_to_check)} 个...")
        else:
            logging.info(f"🔍 发现并检查全部 {len(pt_files)} 个文件...")
    else:
        logging.info(f"🔍 发现 {len(pt_files)} 个文件，将检查所有文件...")

    total_samples = 0
    empty_samples = 0
    non_empty_peak_counts = []

    for pt_file in tqdm(files_to_check, desc="扫描文件"):
        try:
            # 添加 weights_only=False 以兼容旧版torch并消除警告，但在实际生产中需谨慎
            samples = torch.load(pt_file, weights_only=False)
            for sample in samples:
                total_samples += 1
                attention_mask = sample.get('attention_mask')

                if attention_mask is None or not isinstance(attention_mask, torch.Tensor):
                    empty_samples += 1
                    continue

                num_real_peaks = attention_mask.sum().item()
                if num_real_peaks == 0:
                    empty_samples += 1
                else:
                    non_empty_peak_counts.append(num_real_peaks)

        except Exception as e:
            logging.error(f"\n🚨 加载或处理 {pt_file} 时出错: {e}", exc_info=True)
            continue
    
    print("\n" + "="*60)
    print("--- 📊 数据完整性检查报告 ---")
    print("="*60)

    if total_samples == 0:
        logging.error("未能成功分析任何样本，请检查.pt文件是否为空或格式错误。")
        return
        
    empty_percentage = (empty_samples / total_samples) * 100
    
    print(f"总共扫描文件数: {len(files_to_check)}")
    print(f"总共分析样本数: {total_samples}")
    print(f"不包含任何真实峰的样本数 (空样本): {empty_samples}")
    print(f"空样本比例: {empty_percentage:.2f}%")
    
    if non_empty_peak_counts:
        avg_peaks = sum(non_empty_peak_counts) / len(non_empty_peak_counts)
        min_peaks = min(non_empty_peak_counts)
        max_peaks = max(non_empty_peak_counts)
        print("\n--- 对于非空样本的统计 ---")
        print(f"成功提取的非空样本数量: {len(non_empty_peak_counts)}")
        print(f"每个非空样本的平均峰数: {avg_peaks:.2f}")
        print(f"非空样本的最小/最大峰数: {min_peaks} / {max_peaks}")
    else:
        print("\n在扫描的文件中未能找到任何包含真实峰的样本。")
        
    print("\n" + "="*60)
    print("--- 💡 结论 ---")
    print("="*60)
    if empty_percentage == 100.0:
        print("❗️ [失败] 检查发现所有样本均为空，与上次错误时的情况相同。")
        print("   'loss=0' 的问题几乎肯定会再次出现。")
        print("   👉 请再次审查 'src/data/preprocess.py' 的数据提取逻辑。")
    elif empty_percentage > 5.0:
        print("⚠️ [警告] 发现了较高比例的空样本 (>5%)。")
        print("   这可能是数据集本身的特性，但也值得关注。训练可能仍会成功，但数据有效率不高。")
    else:
        print("✅ [成功] 数据完整性检查通过！空样本比例很低。")
        print("   这表明 'preprocess.py' 的修复是成功的，'loss=0' 的问题根源已解决。")
        print("   您可以充满信心地进行下一步的训练。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查预处理后 .pt 文件的完整性和质量。")
    parser.add_argument(
        "--dir", 
        type=str, 
        default="data/processed",
        help="包含预处理好的 .pt 文件的目录。"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=10, # 稍微增加默认检查文件数以提高样本覆盖率
        help="检查的最大文件数量。设置为 0 则检查所有文件。"
    )
    args = parser.parse_args()
    
    check_data(args.dir, args.max_files)
