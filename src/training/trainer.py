# <-- [核心] 封装训练、验证循环和指标计算
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import logging
from pathlib import Path

class Trainer:
    """
    训练器类，封装了所有与模型训练和评估相关的逻辑。
    这使得主训练脚本可以保持简洁，只负责对象的初始化和启动训练流程。
    """
    def __init__(self, model, optimizer, train_loader, val_loader, config):
        """
        初始化训练器。
        
        :param model: 要训练的PyTorch模型。
        :param optimizer: 用于更新模型参数的优化器。
        :param train_loader: 训练数据的DataLoader。
        :param val_loader: 验证数据的DataLoader。
        :param config: 包含所有超参数和设置的配置对象 (Box)。
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        # 自动检测并选择可用的设备（优先使用GPU）
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logging.info(f"✅ 模型和数据将使用设备: {self.device}")

        # 初始化 Weights & Biases (W&B) 用于实验跟踪
        # `project_name` 和 `run_name` 用于在W&B仪表板上组织实验
        wandb.init(project=config.wandb.project_name, name=config.wandb.run_name, config=config)
        # `wandb.watch` 会自动记录模型的梯度和参数，便于调试和分析
        wandb.watch(self.model, log_freq=100)

    def _calculate_loss(self, predictions, labels):
        """
        计算用于预训练任务的混合损失函数。
        由于我们的峰向量包含连续数值和离散类别，因此需要组合不同的损失函数。

        :param predictions: 模型的输出，形状为 [B, L, D]。
        :param labels: 真实的标签，形状为 [B, L, D]，未被掩码的位置填充了-100。
        :return: 计算得到的标量损失值。
        """
        # 1. 找到被掩码的位置 (即标签中不为-100的地方)
        # 我们只关心模型在这些被掩码位置上的预测性能。
        # 这里我们假设，如果一个峰向量的第一个元素不是-100，那么整个向量都是有效标签。
        # 这会生成一个形状为 [B, L] 的布尔掩码。
        active_mask = labels[:, :, 0] != -100
        
        # 2. 使用上面生成的掩码，从预测和标签中只抽取出被掩码位置的向量。
        # 这样，损失将只在这些子集上计算。
        active_preds = predictions[active_mask]
        active_labels = labels[active_mask]

        # 如果当前批次中没有任何被掩码的样本，直接返回0损失。
        if active_preds.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 3. 从配置文件中获取特征向量不同部分的切片索引
        s_mod, e_mod = self.config.feature_slices.modality
        s_num, e_num = self.config.feature_slices.numerical
        s_mult, e_mult = self.config.feature_slices.multiplicity

        # 4. 分别计算不同部分的损失
        # a) 对于连续数值部分（如强度、m/z），使用均方误差损失 (MSE)
        loss_mse = F.mse_loss(active_preds[:, s_num:e_num], active_labels[:, s_num:e_num])

        # b) 对于离散类别部分（如模态、多重性），使用交叉熵损失 (Cross-Entropy)
        # 模型的输出是logits，而标签是one-hot编码。我们需要先用argmax从标签中获取类别索引。
        loss_ce_modality = F.cross_entropy(active_preds[:, s_mod:e_mod], torch.argmax(active_labels[:, s_mod:e_mod], dim=-1))
        loss_ce_mult = F.cross_entropy(active_preds[:, s_mult:e_mult], torch.argmax(active_labels[:, s_mult:e_mult], dim=-1))

        # 5. 将各部分损失加权求和（当前权重均为1）
        # 未来可以根据不同任务的重要性调整这些权重。
        total_loss = loss_mse + loss_ce_modality + loss_ce_mult
        return total_loss

    def _run_epoch(self, dataloader, is_training=True):
        """
        运行一个完整的epoch，可以是训练或验证。

        :param dataloader: 用于该epoch的数据加载器。
        :param is_training: 布尔值，如果为True，则执行训练步骤（反向传播和优化）。
        :return: 该epoch的平均损失。
        """
        # 根据是训练还是评估，设置模型的模式
        self.model.train(is_training)
        total_loss = 0.0
        
        # 使用tqdm创建进度条，方便监控
        desc = "Train" if is_training else "Eval"
        progress_bar = tqdm(dataloader, desc=desc)
        
        for batch in progress_bar:
            # 将批次中的所有张量移动到指定设备
            masked_inputs, masks, labels = [b.to(self.device) for b in batch]

            # 在训练模式下，每个批次前清空之前的梯度
            if is_training:
                self.optimizer.zero_grad()
            
            # 使用torch.set_grad_enabled上下文管理器来控制是否计算梯度
            # 在验证时关闭梯度计算可以节省内存并加速计算
            with torch.set_grad_enabled(is_training):
                predictions = self.model(masked_inputs, masks)
                loss = self._calculate_loss(predictions, labels)

            # 如果是训练模式，执行反向传播和优化器步骤
            if is_training:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            # 在进度条上实时显示当前批次的损失
            progress_bar.set_postfix(loss=loss.item())

        # 返回整个epoch的平均损失
        return total_loss / len(dataloader)

    def _save_checkpoint(self, epoch, val_loss, is_best):
        """
        保存模型检查点。只在验证损失改善时保存“最佳模型”。

        :param epoch: 当前的epoch号。
        :param val_loss: 当前的验证损失。
        :param is_best: 布尔值，指示当前模型是否是迄今为止最好的。
        """
        # 如果当前模型不是最好的，则不执行任何操作
        if not is_best:
            return

        # 确保检查点目录存在
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "best_model.pt"
        
        # 保存模型状态字典、优化器状态、epoch号和验证损失
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
        logging.info(f"🚀 新的最佳模型已保存 (Epoch {epoch+1}, Val Loss: {val_loss:.4f}) 至: {checkpoint_path}")

    def train(self):
        """
        执行完整的模型训练流程，包括多个epoch的训练和验证。
        """
        # 初始化最佳验证损失为一个极大值
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.epochs):
            # 运行一个训练epoch
            train_loss = self._run_epoch(self.train_loader, is_training=True)
            # 运行一个验证epoch
            val_loss = self._run_epoch(self.val_loader, is_training=False)
            
            # 记录日志
            logging.info(f"Epoch {epoch+1}/{self.config.training.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            # 将指标记录到W&B
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            # 检查当前验证损失是否是历史最佳
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # 根据is_best标志决定是否保存模型
            self._save_checkpoint(epoch, val_loss, is_best)