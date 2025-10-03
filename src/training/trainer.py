# <-- [核心] 封装训练、验证循环和指标计算
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import logging
from pathlib import Path

class Trainer:
    """封装训练和评估逻辑的训练器类。"""
    def __init__(self, model, optimizer, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 初始化W&B
        wandb.init(project=config.wandb.project_name, name=config.wandb.run_name, config=config)
        wandb.watch(self.model, log_freq=100)

    def _calculate_loss(self, predictions, labels):
        """计算混合损失函数。"""
        # 找到被mask的位置 (labels中不为-100的地方)
        active_loss = labels != -100
        active_preds = predictions[active_loss]
        active_labels = labels[active_loss]

        if active_preds.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 从配置中获取特征切片信息
        s_mod, e_mod = self.config.feature_slices.modality
        s_num, e_num = self.config.feature_slices.numerical
        s_mult, e_mult = self.config.feature_slices.multiplicity

        # 数值部分损失 (MSE)
        loss_mse = F.mse_loss(active_preds[:, s_num:e_num], active_labels[:, s_num:e_num])

        # 类别部分损失 (CrossEntropy)
        loss_ce_modality = F.cross_entropy(active_preds[:, s_mod:e_mod], torch.argmax(active_labels[:, s_mod:e_mod], dim=-1))
        loss_ce_mult = F.cross_entropy(active_preds[:, s_mult:e_mult], torch.argmax(active_labels[:, s_mult:e_mult], dim=-1))

        # 加权求和 (可以根据需要调整权重)
        total_loss = loss_mse + loss_ce_modality + loss_ce_mult
        return total_loss

    def _run_epoch(self, dataloader, is_training=True):
        """运行一个epoch的训练或验证。"""
        self.model.train(is_training)
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"{'Train' if is_training else 'Eval'}")
        for batch in progress_bar:
            masked_inputs, masks, labels = [b.to(self.device) for b in batch]

            if is_training:
                self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(is_training):
                predictions = self.model(masked_inputs, masks)
                loss = self._calculate_loss(predictions, labels)

            if is_training:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(dataloader)

    def _save_checkpoint(self, epoch, val_loss, is_best):
        """保存模型检查点。"""
        if not is_best:
            return

        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "best_model.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
        logging.info(f"🚀 新的最佳模型已保存 (Epoch {epoch+1}, Val Loss: {val_loss:.4f}) 至: {checkpoint_path}")

    def train(self):
        """完整的训练流程。"""
        best_val_loss = float('inf')
        for epoch in range(self.config.training.epochs):
            train_loss = self._run_epoch(self.train_loader, is_training=True)
            val_loss = self._run_epoch(self.val_loader, is_training=False)
            
            logging.info(f"Epoch {epoch+1}/{self.config.training.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            # 仅在验证损失改善时保存最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            self._save_checkpoint(epoch, val_loss, is_best)