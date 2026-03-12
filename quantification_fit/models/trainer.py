"""神经网络训练模块 - PyTorch"""

from typing import Optional, Tuple, Dict, Any
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class TradingClassifier(nn.Module):
    """交易分类神经网络 (MLP)"""

    def __init__(self, input_dim: int, hidden_dims: list = None, num_classes: int = 3):
        """初始化模型

        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_classes: 分类类别数（3: BUY/HOLD/SELL）
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Trainer:
    """模型训练器"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        """初始化训练器

        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度
            learning_rate: 学习率
            device: 设备 (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradingClassifier(input_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # 特征标准化器
        self.scaler = StandardScaler()

        logger.info(f"Model initialized on {self.device}")

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[DataLoader, DataLoader]:
        """准备训练和验证数据

        Args:
            X: 特征数据
            y: 标签数据
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            (train_loader, val_loader)
        """
        # 转换为numpy
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.int64)

        # 标准化
        X_scaled = self.scaler.fit_transform(X_np)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_np, test_size=test_size, random_state=random_state, stratify=y_np
        )

        # 转换为Tensor
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        # 创建DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        return train_loader, val_loader

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """训练模型

        Args:
            train_loader: 训练数据
            val_loader: 验证数据
            epochs: 轮数
            early_stopping_patience: 早停耐心值
            verbose: 是否打印日志

        Returns:
            训练历史
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += y_batch.size(0)
                train_correct += predicted.eq(y_batch).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += y_batch.size(0)
                    val_correct += predicted.eq(y_batch).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            # 记录历史
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

            # 早停
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # 恢复最佳模型
        if hasattr(self, "best_model_state"):
            self.model.load_state_dict(self.best_model_state)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测

        Args:
            X: 特征数据 (n_samples, n_features)

        Returns:
            预测类别
        """
        self.model.eval()

        X_scaled = self.scaler.transform(X.astype(np.float32))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率

        Args:
            X: 特征数据

        Returns:
            各类别概率 (n_samples, 3)
        """
        self.model.eval()

        X_scaled = self.scaler.transform(X.astype(np.float32))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()

    def save(self, path: str):
        """保存模型和标准化器"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 保存模型
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """加载模型和标准化器"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.mean_ = checkpoint["scaler_mean"]
        self.scaler.scale_ = checkpoint["scaler_scale"]

        logger.info(f"Model loaded from {path}")


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    hidden_dims: list = None,
    epochs: int = 100,
    test_size: float = 0.2,
) -> Tuple[Trainer, Dict[str, Any]]:
    """便捷训练函数

    Args:
        X: 特征数据
        y: 标签数据
        hidden_dims: 隐藏层维度
        epochs: 训练轮数
        test_size: 测试集比例

    Returns:
        (训练器, 训练历史)
    """
    input_dim = X.shape[1]
    trainer = Trainer(input_dim, hidden_dims)

    train_loader, val_loader = trainer.prepare_data(X, y, test_size)
    history = trainer.train(train_loader, val_loader, epochs)

    return trainer, history
