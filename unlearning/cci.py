import torch
import torch.nn as nn
import torch.optim as optim
from torch_sparse import SparseTensor
import numpy as np
from utils.utils import get_topK_recommendations


class CausalContributionIdentifier:
    """
    因果贡献识别器 (Causal Contribution Identifier, CCI)
    该模块用于识别推荐结果的因果支撑集及其贡献度
    """

    def __init__(self, model, num_users, num_items, user_idx, adj, lambda_reg=1.0):
        """
        初始化CCI模块

        参数:
            model: 预训练的推荐模型
            num_users: 用户数量
            num_items: 物品数量
            user_idx: 目标用户索引
            adj: 邻接矩阵
            lambda_reg: 稀疏性正则项的权重
        """
        self.model = model
        self.num_users = num_users
        self.num_items = num_items
        self.user_idx = user_idx
        self.adj = adj
        self.lambda_reg = lambda_reg

        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 获取用户的历史交互数
        if isinstance(self.adj, SparseTensor):
            row, col, value = self.adj.coo()
        else:
            # 处理稠密矩阵或其它格式
            if self.adj.is_sparse:
                indices = self.adj._indices()
                row, col = indices[0], indices[1]
            else:
                # 稠密矩阵
                row, col = torch.where(self.adj > 0)
        
        self.user_hist_count = (row == self.user_idx).sum().item()
        print(f'用户 {self.user_idx} 有 {self.user_hist_count} 条历史记录')

        # 初始化扰动向量P，维度等于用户历史交互数
        self.P_vec = nn.Parameter(torch.ones(self.user_hist_count, requires_grad=True))

    def identify_causal_contributions(self, target_item, optimizer_type="Adam", lr=0.01, 
                                      num_epochs=100, lambda_reg=None):
        """
        识别目标推荐项的因果贡献

        参数:
            target_item: 目标推荐物品索引
            optimizer_type: 优化器类型
            lr: 学习率
            num_epochs: 训练轮数
            lambda_reg: 稀疏性正则项权重

        返回:
            causal_support_set: 因果支撑集
            contributions: 贡献度函数
        """
        if lambda_reg is not None:
            self.lambda_reg = lambda_reg

        # 初始化优化器
        if optimizer_type == "SGD":
            optimizer = optim.SGD([self.P_vec], lr=lr)
        elif optimizer_type == "Adam":
            optimizer = optim.Adam([self.P_vec], lr=lr)
        else:
            optimizer = optim.Adadelta([self.P_vec], lr=lr)

        # 训练过程
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # 前向传播获取原始评分
            users_emb_original, _, items_emb_original, _ = self.model.forward(self.adj)
            rating_original = torch.matmul(users_emb_original, items_emb_original.T)
            original_score = rating_original[self.user_idx, target_item]

            # 应用扰动向量P
            perturbed_adj = self._apply_perturbation()
            users_emb_perturbed, _, items_emb_perturbed, _ = self.model.forward(perturbed_adj)
            rating_perturbed = torch.matmul(users_emb_perturbed, items_emb_perturbed.T)
            perturbed_score = rating_perturbed[self.user_idx, target_item]

            # 计算翻转损失（使目标物品的评分下降）
            flip_loss = perturbed_score

            # 稀疏性正则项
            sparsity_loss = self.lambda_reg * torch.norm(1 - self.P_vec, p=1)

            # 总损失
            total_loss = flip_loss + sparsity_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 约束P_vec在[0, 1]范围内
            with torch.no_grad():
                self.P_vec.clamp_(0.0, 1.0)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, "
                      f"Flip Loss: {flip_loss.item():.4f}, Sparsity Loss: {sparsity_loss.item():.4f}")

        # 识别因果支撑集和贡献度
        causal_support_set, contributions = self._extract_causal_support_and_contributions()

        return causal_support_set, contributions

    def _apply_perturbation(self):
        """
        应用扰动向量P到邻接矩阵

        返回:
            perturbed_adj: 扰动后的邻接矩阵
        """
        if isinstance(self.adj, SparseTensor):
            row, col, value = self.adj.coo()
            # 创建扰动后的权重
            perturbed_values = value.clone()
        else:
            # 处理稠密矩阵或其它格式
            if self.adj.is_sparse:
                indices = self.adj._indices()
                values = self.adj._values()
                row, col = indices[0], indices[1]
                perturbed_values = values.clone()
            else:
                # 稠密矩阵
                row, col = torch.where(self.adj > 0)
                perturbed_values = torch.ones(row.shape[0])

        # 获取与目标用户相关的边索引
        user_edge_mask = (row == self.user_idx)
        user_edge_indices = torch.nonzero(user_edge_mask).squeeze()
        
        # 应用扰动向量P
        if len(user_edge_indices.shape) == 0:  # 只有一条边
            perturbed_values[user_edge_indices] = perturbed_values[user_edge_indices] * self.P_vec[0]
        else:
            perturbed_values[user_edge_indices] = perturbed_values[user_edge_indices] * self.P_vec

        # 创建扰动后的邻接矩阵
        if isinstance(self.adj, SparseTensor):
            perturbed_adj = SparseTensor(
                row=row, col=col, value=perturbed_values, 
                sparse_sizes=self.adj.sparse_sizes()
            )
        else:
            # 对于非SparseTensor类型，创建稀疏矩阵
            perturbed_adj = torch.sparse_coo_tensor(
                torch.stack([row, col]), 
                perturbed_values, 
                self.adj.shape if hasattr(self.adj, 'shape') else (self.num_users + self.num_items, self.num_users + self.num_items)
            )

        return perturbed_adj

    def _extract_causal_support_and_contributions(self, threshold=0.1):
        """
        提取因果支撑集和贡献度

        参数:
            threshold: 贡献度阈值，低于此值的交互不被认为是因果支撑

        返回:
            causal_support_set: 因果支撑集
            contributions: 贡献度函数
        """
        # 贡献度定义为 1 - P_h
        contributions = 1 - self.P_vec.detach()

        # 因果支撑集是贡献度大于阈值的交互
        causal_support_mask = contributions > threshold
        causal_support_set = torch.nonzero(causal_support_mask).squeeze()

        # 如果只有一个元素，转换为列表
        if causal_support_set.dim() == 0:
            causal_support_set = causal_support_set.unsqueeze(0)

        return causal_support_set.tolist(), contributions.tolist()


# 示例使用函数
def example_usage():
    """
    CCI模块使用示例
    """
    # 注意：这只是一个示例，实际使用时需要传入真实的模型和数据
    print("CCI模块使用示例")
    print("请在实际应用中传入预训练的推荐模型和相关数据")


if __name__ == "__main__":
    example_usage()