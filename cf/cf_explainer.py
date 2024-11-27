import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch_sparse import SparseTensor
from torch.nn.utils import clip_grad_norm_
from utils.utils import get_degree_matrix
from .gcn_perturb import LGCNPerturb
from utils.utils import normalize_adj, get_topK_recommendations
from torch_geometric.utils import to_dense_adj


class LightGCN_CF(nn.Module):
    def __init__(self, model, num_users, num_items, user_idx, adj, perd_rate, No_x, lambda_val=10.0, beta=0.1):
        """
        初始化LightGCN_CF模型。
        :param model: 预训练的Light GCN模型。
        :param num_users: 用户数量。
        :param num_items: 物品数量。
        """
        super(LightGCN_CF, self).__init__()
        self.model = model  # 预训练的Light GCN模型
        self.lambda_val = lambda_val
        self.beta = beta
        self.num_users = num_users
        self.num_items = num_items
        self.adj = adj  # 邻接矩阵
        # 设置需要解释的节点索引
        self.user_idx = user_idx

        # 解释数量
        self.rate = perd_rate
        # 解释前几个
        self.No_x = No_x

        # 计算self.user_idx用户的历史交互数
        row, col, value = self.adj.coo()
        self.user_hist_count = (row == self.user_idx).sum().item()
        print('user: ', self.user_idx, 'had ', self.user_hist_count, 'history records')

        self.model.load_state_dict(torch.load("model/light_GCN.pt"))
        self.model.eval()

        self.cf_model = LGCNPerturb(self.num_users, self.num_items, self.user_idx, self.user_hist_count,
                                    edge_additions=False)
        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)  # strict=False 允许部分加载模型参数

        # 冻结模型的权重
        for name, param in self.named_parameters():
            print(name)
            if 'users_emb' in name or 'items_emb' in name:
                param.requires_grad = False  # 冻结嵌入层的参数

        # 打印原始模型和CF模型的权重是否需要梯度更新
        for name, param in self.model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.cf_model.named_parameters():
            print("cf model requires_grad: ", name, param.requires_grad)

    def explain(self, cf_optimizer, lr, n_momentum, num_epochs, test_edge_index, exclude_edge_indices):
        # cf_optimizer: 选择的优化器类型（如 SGD 或 Adadelta）
        # lr: 学习率
        # n_momentum: 动量（如果使用 SGD 优化器）
        # num_epochs: 训练的轮数

        # 根据选择的优化器和参数初始化优化器
        if cf_optimizer == "SGD":
            if n_momentum == 0.0:
                self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
            else:
                self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum)
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "Adam" :
            self.cf_optimizer = optim.Adam(self.cf_model.parameters(), lr=lr)

        # 初始化最佳反事实示例和损失
        best_cf_example = []  # 存储最佳反事实示例
        best_loss = np.inf  # 初始最佳损失设置为无穷大
        num_cf_examples = 0  # 计数器，记录找到的反事实示例数量

        # 开始训练过程，对每个 epoch 进行操作
        for epoch in range(num_epochs):
            new_example, loss_total = self.train1(epoch, test_edge_index,
                                                  exclude_edge_indices)  # 对每个 epoch 调用 train 方法进行训练
            # 检查是否找到了更好的反事实示例（即损失更小）
            if new_example != [] and loss_total < best_loss:
                best_cf_example.append(new_example)  # 更新最佳反事实示例
                best_loss = loss_total  # 更新最佳损失
                num_cf_examples += 1  # 增加反事实示例的计数

        # 打印找到的反事实示例的数量和目标节点索引
        print("{} CF examples for user_idx = {}".format(num_cf_examples, self.user_idx))
        print(self.user_idx, self.user_hist_count, self.cf_model.P_vec)

        # 获取原始推荐和扰动推荐
        # 获取原始推荐和扰动推荐

        # Convert adj to COO format
        row, col, value = self.adj.coo()

        # For the original recommendation, create an adjacency matrix with all ones
        adj_values_original = torch.ones(len(row))
        adj_original = SparseTensor(row=row, col=col, value=adj_values_original, sparse_sizes=self.adj.sparse_sizes())

        # For the perturbed recommendation
        P_vec = self.cf_model.P_vec.detach()  # Detach to avoid gradient computation

        # Set the smallest 10% of P_vec to zero, others to one
        num_edges = P_vec.numel()
        num_zero = int(self.rate * num_edges)
        sorted_P_vec, indices = torch.sort(P_vec)
        smallest_indices = indices[:num_zero]

        perturbed_P_vec = torch.ones_like(P_vec)
        perturbed_P_vec[smallest_indices] = 0.0

        # Multiply adj values by perturbed_P_vec
        perturbed_values = 1 * perturbed_P_vec

        adj_perturbed = SparseTensor(row=row, col=col, value=perturbed_values, sparse_sizes=self.adj.sparse_sizes())

        # Use the model to compute recommendations

        # Compute embeddings and ratings with original adj
        users_emb_original, _, items_emb_original, _ = self.model.forward(adj_original)
        rating_original = torch.matmul(users_emb_original, items_emb_original.T)

        # Compute embeddings and ratings with perturbed adj
        users_emb_perturbed, _, items_emb_perturbed, _ = self.model.forward(adj_perturbed)
        rating_perturbed = torch.matmul(users_emb_perturbed, items_emb_perturbed.T)

        # Get top-K recommendations
        top_k = 10  # You can adjust this value as needed

        top_k_indices_original = get_topK_recommendations(rating_original, top_k, test_edge_index, exclude_edge_indices)
        top_k_indices_perturbed = get_topK_recommendations(rating_perturbed, top_k, test_edge_index,
                                                           exclude_edge_indices)

        # For the user of interest
        original_recommendations = torch.tensor(top_k_indices_original[self.user_idx])
        perturbed_recommendations = torch.tensor(top_k_indices_perturbed[self.user_idx])

        # Print the recommendations
        print(f'Final Original recommendations for user {self.user_idx}: {original_recommendations}')
        print(f'Final Perturbed recommendations for user {self.user_idx}: {perturbed_recommendations}')

        # Ensure the recommendations are converted to Python lists
        original_recommendations_list = original_recommendations.tolist()
        perturbed_recommendations_list = perturbed_recommendations.tolist()

        # 返回最佳反事实示例
        return [self.user_idx, self.user_hist_count, self.rate, original_recommendations_list,
                perturbed_recommendations_list]

    def train1(self, epoch, test_edge_index, exclude_edge_indices):
        # 训练过程
        t = time.time()
        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        # 前向传播，得到预测结果
        users_emb_final, _, items_emb_final, _ = self.cf_model.forward(self.adj)
        rating = torch.matmul(users_emb_final, items_emb_final.T)

        # 使用原模型获取原始预测结果
        original_users_emb, _, original_items_emb, _ = self.model.forward(self.adj)
        original_rating = torch.matmul(original_users_emb, original_items_emb.T)

        # 获取推荐结果
        top_k = 10  # 设置top k值，可以根据需要调整

        # 获取原始模型的top k推荐
        top_k_indices_original = get_topK_recommendations(original_rating, top_k, test_edge_index, exclude_edge_indices)
        # 获取扰动后模型的top k推荐
        top_k_indices_perturbed = get_topK_recommendations(rating, top_k, test_edge_index, exclude_edge_indices)

        # 获取目标用户的原始推荐和扰动推荐
        original_recommendations = torch.tensor(top_k_indices_original[self.user_idx])
        perturbed_recommendations = torch.tensor(top_k_indices_perturbed[self.user_idx])

        # 计算图距离损失
        ones_vec = torch.ones_like(self.cf_model.P_vec)
        squared_differences = (self.cf_model.P_vec - ones_vec) ** 2
        loss_graph_dist = torch.mean(squared_differences)

        # L1 正则化项
        l1_loss = torch.linalg.norm(self.cf_model.P_vec,ord=1)

        # 获取原始推荐前X项物品在扰动后推荐得分
        top_k_original = self.No_x # 135
        original_topk_items = original_recommendations[:top_k_original]
        perturbed_scores = rating[self.user_idx, original_topk_items]
        loss_decrease_original_items = torch.mean(perturbed_scores)

        # 总损失函数
        total_loss = self.lambda_val *loss_graph_dist + self.beta * loss_decrease_original_items+0.0002 * l1_loss

        total_loss.backward()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)

        # 在优化器更新参数后
        self.cf_optimizer.step()
        # 将 P_vec 中的负值截断为 0
        self.cf_model.P_vec.data = torch.clamp(self.cf_model.P_vec.data, min=0.01, max=1.0)

        # 打印梯度
        if self.cf_model.P_vec.grad is not None:
            print(f'Epoch {epoch + 1}, P_vec Gradient: {self.cf_model.P_vec.grad.norm().item()}')
        else:
            print(f'Epoch {epoch + 1}, No gradient for P_vec')

        # 打印训练信息
        print(
            f'Epoch: {epoch + 1}, Total Loss: {total_loss.item():.4f},L1 Loss:{l1_loss.item():.4f}, Graph Loss: {loss_graph_dist.item():.4f}, Decrease Original Items Loss: {loss_decrease_original_items.item():.4f}')
        print(self.cf_model.P_vec)
        print(f'Original recommendations for user {self.user_idx}: {original_recommendations}')
        print(f'Perturbed recommendations for user {self.user_idx}: {perturbed_recommendations}')
        print(" ")

        cf_stats = []
        if any([loss_graph_dist, loss_decrease_original_items]):
            cf_stats = [self.user_idx, total_loss.item(), loss_graph_dist.item(),
                        loss_decrease_original_items.item()]

        return cf_stats, total_loss.item()
