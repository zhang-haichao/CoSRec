import torch
from torch.nn.parameter import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv
import torch  # PyTorch深度学习框架，用于建立和训练模型
from torch import nn, optim, Tensor  # nn: 网络层; optim: 优化器; Tensor: 张量
from torch_sparse import SparseTensor, matmul  # 处理稀疏张量的操作，例如矩阵乘法
from torch_geometric.nn.conv.gcn_conv import gcn_norm  # GCN标准化邻接矩阵
from torch_geometric.nn.conv import MessagePassing  # 消息传递机制，用于自定义图卷积层


# 定义一个 LGCNPerturb 类，它继承自 nn.Module
class LGCNPerturb(MessagePassing):
    def __init__(self, num_users, num_items, user_idx, user_hist_count, embedding_dim=64, K=3, edge_additions=False, add_self_loops=True):
        super().__init__()
        # 初始化类的基本参数
        self.num_users, self.num_items = num_users, num_items  # 用户数和物品数
        self.num_nodes = num_users + num_items  # 节点总数
        self.user_idx = user_idx
        self.user_hist_count = user_hist_count

        self.embedding_dim, self.K = embedding_dim, K  # 嵌入维度和图卷积层数
        self.add_self_loops = add_self_loops  # 是否添加自环
        self.edge_additions = edge_additions  # 是否考虑边的添加作为扰动
        self.edge_index_orig, self.edge_index_pred = None, None

        # 初始化用户和物品的嵌入向量
        self.users_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        # 使用正态分布初始化嵌入向量的权重
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

        # 初始化扰动向量 P_vec，根据是否添加边采用不同的初始化方法
        if self.edge_additions:
            self.P_vec = Parameter(torch.ones(self.user_hist_count)/2)
        else:
            self.P_vec = Parameter(torch.ones(self.user_hist_count)*0.9)

        self.reset_parameters()  # 初始化模型参数

    def reset_parameters(self, eps=0.01):
        with torch.no_grad():
            if self.edge_additions:
                self.P_vec.add_(eps)  # 原地操作，修改P_vec的值
            else:
                self.P_vec.sub_(eps)  # 原地操作，修改P_vec的值

    def modify_adj_weights(self, adj, user_idx, p_vec):

        # 从coo格式获取行和列索引
        row_indices, col_indices, _ = adj.coo()

        # 创建一个全1的权重张量
        weights = torch.ones(adj.nnz())

        # 获取对应user_idx的行索引
        user_mask = (row_indices == user_idx)
        weights[user_mask] = p_vec[:user_mask.sum()]  # 批量更新指定用户的权重

        # 创建一个新的SparseTensor，保留原有的行列索引，但使用更新后的权重
        new_adj = SparseTensor(row=row_indices, col=col_indices, value=weights,
                               sparse_sizes=adj.sparse_sizes())

        return new_adj

    def forward(self, edge_index: SparseTensor):
        # 前向传播函数，接受稀疏邻接矩阵为输入

        if self.edge_index_orig is None:
            # 在第一次前向传播时保存原始邻接矩阵
            self.edge_index_orig = edge_index

        self.edge_index_pred = self.modify_adj_weights(edge_index, self.user_idx, self.P_vec)

        # 取出稀疏矩阵的 COO 格式的行和列 (即源节点和目标节点)
        row, col, value = self.edge_index_pred.coo()

        row1 = torch.cat([row,col], dim=0)
        col1 = torch.cat([col,row], dim=0)
        value1 = torch.cat([value,value], dim=0)

        bidirectional_edge_index = SparseTensor(row=row1, col=col1, value=value1, sparse_sizes=edge_index.sparse_sizes())

        # 对邻接矩阵进行GCN标准化处理，传递edge_weight参数
        edge_index_norm = gcn_norm(
            bidirectional_edge_index,
            edge_weight=bidirectional_edge_index.storage.value(),
            add_self_loops=self.add_self_loops
        )

        # 初始用户和物品的嵌入拼接起来，形成初始嵌入矩阵 E^0
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = [emb_0]  # 记录每一层的嵌入
        emb_k = emb_0

        # 消息传递K次，执行K层的图卷积
        for _ in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        # 将每一层的嵌入取平均，得到最终的嵌入 E^K
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)

        # 将最终的嵌入拆分为用户嵌入和物品嵌入
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])

        # 返回用户和物品的最终嵌入，以及它们的初始嵌入
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        """
        消息传递函数，直接返回接收到的邻居节点的嵌入，不做额外变换
        """
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """
        消息传递与聚合函数，计算标准化邻接矩阵与节点嵌入的乘积，用于更新嵌入
        """
        return matmul(adj_t, x)

