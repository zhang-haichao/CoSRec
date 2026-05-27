import torch
from torch import nn, Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing


# 定义LightGCN模型
class LightGCN(MessagePassing):
    """
    LightGCN模型实现，参考论文 https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False):
        """
        初始化LightGCN模型

        参数:
            num_users (int): 用户数量
            num_items (int): 物品数量
            embedding_dim (int): 嵌入向量的维度，默认64
            K (int): 消息传递层数，默认3层
            add_self_loops (bool): 是否为邻接矩阵添加自环，默认不添加
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.add_self_loops = add_self_loops

        # 初始化用户和物品的嵌入向量
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        # 使用正态分布初始化嵌入向量的权重
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        """
        LightGCN模型的前向传播

        参数:
            edge_index (SparseTensor): 稀疏的邻接矩阵

        返回:
            返回用户和物品的最终嵌入，以及它们的初始嵌入
        """

        # 取出稀疏矩阵的 COO 格式的行和列 (即源节点和目标节点)
        row, col, value = edge_index.coo()

        row1 = torch.cat([row,col], dim=0)
        col1 = torch.cat([col,row], dim=0)

        if value is None:
            value = (torch.ones(len(row)))

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
