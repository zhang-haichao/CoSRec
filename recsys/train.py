import numpy as np  # 处理多维数组和进行数学计算
import matplotlib.pyplot as plt  # 绘制图表，用于数据可视化
from sklearn.model_selection import train_test_split  # 划分数据集为训练集和测试集
import torch  # PyTorch深度学习框架，用于建立和训练模型
from torch import nn, optim, Tensor  # nn: 网络层; optim: 优化器; Tensor: 张量
from torch_sparse import SparseTensor, matmul  # 处理稀疏张量的操作，例如矩阵乘法
from torch_geometric.utils import structured_negative_sampling  # 结构化负采样，用于图数据
from data.DataLoader import DataLoader  # 自定义数据加载器
from LightGCN import LightGCN
from utils.utils import bpr_loss

def load_node_csv_from_df(df, column_name):
    # 创建一个从原始节点ID到连续整数的映射字典
    mapping = {index: i for i, index in enumerate(df[column_name].unique())}
    return mapping


def load_edge_csv_from_df(df, src_column_name, src_mapping, dst_column_name, dst_mapping, link_column_name, rating_threshold=3):
    # 筛选评分大于等于阈值的行
    positive_interactions = df[df[link_column_name] >= rating_threshold]
    # 映射源节点和目标节点ID
    src = [src_mapping[index] for index in positive_interactions[src_column_name]]
    dst = [dst_mapping[index] for index in positive_interactions[dst_column_name]]
    edge_index = [src, dst]  # 创建边索引
    return torch.tensor(edge_index, dtype=torch.long)  # 转换为PyTorch张量


def sample_mini_batch(batch_size, edge_index):
    # 其中edge_index是图中所有边的索引。
    edges = structured_negative_sampling(edge_index)

    # 将edges堆叠成一个新的Tensor，使其维度为[3, N]，
    # 其中N是edges中边的数量。这样做是为了后续能够方便地处理正负样本。
    edges = torch.stack(edges, dim=0)

    # 随机生成一个排列，并选择前batch_size个索引，
    # 以从edges中随机抽取一个小批量的样本。
    indices = torch.randperm(edges.size(1))[:batch_size]

    # 根据上面随机选择的索引，从edges中抽取对应的小批量样本。
    batch = edges[:, indices]

    # 分解batch为用户索引、正样本项索引和负样本项索引。
    # 这一步是将前面随机抽取的边分成三个部分，
    # 分别对应图中的用户节点、正交互项节点和负交互项节点。
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]

    # 返回分解后的索引，可用于后续的模型训练。
    return user_indices, pos_item_indices, neg_item_indices


# helper function to get N_u
def get_user_positive_items(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items = {}  # 初始化字典
    for i in range(edge_index.shape[1]):  # 遍历边索引的每一列
        user = edge_index[0][i].item()  # 获取用户ID
        item = edge_index[1][i].item()  # 获取物品ID
        if user not in user_pos_items:
            user_pos_items[user] = []  # 如果用户不在字典中，则添加
        user_pos_items[user].append(item)  # 将物品添加到用户的正样本列表中
    return user_pos_items

# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(r, dim=-1)  # 计算每个用户正确预测的数量
    user_num_liked = torch.Tensor([len(groundTruth[i]) for i in range(len(groundTruth))])  # 计算每个用户在测试集中喜欢的物品数量
    recall = torch.mean(num_correct_pred / user_num_liked)  # 计算召回率
    precision = torch.mean(num_correct_pred) / k  # 计算精确率
    return recall.item(), precision.item()

# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k
    参数：
        groundTruth (list): 每个用户高评价物品的列表
        r (list): 列表的列表，指示每个用户推荐的前k个物品是否是前k个真实喜欢的物品
        k (int): 计算ndcg的前k个物品
    返回：
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)  # 确保预测和真实数据长度相同

    test_matrix = torch.zeros((len(r), k))  # 初始化测试矩阵

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)  # 取真实喜好物品数和k中较小的一个
        test_matrix[i, :length] = 1  # 在测试矩阵中标记真实喜好的物品位置
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)  # 计算理想DCG
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))  # 计算DCG
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.  # 防止除以0
    ndcg = dcg / idcg  # 计算归一化DCG
    ndcg[torch.isnan(ndcg)] = 0.  # 处理NaN值
    return torch.mean(ndcg).item()


# wrapper function to get evaluation metrics
def get_metrics(model, edge_index, exclude_edge_indices, k):
    """计算评估指标: recall, precision, 和 ndcg @ k
    参数：
        model (LightGCN): lightgcn模型
        edge_index (torch.Tensor): 评估分割的2 x N边列表
        exclude_edge_indices ([type]): 从评估中排除的边的2 x N列表
        k (int): 计算指标的前k个物品
    返回：
        tuple: recall @ k, precision @ k, 和 ndcg @ k
    """
    # 获取用户和物品的嵌入
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    # 计算每个用户和物品之间的评分 - 形状为 用户数 x 物品数
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        # 从边索引中获取每个用户的正样本物品
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # 获取所有要排除的边的坐标
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # 将排除边的评分设置为大的负值
        rating[exclude_users, exclude_items] = -(1 << 10)

    # 获取每个用户的前k个推荐物品
    _, top_K_items = torch.topk(rating, k=k)

    # 获取评估分割中的所有独特用户
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    # 将测试用户正样本物品字典转换为列表
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    # 确定topk预测的正确性
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg


# wrapper function to evaluate model
def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    """
    评估模型的损失和指标，包括召回率、精确率、NDCG@k
    参数:
        model (LightGCN): LightGCN模型实例
        edge_index (torch.Tensor): 评估分割的2xN边列表
        sparse_edge_index (SparseTensor): 评估分割的稀疏邻接矩阵
        exclude_edge_indices ([type]): 从评估中排除的2xN边列表
        k (int): 计算指标的前k个物品
        lambda_val (float): BPR损失的正则化系数lambda
    返回:
        tuple: BPR损失, 召回率@k, 精确率@k, NDCG@k
    """
    # 获取嵌入
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(sparse_edge_index)
    # 进行结构化负采样
    edges = structured_negative_sampling(edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    # 分别获取用户、正样本物品、负样本物品的最终和初始嵌入
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

    # 计算BPR损失
    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    # 获取评估指标：召回率、精确率和NDCG
    recall, precision, ndcg = get_metrics(model, edge_index, exclude_edge_indices, k)

    return loss, recall, precision, ndcg





# 使用DataLoader类加载数据
data_loader = DataLoader(dataset_name='ml-100k')
Origin_data = data_loader.load_data()
# 验证Origin_data的前几行
print("原始数据的前几行:\n", Origin_data.head())

# 使用修改后的函数和DataFrame创建映射
user_mapping = load_node_csv_from_df(Origin_data, column_name='user_id')
movie_mapping = load_node_csv_from_df(Origin_data, column_name='item_id')
# 打印用户映射和电影映射的一小部分，以确保它们已被正确创建
print("用户映射样例:", dict(list(user_mapping.items())[:5]))
print("电影映射样例:", dict(list(movie_mapping.items())[:5]))

# 使用修改后的函数和DataFrame创建边缘索引
edge_index = load_edge_csv_from_df(
    Origin_data,
    src_column_name='user_id',
    src_mapping=user_mapping,
    dst_column_name='item_id',
    dst_mapping=movie_mapping,
    link_column_name='score',
    rating_threshold=3,
)

# 打印边索引的一小部分，以确保它们已被正确加载
print("边索引样例:", edge_index[:, :5])

# 分割图的边缘使用80/10/10的训练/验证/测试集分割
num_users, num_movies = len(user_mapping), len(movie_mapping)
num_interactions = edge_index.shape[1]
all_indices = list(range(num_interactions))

train_indices, test_indices = train_test_split(
    all_indices, test_size=0.2, random_state=1)
val_indices, test_indices = train_test_split(
    test_indices, test_size=0.5, random_state=1)

train_edge_index = edge_index[:, train_indices]
val_edge_index = edge_index[:, val_indices]
test_edge_index = edge_index[:, test_indices]

# 将边缘索引转换为稀疏张量
train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))
val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))
test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1], sparse_sizes=(
    num_users + num_movies, num_users + num_movies))

model = LightGCN(num_users, num_movies)

# 定义常量
# 总迭代次数: 定义模型训练过程中的总迭代步数
ITERATIONS = 500
# 批大小: 指定每次迭代中用于训练模型的样本数
BATCH_SIZE = 2048
# 学习率: 指定模型训练过程中的学习率，控制权重更新的步长
LR = 2 * 1e-3
# 每隔多少迭代进行一次评估: 定期评估模型性能，帮助监控训练过程

ITERS_PER_EVAL = 50
# 学习率衰减间隔: 指定每隔多少迭代对学习率进行衰减，以细化学习过程
ITERS_PER_LR_DECAY = 200
# 评估指标的K值: 在计算诸如召回率、精确率和NDCG等评估指标时使用的项目数
K = 20
# 正则化系数: 用于防止模型过拟合的正则化项系数
LAMBDA = 1e-6


# 设置设备
device = torch.device('cpu')
print(f"使用设备: {device}.")

# 将模型和数据移至设备
model.to(device)
edge_index.to(device)
train_edge_index.to(device)
train_sparse_edge_index.to(device)
val_edge_index.to(device)
val_sparse_edge_index.to(device)

# 初始化模型和优化器
model.train()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# 训练循环
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None

for iter in range(ITERATIONS):
    model.train()
    # 前向传播
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model(train_sparse_edge_index)

    # 迷你批量
    user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(BATCH_SIZE, train_edge_index)
    user_indices, pos_item_indices, neg_item_indices = user_indices.to(device), pos_item_indices.to(device), neg_item_indices.to(device)

    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

    # 计算损失
    train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)

    # 反向传播
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # 定期评估
    if iter % ITERS_PER_EVAL == 0:
        model.eval()
        with torch.no_grad():
            val_loss, recall, precision, ndcg = evaluation(model, val_edge_index, val_sparse_edge_index, [train_edge_index], K, LAMBDA)
            print(f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")
        train_losses.append(train_loss.item())
        val_losses.append(val_loss)

    # 学习率调度
    if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
        scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()  # Save the current best model state

# 计算每次评估的迭代次数
iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]

# 绘制训练和验证损失曲线
plt.plot(iters, train_losses, label='Train')
plt.plot(iters, val_losses, label='Validation')
plt.xlabel('Iteration ')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 在测试集上评估模型
model.eval()  # 将模型设置为评估模式
test_edge_index = test_edge_index.to(device)
test_sparse_edge_index = test_sparse_edge_index.to(device)

# 调用评估函数
test_loss, test_recall, test_precision, test_ndcg = evaluation(
    model, test_edge_index, test_sparse_edge_index, [train_edge_index, val_edge_index], K, LAMBDA)

# 打印测试结果
print(f"测试损失: {round(test_loss, 5)}, 测试召回率@{K}: {round(test_recall, 5)}, 测试精确度@{K}: {round(test_precision, 5)}, 测试NDCG@{K}: {round(test_ndcg, 5)}")

torch.save(best_model_state, '../model/light_GCN.pt')

