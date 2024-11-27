import sys
import time
import os
import json
# 数学计算和多维数组处理
import numpy as np
# 深度学习库
import torch
# 图数据处理
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import structured_negative_sampling, dense_to_sparse
# 数据可视化
import matplotlib.pyplot as plt
# 机器学习模型和推荐系统
from recsys.LightGCN import LightGCN
from cf.cf_explainer import LightGCN_CF
# 数据加载和预处理
from data.ml_100k import ml_100k_DataLoader
from utils.utils import normalize_adj, get_neighbourhood, safe_open, bpr_loss, get_user_positive_items, \
    get_topK_recommendations
# 命令行参数解析
import argparse
# 数据序列化和反序列化
import pickle
# 数据集划分
from sklearn.model_selection import train_test_split


def load_node_csv_from_df(df, column_name):
    # 创建一个从原始节点ID到连续整数的映射字典
    mapping = {index: i for i, index in enumerate(df[column_name].unique())}
    return mapping


def load_edge_csv_from_df(df, src_column_name, src_mapping, dst_column_name, dst_mapping, link_column_name,
                          rating_threshold=3):
    # 筛选评分大于等于阈值的行
    positive_interactions = df[df[link_column_name] >= rating_threshold]
    # 映射源节点和目标节点ID
    src = [src_mapping[index] for index in positive_interactions[src_column_name]]
    dst = [dst_mapping[index] for index in positive_interactions[dst_column_name]]
    edge_index = [src, dst]  # 创建边索引
    return torch.tensor(edge_index, dtype=torch.long)  # 转换为PyTorch张量


def sample_mini_batch(batch_size, edge_index):
    # 生成正负样本对。此函数假设已由torch_geometric提供，
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


def create_full_adj_matrix(edge_index, num_users, num_movies):
    """
    创建全图的邻接矩阵

    参数:
    edge_index (torch.Tensor): 2xN的边索引，其中N是边的总数
    num_users (int): 用户节点的数量
    num_movies (int): 物品（电影）节点的数量

    返回:
    SparseTensor: 全图的邻接矩阵
    """
    # 计算总节点数（用户+物品）
    num_nodes = num_users + num_movies

    # 使用SparseTensor构建邻接矩阵
    adj_matrix = SparseTensor(row=edge_index[0], col=edge_index[1],
                              sparse_sizes=(num_nodes, num_nodes))

    return adj_matrix


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


sys.path.append('..')  # 将上级目录添加到系统路径中


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-100k')  # 默认数据集
    parser.add_argument('--user_id', type=int, default=16,
                        help='The user ID to generate counterfactual explanations for.')  # 添加指定用户参数
    parser.add_argument('--rate', type=float, default=0.1, )  # 抛弃率
    parser.add_argument('--topk', type=int, default=1, )  # 抛弃率

    # 基于原始GCN模型的设置，不做改变
    parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')  # 隐藏层单元数
    parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')  # 卷积层的数量
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (between 0 and 1)')  # Dropout比率

    # 对解释器的设置
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')  # 随机种子
    parser.add_argument('--lr', type=float, default=0.08, help='Learning rate for explainer')  # 解释器的学习率
    parser.add_argument('--optimizer', type=str, default="SGD", help='SGD or Adadelta')  # 优化器类型
    parser.add_argument('--n_momentum', type=float, default=0.0, help='Nesterov momentum')  # Nesterov动量
    parser.add_argument('--beta', type=float, default=0.5, help='Tradeoff for dist loss')  # 距离损失的权重
    parser.add_argument('--num_epochs', type=int, default=50, help='Num epochs for explainer')  # 解释器的训练周期
    parser.add_argument('--device', default='cpu', help='CPU or GPU.')  # 计算设备
    args = parser.parse_args()

    print(args)

    # 设置计算设备
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)  # 设置NumPy的随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch的随机种子
    torch.autograd.set_detect_anomaly(True)  # 开启异常检测，有助于调试

    # 使用DataLoader类加载数据
    data_loader = ml_100k_DataLoader(dataset_name='ml-100k')
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
    # 创建全图的邻接矩阵
    num_users, num_movies = len(user_mapping), len(movie_mapping)
    adj = create_full_adj_matrix(edge_index, num_users, num_movies)

    # 获取每个用户的交互边缘
    user_edges = {}
    for i in range(num_users):
        user_edges[i] = torch.nonzero(edge_index[0] == i).squeeze()

    # 用于存储训练集和测试集的边缘索引
    train_edge_index = []
    test_edge_index = []

    # 按照 80:20 分割每个用户的交互
    for user, edges in user_edges.items():
        # 80%训练集，20%测试集
        train_indices, test_indices = train_test_split(edges.tolist(), test_size=0.2, random_state=1)

        # 分配给训练集、测试集
        train_edge_index.append(edge_index[:, train_indices])
        test_edge_index.append(edge_index[:, test_indices])

    # 合并所有用户的边缘索引
    train_edge_index = torch.cat(train_edge_index, dim=1)
    test_edge_index = torch.cat(test_edge_index, dim=1)

    # 将边缘索引转换为稀疏张量
    train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1],
                                           sparse_sizes=(num_users + num_movies, num_users + num_movies))
    test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1],
                                          sparse_sizes=(num_users + num_movies, num_users + num_movies))

    # 仅筛选训练集中的指定用户交互边缘
    specified_user_edges = train_edge_index[:, train_edge_index[0] == args.user_id]
    specified_sparse_edge_index = SparseTensor(row=specified_user_edges[0], col=specified_user_edges[1],
                                               sparse_sizes=(num_users + num_movies, num_users + num_movies))

    #  原模型
    rec_model = LightGCN(num_users, num_movies)

    # 对指定用户的每个物品生成反事实解释
    specified_user_movie_pairs = list(zip(specified_user_edges[0].tolist(), specified_user_edges[1].tolist()))

    # 打印一些用户-物品对以进行验证
    print(f"指定用户 {args.user_id} 的用户-物品对样例:", specified_user_movie_pairs[:5])

    start = time.time()  # 记录开始时间
    train_cf_examples = []

    explainer = LightGCN_CF(rec_model, num_users, num_movies, args.user_id, specified_sparse_edge_index, No_x=args.topk,
                            perd_rate=args.rate)

    cf_example = explainer.explain(cf_optimizer=args.optimizer, lr=args.lr,
                                   n_momentum=args.n_momentum, num_epochs=args.num_epochs,
                                   test_edge_index=specified_user_edges,
                                   exclude_edge_indices=[specified_user_edges])
    train_cf_examples.append(cf_example)
    print("Time for {} epochs of one example: {:.4f}min".format(args.num_epochs, (time.time() - start) / 60))

    print(train_cf_examples)

    # 保存反事实解释示例
    # 首先，创建目录（如果不存在）
    output_dir = os.path.join('results', args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # 构建文件名
    filename = f"top_{args.topk}_rate{args.rate}_lr{args.lr}_beta{args.beta}_mom{args.n_momentum}_epochs{args.num_epochs}_seed{args.seed}.json"
    file_path = os.path.join(output_dir, filename)

    # 检查文件是否存在
    if os.path.exists(file_path):
        # 如果文件存在，加载已有的数据
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    # 确保 existing_data 是一个字典
    if not isinstance(existing_data, dict):
        existing_data = {}

    # 假设 train_cf_examples 是一个列表，每个元素是 [user_idx, user_hist_count, rate, original_recommendations_list, perturbed_recommendations_list]

    # 更新 existing_data
    for cf_example in train_cf_examples:
        user_idx = cf_example[0]
        existing_data[str(user_idx)] = cf_example[1:]  # 将其余的数据保存为值

    # 将更新后的数据保存回文件
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"反事实解释示例已保存到 {file_path}")


if __name__ == '__main__':
    main()
