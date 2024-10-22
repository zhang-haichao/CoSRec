import os
import errno
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph


def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0,
             lambda_val):
    """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

	Args:
		users_emb_final (torch.Tensor): e_u_k
		users_emb_0 (torch.Tensor): e_u_0
		pos_items_emb_final (torch.Tensor): positive e_i_k
		pos_items_emb_0 (torch.Tensor): positive e_i_0
		neg_items_emb_final (torch.Tensor): negative e_i_k
		neg_items_emb_0 (torch.Tensor): negative e_i_0
		lambda_val (float): lambda value for regularization loss term

	Returns:
		torch.Tensor: scalar bpr loss value
	"""
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2))  # L2正则化损失

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)  # 正样本的预测得分
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)  # 负样本的预测得分

    # softplus函数计算正样本得分减去负样本得分的对数似然，然后与正则化损失相加。
    # 目的是使模型更倾向于对正样本给出更高的预测得分，从而优化推荐系统的个性化排序
    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss


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


def get_topK_recommendations(rating, k, test_edge_index, exclude_edge_indices):
    """
    获取topK推荐项

    Args:
        rating (torch.Tensor): 用户和物品之间的评分矩阵，形状为用户数 x 物品数
        k (int): 要获取的推荐项数量
        test_edge_index (torch.Tensor): 测试集的边缘索引
        exclude_edge_indices (list[torch.Tensor]): 要排除的边缘索引，通常为训练集和验证集的边缘索引

    Returns:
        dict: 每个测试用户的topK推荐项，键为用户ID，值为推荐的物品ID列表
    """
    # 从边索引中获取每个用户的正样本物品，并排除这些物品
    exclude_users, exclude_items = [], []
    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

    # 将排除边的评分设置为大的负值
    rating[exclude_users, exclude_items] = -(1 << 10)

    topK_recommendations = {}
    for user in test_edge_index[0].unique():
        scores = rating[user]
        _, top_items = torch.topk(scores, k)
        topK_recommendations[user.item()] = top_items.tolist()

    return topK_recommendations


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def safe_open(path, w):
    ''' Open "path" for writing, creating any parent directories as needed.'''
    mkdir_p(os.path.dirname(path))
    return open(path, w)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_degree_matrix(adj):
    return torch.diag(sum(adj))


def normalize_adj(adj):
    # Normalize adjacancy matrix according to reparam trick in GCN paper
    A_tilde = adj + torch.eye(adj.shape[0])
    D_tilde = get_degree_matrix(A_tilde)
    # Raise to power -1/2, set all infs to 0s
    D_tilde_exp = D_tilde ** (-1 / 2)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

    # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
    norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
    return norm_adj


def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])  # Get all nodes involved
    edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)  # Get relabelled subset of edges
    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    sub_feat = features[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    new_index = np.array([i for i in range(len(edge_subset[0]))])
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))  # Maps orig labels to new
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    return sub_adj, sub_feat, sub_labels, node_dict


def create_symm_matrix_from_vec(vector, n_rows):
    matrix = torch.zeros(n_rows, n_rows)
    idx = torch.tril_indices(n_rows, n_rows)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def get_S_values(pickled_results, header):
    df_prep = []
    for example in pickled_results:
        if example != []:
            df_prep.append(example[0])
    return pd.DataFrame(df_prep, columns=header)


def redo_dataset_pgexplainer_format(dataset, train_idx, test_idx):
    dataset.data.train_mask = index_to_mask(train_idx, size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(test_idx[len(test_idx)], size=dataset.data.num_nodes)
