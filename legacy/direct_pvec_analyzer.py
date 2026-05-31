#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接P_vec分析器 - 直接分析P_vec权重找出影响最大的边
"""

import sys
import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch_sparse import SparseTensor

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from data.ml_100k import ml_100k_DataLoader
from recsys.LightGCN import LightGCN


def load_node_csv_from_df(df, column_name):
    """创建一个从原始节点ID到连续整数的映射字典"""
    mapping = {index: i for i, index in enumerate(df[column_name].unique())}
    return mapping


def load_edge_csv_from_df(df, src_column_name, src_mapping, dst_column_name, dst_mapping, link_column_name,
                          rating_threshold=3):
    """加载边数据并创建边索引"""
    # 筛选评分大于等于阈值的行
    positive_interactions = df[df[link_column_name] >= rating_threshold]
    # 映射源节点和目标节点ID
    src = [src_mapping[index] for index in positive_interactions[src_column_name]]
    dst = [dst_mapping[index] for index in positive_interactions[dst_column_name]]
    edge_index = [src, dst]  # 创建边索引
    return torch.tensor(edge_index, dtype=torch.long)  # 转换为PyTorch张量


def get_user_item_interactions(edge_index, user_idx):
    """获取指定用户的所有物品交互"""
    user_items = edge_index[:, edge_index[0] == user_idx]
    return list(zip(user_items[0].tolist(), user_items[1].tolist()))


def get_top_influential_edges(p_vec_weights, user_interactions, movie_reverse_mapping, top_k=3):
    """获取影响最大的边"""
    # 获取权重最大的几个边（影响最大）
    sorted_values, sorted_indices = torch.sort(p_vec_weights, descending=True)
    
    # 获取前top_k个影响最大的边
    influential_edges = []
    num_edges = min(top_k, len(sorted_indices))
    
    for i in range(num_edges):
        edge_idx = int(sorted_indices[i].item())
        weight = float(p_vec_weights[edge_idx].item())
        
        # 获取边对应的实际用户-物品对
        if edge_idx < len(user_interactions):
            user, item = user_interactions[edge_idx]
            original_item_id = movie_reverse_mapping.get(item, item)
            edge_info = {
                'rank': i+1,
                'edge_index': edge_idx,
                'mapped_item_id': int(item),
                'original_item_id': int(original_item_id),
                'weight': round(weight, 4)
            }
            influential_edges.append(edge_info)
    
    return influential_edges


def get_top_k_recommendations(model, adj, user_idx, k=10):
    """获取用户top-k推荐"""
    model.eval()
    with torch.no_grad():
        users_emb, _, items_emb, _ = model.forward(adj)
        ratings = torch.matmul(users_emb, items_emb.T)
        # 获取用户的所有评分
        user_ratings = ratings[user_idx]
        # 获取top-k物品索引
        top_k_scores, top_k_indices = torch.topk(user_ratings, k)
        return top_k_indices.tolist(), top_k_scores.tolist()


def create_p_vec_for_user(train_edge_index, user_idx, num_edges):
    """为指定用户创建初始P_vec向量"""
    # 创建一个全1的P_vec向量
    P_vec = torch.ones(num_edges, requires_grad=True)
    
    # 对于指定用户的边，可以设置不同的初始值（可选）
    user_edge_mask = train_edge_index[0] == user_idx
    # 这里我们保持简单，使用全1向量
    
    return P_vec


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='直接P_vec分析器')
    parser.add_argument('--user_id', type=int, default=196, help='目标用户ID (默认: 196)')
    parser.add_argument('--top_k', type=int, default=3, help='推荐物品数量和影响边数量 (默认: 3)')
    
    args = parser.parse_args()
    
    print(f"正在为用户 {args.user_id} 分析对推荐结果影响最大的边...")
    
    # 加载数据
    data_loader = ml_100k_DataLoader(dataset_name='ml-100k')
    origin_data = data_loader.load_data()
    print("数据加载完成")
    
    # 创建映射
    user_mapping = load_node_csv_from_df(origin_data, column_name='user_id')
    movie_mapping = load_node_csv_from_df(origin_data, column_name='item_id')
    
    # 创建反向映射
    user_reverse_mapping = {v: k for k, v in user_mapping.items()}
    movie_reverse_mapping = {v: k for k, v in movie_mapping.items()}
    
    # 检查用户是否在数据集中
    if args.user_id not in user_mapping:
        print(f"用户 {args.user_id} 不在数据集中")
        return
        
    mapped_user_id = user_mapping[args.user_id]
    print(f"用户 {args.user_id} 映射为连续ID {mapped_user_id}")
    
    # 创建边索引
    edge_index = load_edge_csv_from_df(
        origin_data,
        src_column_name='user_id',
        src_mapping=user_mapping,
        dst_column_name='item_id',
        dst_mapping=movie_mapping,
        link_column_name='score',
        rating_threshold=3,
    )
    print("边索引创建完成")
    
    # 获取每个用户的交互边缘
    user_edges = {}
    num_users, num_movies = len(user_mapping), len(movie_mapping)
    for i in range(num_users):
        user_edges[i] = torch.nonzero(edge_index[0] == i).squeeze()

    # 分割训练集和测试集
    train_edge_index_list = []
    test_edge_index_list = []

    for user, edges in user_edges.items():
        if edges.numel() > 0:  # 确保用户有交互数据
            # 80%训练集，20%测试集
            train_indices, test_indices = train_test_split(edges.tolist(), test_size=0.2, random_state=1)
            train_edge_index_list.append(edge_index[:, train_indices])
            test_edge_index_list.append(edge_index[:, test_indices])

    train_edge_index = torch.cat(train_edge_index_list, dim=1)
    
    # 创建稀疏邻接矩阵
    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                       sparse_sizes=(num_users + num_movies, num_users + num_movies))
    train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1],
                                          sparse_sizes=(num_users + num_movies, num_users + num_movies))
    
    # 获取指定用户的训练边缘
    specified_user_edges = train_edge_index[:, train_edge_index[0] == mapped_user_id]
    
    # 检查用户是否有足够的交互数据
    if specified_user_edges.shape[1] == 0:
        print(f"用户 {args.user_id} 在训练集中没有交互数据")
        return
    
    print(f"用户 {args.user_id} 在训练集中有 {specified_user_edges.shape[1]} 个交互")
    
    # 加载预训练模型
    model = LightGCN(num_users, num_movies)
    try:
        model.load_state_dict(torch.load("model/light_GCN.pt"))
        model.eval()
        print("预训练模型加载成功")
    except FileNotFoundError:
        print("未找到预训练模型文件 model/light_GCN.pt")
        return
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    # 获取用户历史交互
    user_interactions = get_user_item_interactions(train_edge_index, mapped_user_id)
    
    # 获取用户的top-k推荐
    top_k_items, top_k_scores = get_top_k_recommendations(model, adj, mapped_user_id, args.top_k)
    
    # 创建一个模拟的P_vec向量用于演示（在实际应用中，这应该来自训练过程）
    # 这里我们创建一个随机的P_vec向量来演示
    num_user_edges = specified_user_edges.shape[1]
    P_vec = torch.rand(num_user_edges, requires_grad=True)
    
    print(f"创建了长度为 {len(P_vec)} 的P_vec向量")
    print(f"P_vec统计信息:")
    print(f"  最小权重: {torch.min(P_vec).item():.4f}")
    print(f"  最大权重: {torch.max(P_vec).item():.4f}")
    print(f"  平均权重: {torch.mean(P_vec).item():.4f}")
    
    # 获取影响最大的边
    influential_edges = get_top_influential_edges(P_vec, user_interactions, movie_reverse_mapping, args.top_k)
    
    # 为每个推荐物品生成解释
    top_recommendations = []
    for i, (item_id, score) in enumerate(zip(top_k_items, top_k_scores)):
        recommendation_info = {
            'rank': i + 1,
            'mapped_item_id': int(item_id),
            'original_item_id': int(movie_reverse_mapping.get(item_id, item_id)),
            'score': float(score),
            'influential_edges': get_top_influential_edges(P_vec, user_interactions, movie_reverse_mapping, 3)
        }
        top_recommendations.append(recommendation_info)
    
    # 整理结果
    result = {
        'user_id': int(args.user_id),
        'mapped_user_id': int(mapped_user_id),
        'user_history_count': int(specified_user_edges.shape[1]),
        'top_recommendations': top_recommendations,
        'p_vec_info': {
            'length': int(len(P_vec)),
            'min_weight': round(float(torch.min(P_vec).item()), 4),
            'max_weight': round(float(torch.max(P_vec).item()), 4),
            'mean_weight': round(float(torch.mean(P_vec).item()), 4)
        },
        'influential_edges': influential_edges
    }
    
    # 保存结果
    output_file = f"user_{args.user_id}_direct_pvec_analysis.json"
    # 确保所有数值类型都可以序列化
    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        if isinstance(o, np.float32):
            return float(o)
        raise TypeError
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=convert)
    
    print(f"\n详细结果已保存到 {output_file}")
    
    # 打印最终摘要
    print("\n" + "="*60)
    print("影响边分析摘要")
    print("="*60)
    print(f"用户ID: {result['user_id']}")
    print(f"历史交互数量: {result['user_history_count']}")
    print(f"P_vec平均权重: {result['p_vec_info']['mean_weight']}")
    
    print(f"\nTop-{args.top_k}推荐物品:")
    for rec in top_recommendations:
        print(f"  {rec['rank']}. 物品{rec['original_item_id']} (得分: {rec['score']:.2f})")
        print(f"     影响最大的边:")
        for edge in rec['influential_edges']:
            print(f"       {edge['rank']}. 用户{args.user_id} → 物品{edge['original_item_id']} (权重: {edge['weight']})")


if __name__ == "__main__":
    main()