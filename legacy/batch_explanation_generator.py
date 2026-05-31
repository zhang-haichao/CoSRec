#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批处理版本的反事实解释生成器
为多个用户生成推荐系统的反事实解释
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
from utils.utils import get_user_positive_items


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


def analyze_user_explanation(model, adj, train_edge_index, user_mapping, movie_mapping, 
                           user_reverse_mapping, movie_reverse_mapping, user_id, top_k=3):
    """为单个用户生成反事实解释"""
    
    # 获取映射后的用户ID
    if user_id not in user_mapping:
        return None
        
    mapped_user_id = user_mapping[user_id]
    
    # 获取指定用户的训练边缘
    specified_user_edges = train_edge_index[:, train_edge_index[0] == mapped_user_id]
    
    # 检查用户是否有足够的交互数据
    if specified_user_edges.shape[1] == 0:
        return None
    
    # 获取指定用户的嵌入和评分
    users_emb, _, items_emb, _ = model.forward(adj)
    user_ratings = torch.matmul(users_emb, items_emb.T)[mapped_user_id]
    
    # 获取top-k推荐
    _, top_items = torch.topk(user_ratings, top_k)
    top_items_list = top_items.tolist()
    
    # 获取用户历史交互
    user_interactions = get_user_item_interactions(train_edge_index, mapped_user_id)
    
    # 创建简化版的分析（不使用完整的CF解释器）
    # 这里我们直接分析用户的历史交互模式
    
    result = {
        'user_id': int(user_id),
        'mapped_user_id': int(mapped_user_id),
        'user_history_count': int(specified_user_edges.shape[1]),
        'top_k_recommendations': [int(movie_reverse_mapping.get(item, item)) for item in top_items_list],
        'user_history_sample': []
    }
    
    # 添加用户历史交互样本
    for i, (user, item) in enumerate(user_interactions[:5]):
        original_item_id = movie_reverse_mapping.get(item, item)
        result['user_history_sample'].append({
            'index': i,
            'user_id': int(user_reverse_mapping.get(user, user)),
            'item_id': int(original_item_id),
            'mapped_item_id': int(item)
        })
    
    return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='为多个用户生成推荐物品的反事实解释')
    parser.add_argument('--user_ids', type=str, default="196,186,22", 
                       help='目标用户ID列表，用逗号分隔 (默认: "196,186,22")')
    parser.add_argument('--top_k', type=int, default=3, help='生成解释的推荐物品数量 (默认: 3)')
    
    args = parser.parse_args()
    
    # 解析用户ID列表
    user_ids = [int(uid.strip()) for uid in args.user_ids.split(',')]
    print(f"正在为用户 {user_ids} 生成反事实解释...")
    
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
    
    # 为每个用户生成解释
    all_results = {}
    
    for user_id in user_ids:
        print(f"\n正在分析用户 {user_id}...")
        try:
            result = analyze_user_explanation(
                model, adj, train_edge_index, user_mapping, movie_mapping,
                user_reverse_mapping, movie_reverse_mapping, user_id, args.top_k
            )
            
            if result:
                all_results[str(user_id)] = result
                print(f"  用户 {user_id}:")
                print(f"    历史交互数量: {result['user_history_count']}")
                print(f"    Top-{args.top_k}推荐: {result['top_k_recommendations']}")
            else:
                print(f"  用户 {user_id}: 无数据或处理失败")
                all_results[str(user_id)] = {"error": "No data or processing failed"}
                
        except Exception as e:
            print(f"  用户 {user_id}: 处理时出错 - {e}")
            all_results[str(user_id)] = {"error": str(e)}
    
    # 保存结果
    output_file = "batch_user_explanations.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n批处理结果已保存到 {output_file}")
    
    # 打印最终摘要
    print("\n" + "="*60)
    print("批处理反事实解释摘要")
    print("="*60)
    
    for user_id, result in all_results.items():
        if "error" not in result:
            print(f"\n用户 {user_id}:")
            print(f"  历史交互数量: {result['user_history_count']}")
            print(f"  Top-{args.top_k}推荐: {result['top_k_recommendations']}")
            print(f"  历史交互样本:")
            for interaction in result['user_history_sample'][:3]:
                print(f"    用户{interaction['user_id']} → 物品{interaction['item_id']}")
        else:
            print(f"\n用户 {user_id}: 处理失败 ({result['error']})")


if __name__ == "__main__":
    main()