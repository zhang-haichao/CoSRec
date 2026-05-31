#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版的反事实解释生成器
为指定用户生成针对每个推荐物品的反事实解释
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
from cf.cf_explainer import LightGCN_CF
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


def get_recommendations(model, adj, user_idx, k=10):
    """获取指定用户的top-k推荐"""
    # 获取用户和物品的嵌入
    users_emb, _, items_emb, _ = model.forward(adj)
    rating = torch.matmul(users_emb, items_emb.T)
    
    # 获取指定用户的评分
    user_ratings = rating[user_idx]
    
    # 获取top-k物品
    _, top_items = torch.topk(user_ratings, k)
    
    return top_items.tolist()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='为指定用户生成推荐物品的反事实解释')
    parser.add_argument('--user_id', type=int, default=196, help='目标用户ID (默认: 196)')
    parser.add_argument('--top_k', type=int, default=5, help='生成解释的推荐物品数量 (默认: 5)')
    
    args = parser.parse_args()
    
    print(f"正在为用户 {args.user_id} 生成反事实解释...")
    
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
    # test_edge_index = torch.cat(test_edge_index_list, dim=1)
    
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
    
    # 获取推荐结果
    top_items = get_recommendations(model, adj, mapped_user_id, args.top_k)
    print(f"用户 {args.user_id} 的Top-{args.top_k}推荐: {[int(movie_reverse_mapping[item]) for item in top_items]}")
    
    # 为每个推荐物品生成反事实解释
    explanations = {}
    
    for i, target_item in enumerate(top_items[:3]):  # 只处理前3个物品以减少时间
        print(f"\n正在为第 {i+1} 个推荐物品 (原始ID: {int(movie_reverse_mapping[target_item])}) 生成解释...")
        
        try:
            # 创建反事实解释器，针对第i个推荐物品
            explainer = LightGCN_CF(
                model, 
                num_users, 
                num_movies, 
                mapped_user_id, 
                train_sparse_edge_index, 
                target_index=i,  # 解释第i个推荐物品
                perd_rate=0.1
            )
            
            # 生成解释
            cf_result = explainer.explain(
                cf_optimizer="SGD",
                lr=0.08,
                n_momentum=0.0,
                num_epochs=20,  # 减少训练轮数以加快速度
                test_edge_index=specified_user_edges,
                exclude_edge_indices=[specified_user_edges]
            )
            
            # 获取关键边信息
            P_vec = explainer.cf_model.P_vec.detach()
            sorted_values, sorted_indices = torch.sort(P_vec)
            
            # 获取权重最小的3条边（对推荐影响最小的边）
            critical_edges = []
            num_critical = min(3, len(sorted_indices))
            for j in range(num_critical):
                edge_idx = int(sorted_indices[j].item())
                weight = float(P_vec[edge_idx].item())
                critical_edges.append({
                    'edge_index': edge_idx,
                    'weight': round(weight, 4)
                })
            
            # 保存解释结果
            explanations[str(i)] = {
                'original_item_id': int(movie_reverse_mapping[target_item]),
                'mapped_item_id': int(target_item),
                'critical_edges': critical_edges,
                'P_vector_summary': {
                    'min_weight': round(float(torch.min(P_vec).item()), 4),
                    'max_weight': round(float(torch.max(P_vec).item()), 4),
                    'mean_weight': round(float(torch.mean(P_vec).item()), 4)
                }
            }
            
            print(f"  关键边数量: {len(critical_edges)}")
            for j, edge in enumerate(critical_edges):
                print(f"    关键边 {j+1}: 索引={edge['edge_index']}, 权重={edge['weight']}")
                
        except Exception as e:
            print(f"为物品 {int(movie_reverse_mapping[target_item])} 生成解释时出错: {e}")
            explanations[str(i)] = {
                'original_item_id': int(movie_reverse_mapping[target_item]),
                'mapped_item_id': int(target_item),
                'error': str(e)
            }
    
    # 整理结果
    result = {
        'user_id': int(args.user_id),
        'mapped_user_id': int(mapped_user_id),
        'top_k_recommendations': [int(movie_reverse_mapping[item]) for item in top_items[:3]],
        'explanations': explanations
    }
    
    # 保存结果
    output_file = f"user_{args.user_id}_simple_explanations.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n解释结果已保存到 {output_file}")
    
    # 打印摘要
    print("\n=== 解释摘要 ===")
    print(f"用户ID: {result['user_id']}")
    print(f"Top-{len(result['top_k_recommendations'])} 推荐: {result['top_k_recommendations']}")
    print("\n各物品的关键边:")
    for i, (key, explanation) in enumerate(result['explanations'].items()):
        if 'error' not in explanation and 'critical_edges' in explanation:
            item_id = explanation['original_item_id']
            print(f"  物品 {item_id} (第{int(key)+1}个推荐):")
            for j, edge in enumerate(explanation['critical_edges']):
                print(f"    关键边 {j+1}: 索引={edge['edge_index']}, 权重={edge['weight']}")
        else:
            item_id = explanation.get('original_item_id', '未知')
            error_msg = explanation.get('error', '未知错误')
            print(f"  物品 {item_id} (第{int(key)+1}个推荐): 生成解释时出错 ({error_msg})")


if __name__ == "__main__":
    main()