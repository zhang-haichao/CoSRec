#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对每个推荐物品找到影响最大的边
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


def get_top_influential_edges(cf_model, user_interactions, movie_reverse_mapping, top_k=3):
    """获取影响最大的边"""
    # 获取P_vec向量
    P_vec = cf_model.P_vec.detach()
    
    # 获取权重最大的几个边（影响最大）
    sorted_values, sorted_indices = torch.sort(P_vec, descending=True)
    
    # 获取前top_k个影响最大的边
    influential_edges = []
    num_edges = min(top_k, len(sorted_indices))
    
    for i in range(num_edges):
        edge_idx = int(sorted_indices[i].item())
        weight = float(P_vec[edge_idx].item())
        
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


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='为指定用户推荐的每个物品找到影响最大的边')
    parser.add_argument('--user_id', type=int, default=196, help='目标用户ID (默认: 196)')
    parser.add_argument('--top_k', type=int, default=5, help='生成解释的推荐物品数量 (默认: 5)')
    parser.add_argument('--influential_edges', type=int, default=3, help='每个物品影响最大的边数量 (默认: 3)')
    
    args = parser.parse_args()
    
    print(f"正在为用户 {args.user_id} 生成推荐物品的影响边分析...")
    
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
    
    # 为每个推荐物品生成解释
    result = {
        'user_id': int(args.user_id),
        'mapped_user_id': int(mapped_user_id),
        'user_history_count': int(specified_user_edges.shape[1]),
        'recommendations': []
    }
    
    print(f"\n开始为用户 {args.user_id} 的Top-{args.top_k}推荐物品生成影响边分析...")
    
    # 为每个推荐位置生成解释
    for rec_index in range(args.top_k):
        print(f"\n正在分析第 {rec_index + 1} 个推荐物品...")
        
        try:
            # 创建反事实解释器，针对第rec_index个推荐物品
            explainer = LightGCN_CF(
                model, 
                num_users, 
                num_movies, 
                mapped_user_id, 
                train_sparse_edge_index, 
                target_index=rec_index,  # 解释第rec_index个推荐物品
                perd_rate=0.1
            )
            
            # 运行训练来优化P_vec向量
            cf_result = explainer.explain(
                cf_optimizer="SGD",
                lr=0.08,
                n_momentum=0.0,
                num_epochs=20,  # 减少训练轮数以加快速度
                test_edge_index=specified_user_edges,
                exclude_edge_indices=[specified_user_edges]
            )
            
            # 获取推荐结果
            if len(cf_result) >= 5:
                original_recommendations = cf_result[3]  # 原始推荐
                perturbed_recommendations = cf_result[4]  # 扰动后推荐
                
                if rec_index < len(original_recommendations):
                    recommended_item = original_recommendations[rec_index]
                    original_item_id = movie_reverse_mapping.get(recommended_item, recommended_item)
                    
                    print(f"  推荐物品: {int(original_item_id)} (映射ID: {recommended_item})")
                    
                    # 获取影响最大的边
                    influential_edges = get_top_influential_edges(
                        explainer.cf_model, 
                        user_interactions, 
                        movie_reverse_mapping, 
                        args.influential_edges
                    )
                    
                    # 保存结果
                    rec_info = {
                        'rank': rec_index + 1,
                        'recommended_item_mapped_id': int(recommended_item),
                        'recommended_item_original_id': int(original_item_id),
                        'influential_edges': influential_edges
                    }
                    
                    result['recommendations'].append(rec_info)
                    
                    print(f"  影响最大的 {len(influential_edges)} 条边:")
                    for edge in influential_edges:
                        print(f"    {edge['rank']}. 物品{edge['original_item_id']} (权重: {edge['weight']})")
                else:
                    print(f"  无法获取第 {rec_index + 1} 个推荐物品")
            else:
                print(f"  无法获取推荐结果")
                
        except Exception as e:
            print(f"  分析第 {rec_index + 1} 个推荐物品时出错: {e}")
            # 即使出错也添加占位信息
            result['recommendations'].append({
                'rank': rec_index + 1,
                'error': str(e)
            })
    
    # 保存结果
    output_file = f"user_{args.user_id}_influential_edges.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到 {output_file}")
    
    # 打印最终摘要
    print("\n" + "="*60)
    print("推荐物品影响边分析摘要")
    print("="*60)
    print(f"用户ID: {result['user_id']}")
    print(f"历史交互数量: {result['user_history_count']}")
    
    for rec in result['recommendations']:
        if 'error' not in rec:
            print(f"\n第 {rec['rank']} 个推荐物品: {rec['recommended_item_original_id']}")
            print("  影响最大的边:")
            for edge in rec['influential_edges']:
                print(f"    {edge['rank']}. 物品{edge['original_item_id']} (权重: {edge['weight']})")
        else:
            print(f"\n第 {rec['rank']} 个推荐物品: 处理出错 ({rec['error']})")


if __name__ == "__main__":
    main()