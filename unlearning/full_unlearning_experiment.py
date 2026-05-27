import sys
import os
import torch
import numpy as np
import json
import argparse
from sklearn.model_selection import train_test_split

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recsys.LightGCN import LightGCN
from unlearning.ceu_framework import CEUFramework
from data.ml_100k import ml_100k_DataLoader
from utils.utils import get_user_positive_items, get_topK_recommendations
from torch_sparse import SparseTensor


def load_node_csv_from_df(df, column_name):
    """创建节点ID映射"""
    mapping = {index: i for i, index in enumerate(df[column_name].unique())}
    return mapping


def load_edge_csv_from_df(df, src_column_name, src_mapping, dst_column_name, 
                         dst_mapping, link_column_name, rating_threshold=3):
    """创建边索引"""
    # 筛选评分大于等于阈值的行
    positive_interactions = df[df[link_column_name] >= rating_threshold]
    # 映射源节点和目标节点ID
    src = [src_mapping[index] for index in positive_interactions[src_column_name]]
    dst = [dst_mapping[index] for index in positive_interactions[dst_column_name]]
    edge_index = [src, dst]  # 创建边索引
    return torch.tensor(edge_index, dtype=torch.long)  # 转换为PyTorch张量


def create_full_adj_matrix(edge_index, num_users, num_movies):
    """创建全图的邻接矩阵"""
    num_nodes = num_users + num_movies
    adj_matrix = SparseTensor(
        row=edge_index[0], 
        col=edge_index[1] + num_users,  # 物品索引需要偏移
        value=torch.ones(edge_index.shape[1]),
        sparse_sizes=(num_nodes, num_nodes)
    )
    return adj_matrix


def select_forget_items(user_edges, forget_ratio=0.01):
    """
    选择要遗忘的交互项
    
    参数:
        user_edges: 用户边索引
        forget_ratio: 遗忘比例（默认1%）
    
    返回:
        forget_requests: 遗忘请求列表 [(user_idx, item_idx), ...]
    """
    forget_requests = []
    
    # 按用户分组边
    user_items = {}
    for i in range(user_edges.shape[1]):
        user_idx = user_edges[0, i].item()
        item_idx = user_edges[1, i].item()
        if user_idx not in user_items:
            user_items[user_idx] = []
        user_items[user_idx].append(item_idx)
    
    # 为每个用户选择要遗忘的物品
    for user_idx, items in user_items.items():
        num_forget = max(1, int(len(items) * forget_ratio))  # 至少遗忘1个
        if len(items) > 0:
            forget_items = np.random.choice(items, min(num_forget, len(items)), replace=False)
            for item_idx in forget_items:
                forget_requests.append((user_idx, item_idx))
    
    return forget_requests


def evaluate_model_performance(model, edge_index, exclude_edge_indices, k=20):
    """
    评估模型性能
    
    参数:
        model: 推荐模型
        edge_index: 测试边索引
        exclude_edge_indices: 需要排除的边索引列表
        k: Top-K评估指标
    
    返回:
        metrics: 性能指标字典
    """
    # 获取用户和物品嵌入
    with torch.no_grad():
        users_emb = model.users_emb.weight
        items_emb = model.items_emb.weight
        
        # 计算评分矩阵
        rating = torch.matmul(users_emb, items_emb.T)
        
        # 排除训练集中的交互
        for exclude_edge_index in exclude_edge_indices:
            user_pos_items = get_user_positive_items(exclude_edge_index)
            exclude_users = []
            exclude_items = []
            for user, items in user_pos_items.items():
                exclude_users.extend([user] * len(items))
                exclude_items.extend(items)
            if len(exclude_users) > 0 and len(exclude_items) > 0:
                rating[exclude_users, exclude_items] = -1e9  # 设置为极小值
        
        # 计算Top-K推荐
        _, top_K_items = torch.topk(rating, k=min(k, rating.shape[1]))
        
        # 计算评估指标
        users = edge_index[0].unique()
        if len(users) == 0:
            return {'recall@{}'.format(k): 0.0, 'ndcg@{}'.format(k): 0.0}
            
        test_user_pos_items = get_user_positive_items(edge_index)
        test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users if user.item() in test_user_pos_items]
        
        if len(test_user_pos_items_list) == 0:
            return {'recall@{}'.format(k): 0.0, 'ndcg@{}'.format(k): 0.0}
        
        # 确定预测正确性
        r = []
        valid_users = []
        for user in users:
            if user.item() in test_user_pos_items:
                ground_truth_items = test_user_pos_items[user.item()]
                if len(ground_truth_items) > 0 and user < top_K_items.shape[0]:
                    label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
                    r.append(label)
                    valid_users.append(user.item())
        
        if len(r) == 0:
            return {'recall@{}'.format(k): 0.0, 'ndcg@{}'.format(k): 0.0}
            
        r = torch.Tensor(np.array(r).astype('float'))
        
        # 计算召回率
        num_correct_pred = torch.sum(r, dim=-1)
        user_num_liked = torch.Tensor([len(test_user_pos_items[u]) for u in valid_users])
        recall = torch.mean(num_correct_pred / user_num_liked)
        
        # 计算NDCG
        test_matrix = torch.zeros((len(r), k))
        for i, user_idx in enumerate(valid_users):
            items = test_user_pos_items[user_idx]
            length = min(len(items), k)
            test_matrix[i, :length] = 1
        max_r = test_matrix
        idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
        dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
        dcg = torch.sum(dcg, axis=1)
        idcg[idcg == 0.] = 1.
        ndcg = dcg / idcg
        ndcg[torch.isnan(ndcg)] = 0.
        ndcg = torch.mean(ndcg)
        
        return {
            'recall@{}'.format(k): recall.item(),
            'ndcg@{}'.format(k): ndcg.item()
        }


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-100k', help='数据集名称')
    parser.add_argument('--forget_ratio', type=float, default=0.01, help='遗忘比例（默认1%）')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K推荐')
    parser.add_argument('--device', default='cpu', help='计算设备')
    parser.add_argument('--model_path', default='model/light_GCN.pt', help='预训练模型路径')
    args = parser.parse_args()
    
    print("=== CEU框架遗忘实验 ===")
    print(f"数据集: {args.dataset}")
    print(f"遗忘比例: {args.forget_ratio}")
    print(f"设备: {args.device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"警告: 模型文件 {args.model_path} 不存在")
        print("请先运行 recsys/train.py 训练模型")
        return
    
    # 加载数据
    print("加载数据...")
    try:
        data_loader = ml_100k_DataLoader(dataset_name=args.dataset)
        origin_data = data_loader.load_data()
        print(f"成功加载 {len(origin_data)} 条交互记录")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 创建节点映射
    user_mapping = load_node_csv_from_df(origin_data, 'user_id')
    movie_mapping = load_node_csv_from_df(origin_data, 'item_id')
    print(f"用户数: {len(user_mapping)}, 物品数: {len(movie_mapping)}")
    
    # 创建边索引
    edge_index = load_edge_csv_from_df(
        origin_data, 'user_id', user_mapping, 'item_id', movie_mapping, 'score', 3
    )
    print(f"边索引形状: {edge_index.shape}")
    
    # 数据集划分
    num_users, num_movies = len(user_mapping), len(movie_mapping)
    num_interactions = edge_index.shape[1]
    all_indices = list(range(num_interactions))
    
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=42)
    
    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]
    
    # 创建邻接矩阵
    adj = create_full_adj_matrix(train_edge_index, num_users, num_movies)
    print(f"邻接矩阵创建完成")
    
    # 加载预训练模型
    print("加载预训练模型...")
    model = LightGCN(num_users, num_movies)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print(f"成功加载模型: {args.model_path}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("使用随机初始化的模型继续")
    
    model.to(args.device)
    model.eval()
    
    # 评估原始模型性能
    print("评估原始模型性能...")
    try:
        original_metrics = evaluate_model_performance(
            model, test_edge_index, [train_edge_index, val_edge_index], args.top_k
        )
        print(f"原始模型性能: {original_metrics}")
    except Exception as e:
        print(f"模型性能评估失败: {e}")
        original_metrics = {'recall@{}'.format(args.top_k): 0.0, 'ndcg@{}'.format(args.top_k): 0.0}
    
    # 选择要遗忘的交互项
    print(f"选择{args.forget_ratio*100}%的交互项进行遗忘...")
    forget_requests = select_forget_items(train_edge_index, args.forget_ratio)
    print(f"选择 {len(forget_requests)} 个遗忘请求")
    
    if len(forget_requests) == 0:
        print("没有生成遗忘请求，实验结束")
        return
    
    # 创建CEU框架
    print("创建CEU框架...")
    ceu_framework = CEUFramework(model, num_users, num_movies, adj, args.device)
    
    # 处理遗忘请求
    print("处理遗忘请求...")
    try:
        unlearning_results = ceu_framework.process_batch_unlearning_requests(
            forget_requests[:5], args.top_k, test_edge_index, [train_edge_index, val_edge_index]
        )
        print(f"成功处理 {len(unlearning_results)} 个遗忘请求")
    except Exception as e:
        print(f"处理遗忘请求时出错: {e}")
        unlearning_results = []
    
    # 保存结果
    if len(unlearning_results) > 0:
        result_dir = f"results/{args.dataset}"
        os.makedirs(result_dir, exist_ok=True)
        result_file = f"{result_dir}/ceu_unlearning_results_ratio{args.forget_ratio}.json"
        try:
            ceu_framework.save_unlearning_results(unlearning_results, result_file)
            print(f"结果已保存到 {result_file}")
        except Exception as e:
            print(f"保存结果时出错: {e}")
    
    # 评估遗忘效果
    if len(unlearning_results) > 0:
        print("评估遗忘效果...")
        effectiveness_metrics = []
        for result in unlearning_results:
            if len(result['original_recommendations']) > 0:
                metrics = ceu_framework.evaluate_unlearning_effectiveness(
                    result['original_recommendations'],
                    result['adjusted_recommendations'],
                    result['forget_item']
                )
                effectiveness_metrics.append(metrics)
        
        if len(effectiveness_metrics) > 0:
            # 计算平均效果
            avg_forget_removed = np.mean([m['forget_item_removed'] for m in effectiveness_metrics])
            avg_jaccard = np.mean([m['jaccard_similarity'] for m in effectiveness_metrics])
            avg_list_change = np.mean([m['list_change_ratio'] for m in effectiveness_metrics])
            
            print(f"平均遗忘成功率: {avg_forget_removed:.4f}")
            print(f"平均推荐列表相似度 (Jaccard): {avg_jaccard:.4f}")
            print(f"平均推荐列表变化比例: {avg_list_change:.4f}")
            
            # 由于CEU框架不实际更新模型参数，模型性能应保持不变
            try:
                retained_metrics = evaluate_model_performance(
                    model, test_edge_index, [train_edge_index, val_edge_index], args.top_k
                )
                print(f"保留集性能: {retained_metrics}")
            except Exception as e:
                print(f"保留集性能评估失败: {e}")
                retained_metrics = {'recall@{}'.format(args.top_k): 0.0, 'ndcg@{}'.format(args.top_k): 0.0}
            
            # 保存实验总结
            summary = {
                'dataset': args.dataset,
                'forget_ratio': args.forget_ratio,
                'num_forget_requests': len(forget_requests),
                'num_processed_requests': len(unlearning_results),
                'original_performance': original_metrics,
                'retained_performance': retained_metrics,
                'avg_forget_removal_rate': float(avg_forget_removed) if not np.isnan(avg_forget_removed) else 0.0,
                'avg_recommendation_similarity': float(avg_jaccard) if not np.isnan(avg_jaccard) else 0.0,
                'avg_list_change_ratio': float(avg_list_change) if not np.isnan(avg_list_change) else 0.0
            }
            
            summary_file = f"{result_dir}/ceu_experiment_summary_ratio{args.forget_ratio}.json"
            try:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                print(f"实验总结已保存到 {summary_file}")
            except Exception as e:
                print(f"保存实验总结时出错: {e}")
        else:
            print("没有有效的评估指标")
    else:
        print("没有处理任何遗忘请求")
    
    print("=== 实验完成 ===")


if __name__ == "__main__":
    main()