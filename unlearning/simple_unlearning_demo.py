import sys
import os
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recsys.LightGCN import LightGCN
from unlearning.ceu_framework import CEUFramework
from torch_sparse import SparseTensor


def demo_ceu_framework():
    """
    演示CEU框架的核心功能
    """
    print("=== CEU框架演示 ===")
    
    # 模拟参数
    num_users = 100
    num_items = 50
    user_idx = 5
    forget_item = 10
    
    # 创建模拟邻接矩阵（使用SparseTensor格式以确保兼容性）
    # 创建一些模拟的用户-物品交互
    row = torch.randint(0, num_users, (200,))
    col = torch.randint(0, num_items, (200,))
    # 确保目标用户和物品有交互
    row = torch.cat([row, torch.tensor([user_idx, user_idx, user_idx])])
    col = torch.cat([col, torch.tensor([10, 15, 20])])
    
    # 创建SparseTensor格式的邻接矩阵
    adj = SparseTensor(
        row=row, 
        col=col + num_users,  # 物品索引需要偏移
        value=torch.ones(row.shape[0]),
        sparse_sizes=(num_users + num_items, num_users + num_items)
    )
    
    # 创建模型
    model = LightGCN(num_users, num_items)
    
    # 创建CEU框架
    ceu_framework = CEUFramework(model, num_users, num_items, adj)
    
    # 模拟测试边索引
    test_row = torch.randint(0, num_users, (50,))
    test_col = torch.randint(0, num_items, (50,))
    test_edge_index = torch.stack([test_row, test_col])
    
    # 处理单个遗忘请求
    print(f"处理用户 {user_idx} 对物品 {forget_item} 的遗忘请求...")
    result = ceu_framework.process_unlearning_request(
        user_idx, forget_item, top_k=5,
        test_edge_index=test_edge_index,
        exclude_edge_indices=[]
    )
    
    print("原始推荐列表:", result['original_recommendations'])
    print("调整后推荐列表:", result['adjusted_recommendations'])
    
    # 评估遗忘效果
    metrics = ceu_framework.evaluate_unlearning_effectiveness(
        result['original_recommendations'],
        result['adjusted_recommendations'],
        forget_item
    )
    
    print("遗忘效果评估:")
    print(f"  被遗忘物品在原始列表中: {metrics['forget_item_in_original']}")
    print(f"  被遗忘物品在调整后列表中: {metrics['forget_item_in_adjusted']}")
    print(f"  被遗忘物品成功移除: {metrics['forget_item_removed']}")
    print(f"  推荐列表相似度: {metrics['jaccard_similarity']:.4f}")
    print(f"  推荐列表变化比例: {metrics['list_change_ratio']:.4f}")
    
    # 保存结果
    results = [result]
    ceu_framework.save_unlearning_results(
        results, 
        "results/demo_ceu_results.json"
    )
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    demo_ceu_framework()