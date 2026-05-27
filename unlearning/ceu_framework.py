import torch
import numpy as np
import json
import os
from unlearning.cci import CausalContributionIdentifier
from unlearning.ura import UnlearningDrivenRankAdjuster
from utils.utils import get_user_positive_items


class CEUFramework:
    """
    CEU (Causal Explanation-based Unlearning) 框架
    实现"先解释，后遗忘"的机器学习遗忘新范式
    """

    def __init__(self, model, num_users, num_items, adj, device='cpu'):
        """
        初始化CEU框架

        参数:
            model: 预训练的推荐模型
            num_users: 用户数量
            num_items: 物品数量
            adj: 邻接矩阵
            device: 计算设备
        """
        self.model = model
        self.num_users = num_users
        self.num_items = num_items
        self.adj = adj
        self.device = device

        # 将模型移到指定设备
        self.model.to(self.device)

        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False

        print("CEU框架初始化完成")

    def create_unlearning_module(self, user_idx):
        """
        为特定用户创建遗忘模块

        参数:
            user_idx: 用户索引

        返回:
            ura_module: 为该用户配置的URA模块
        """
        # 创建CCI模块
        cci_module = CausalContributionIdentifier(
            self.model, self.num_users, self.num_items, user_idx, self.adj
        )

        # 创建URA模块
        ura_module = UnlearningDrivenRankAdjuster(
            self.model, self.num_users, self.num_items, user_idx, self.adj, cci_module
        )

        return ura_module

    def process_unlearning_request(self, user_idx, forget_item, top_k=10, 
                                  test_edge_index=None, exclude_edge_indices=None):
        """
        处理单个遗忘请求

        参数:
            user_idx: 请求遗忘的用户索引
            forget_item: 请求遗忘的物品索引
            top_k: 推荐列表长度
            test_edge_index: 测试集边索引
            exclude_edge_indices: 需要排除的边索引列表

        返回:
            result: 包含原始和调整后推荐列表的字典
        """
        print(f"处理用户 {user_idx} 对物品 {forget_item} 的遗忘请求")

        # 创建URA模块
        ura_module = self.create_unlearning_module(user_idx)

        # 调整推荐排名
        original_recs, adjusted_recs = ura_module.adjust_ranking_for_unlearning(
            forget_item, top_k, test_edge_index, exclude_edge_indices
        )

        result = {
            'user_idx': user_idx,
            'forget_item': forget_item,
            'original_recommendations': original_recs,
            'adjusted_recommendations': adjusted_recs
        }

        return result

    def process_batch_unlearning_requests(self, unlearning_requests, top_k=10,
                                         test_edge_index=None, exclude_edge_indices=None):
        """
        批量处理遗忘请求

        参数:
            unlearning_requests: 遗忘请求列表，每个元素为 (user_idx, forget_item) 元组
            top_k: 推荐列表长度
            test_edge_index: 测试集边索引
            exclude_edge_indices: 需要排除的边索引列表

        返回:
            results: 所有请求的处理结果列表
        """
        print(f"批量处理 {len(unlearning_requests)} 个遗忘请求")

        results = []
        # 按用户分组请求以提高效率
        user_requests = {}
        for user_idx, forget_item in unlearning_requests:
            if user_idx not in user_requests:
                user_requests[user_idx] = []
            user_requests[user_idx].append(forget_item)

        # 处理每个用户的请求
        for user_idx, forget_items in user_requests.items():
            print(f"处理用户 {user_idx} 的 {len(forget_items)} 个遗忘请求")

            # 创建URA模块
            ura_module = self.create_unlearning_module(user_idx)

            # 批量调整推荐排名
            user_results = ura_module.batch_adjust_ranking_for_unlearning(
                forget_items, top_k, test_edge_index, exclude_edge_indices
            )

            # 整理结果
            for forget_item, recs in user_results.items():
                result = {
                    'user_idx': user_idx,
                    'forget_item': forget_item,
                    'original_recommendations': recs['original'],
                    'adjusted_recommendations': recs['adjusted']
                }
                results.append(result)

        return results

    def evaluate_unlearning_effectiveness(self, original_recs, adjusted_recs, forget_item):
        """
        评估遗忘效果

        参数:
            original_recs: 原始推荐列表
            adjusted_recs: 调整后推荐列表
            forget_item: 被遗忘的物品

        返回:
            metrics: 评估指标字典
        """
        # 处理空列表情况
        if len(original_recs) == 0:
            return {
                'forget_item_in_original': False,
                'forget_item_in_adjusted': False,
                'forget_item_removed': False,
                'jaccard_similarity': 0.0,
                'list_change_ratio': 0.0
            }

        # 检查被遗忘物品是否还在推荐列表中
        forget_in_original = forget_item in original_recs
        forget_in_adjusted = forget_item in adjusted_recs

        # 计算推荐列表的变化
        original_set = set(original_recs)
        adjusted_set = set(adjusted_recs)
        
        # 处理空集合情况
        if len(original_set) == 0 and len(adjusted_set) == 0:
            jaccard_similarity = 1.0
        elif len(original_set) == 0 or len(adjusted_set) == 0:
            jaccard_similarity = 0.0
        else:
            intersection = len(original_set.intersection(adjusted_set))
            union = len(original_set.union(adjusted_set))
            jaccard_similarity = intersection / union if union > 0 else 0

        # 推荐列表变化比例
        if len(original_set) > 0:
            list_change_ratio = len(original_set.difference(adjusted_set)) / len(original_set)
        else:
            list_change_ratio = 0.0

        metrics = {
            'forget_item_in_original': forget_in_original,
            'forget_item_in_adjusted': forget_in_adjusted,
            'forget_item_removed': forget_in_original and not forget_in_adjusted,
            'jaccard_similarity': jaccard_similarity,
            'list_change_ratio': list_change_ratio
        }

        return metrics

    def save_unlearning_results(self, results, file_path):
        """
        保存遗忘结果到文件

        参数:
            results: 遗忘结果列表
            file_path: 保存文件路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 转换结果格式以便JSON序列化
        serializable_results = []
        for result in results:
            serializable_result = {
                'user_idx': int(result['user_idx']) if isinstance(result['user_idx'], (int, float)) else result['user_idx'],
                'forget_item': int(result['forget_item']) if isinstance(result['forget_item'], (int, float)) else result['forget_item'],
                'original_recommendations': [int(item) if isinstance(item, (int, float)) else item for item in result['original_recommendations']],
                'adjusted_recommendations': [int(item) if isinstance(item, (int, float)) else item for item in result['adjusted_recommendations']]
            }
            serializable_results.append(serializable_result)

        # 保存到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        print(f"遗忘结果已保存到 {file_path}")

    def load_unlearning_results(self, file_path):
        """
        从文件加载遗忘结果

        参数:
            file_path: 文件路径

        返回:
            results: 遗忘结果列表
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            serializable_results = json.load(f)

        # 转换回原始格式
        results = []
        for result in serializable_results:
            original_result = {
                'user_idx': result['user_idx'],
                'forget_item': result['forget_item'],
                'original_recommendations': result['original_recommendations'],
                'adjusted_recommendations': result['adjusted_recommendations']
            }
            results.append(original_result)

        return results


# 示例使用函数
def example_usage():
    """
    CEU框架使用示例
    """
    # 注意：这只是一个示例，实际使用时需要传入真实的模型和数据
    print("CEU框架使用示例")
    print("请在实际应用中传入预训练的推荐模型和相关数据")


if __name__ == "__main__":
    example_usage()