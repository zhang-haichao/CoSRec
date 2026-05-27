import torch
import numpy as np
from unlearning.cci import CausalContributionIdentifier
from utils.utils import get_topK_recommendations


class UnlearningDrivenRankAdjuster:
    """
    遗忘驱动的排序调整器 (Unlearning-driven Rank Adjuster, URA)
    该模块利用因果贡献信息来调整推荐列表，实现高效遗忘
    """

    def __init__(self, model, num_users, num_items, user_idx, adj, cci_module=None):
        """
        初始化URA模块

        参数:
            model: 预训练的推荐模型
            num_users: 用户数量
            num_items: 物品数量
            user_idx: 目标用户索引
            adj: 邻接矩阵
            cci_module: 预先创建的CCI模块（可选）
        """
        self.model = model
        self.num_users = num_users
        self.num_items = num_items
        self.user_idx = user_idx
        self.adj = adj

        # 如果没有提供CCI模块，则创建一个新的
        if cci_module is None:
            self.cci = CausalContributionIdentifier(model, num_users, num_items, user_idx, adj)
        else:
            self.cci = cci_module

        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False

    def adjust_ranking_for_unlearning(self, forget_item, top_k=10, test_edge_index=None, 
                                      exclude_edge_indices=None):
        """
        基于遗忘请求调整推荐排名

        参数:
            forget_item: 请求遗忘的物品索引
            top_k: 推荐列表长度
            test_edge_index: 测试集边索引
            exclude_edge_indices: 需要排除的边索引列表

        返回:
            original_recommendations: 原始推荐列表
            adjusted_recommendations: 调整后的推荐列表
        """
        # 获取原始推荐列表
        users_emb, _, items_emb, _ = self.model.forward(self.adj)
        rating_original = torch.matmul(users_emb, items_emb.T)
        
        # 获取Top-K推荐
        top_k_indices_original = get_topK_recommendations(
            rating_original, top_k, test_edge_index, exclude_edge_indices
        )
        original_recommendations = torch.tensor(top_k_indices_original.get(self.user_idx, []))

        # 如果没有推荐结果，直接返回
        if len(original_recommendations) == 0:
            return [], []

        # 为每个推荐物品计算因果贡献
        item_contributions = {}
        for item in original_recommendations:
            # 使用CCI模块识别因果贡献
            causal_support_set, contributions = self.cci.identify_causal_contributions(
                item.item(), num_epochs=50  # 减少训练轮数以提高效率
            )
            item_contributions[item.item()] = (causal_support_set, contributions)

        # 调整推荐分数
        adjusted_scores = rating_original[self.user_idx].clone()
        
        for item in original_recommendations:
            item_idx = item.item()
            causal_support_set, contributions = item_contributions[item_idx]
            
            # 检查遗忘物品是否在因果支撑集中
            if forget_item in causal_support_set:
                # 获取该遗忘物品的贡献度
                contribution_idx = causal_support_set.index(forget_item)
                contribution_value = contributions[contribution_idx]
                
                # 调整分数：score'(i) = score(i) * (1 - wi(h_f))
                original_score = adjusted_scores[item_idx]
                adjusted_scores[item_idx] = original_score * (1 - contribution_value)

        # 基于调整后的分数重新排序
        _, adjusted_indices = torch.topk(adjusted_scores, min(top_k, len(adjusted_scores)))
        adjusted_recommendations = adjusted_indices

        return original_recommendations.tolist(), adjusted_recommendations.tolist()

    def batch_adjust_ranking_for_unlearning(self, forget_items, top_k=10, test_edge_index=None, 
                                            exclude_edge_indices=None):
        """
        批量处理多个遗忘请求

        参数:
            forget_items: 请求遗忘的物品索引列表
            top_k: 推荐列表长度
            test_edge_index: 测试集边索引
            exclude_edge_indices: 需要排除的边索引列表

        返回:
            results: 包含原始和调整后推荐列表的字典
        """
        results = {}
        
        # 获取原始推荐列表
        users_emb, _, items_emb, _ = self.model.forward(self.adj)
        rating_original = torch.matmul(users_emb, items_emb.T)
        
        # 获取Top-K推荐
        top_k_indices_original = get_topK_recommendations(
            rating_original, top_k, test_edge_index, exclude_edge_indices
        )
        original_recommendations = torch.tensor(top_k_indices_original.get(self.user_idx, []))

        # 如果没有推荐结果，返回空结果
        if len(original_recommendations) == 0:
            for forget_item in forget_items:
                results[forget_item] = {
                    'original': [],
                    'adjusted': []
                }
            return results

        # 为每个推荐物品计算因果贡献（只计算一次以提高效率）
        item_contributions = {}
        for item in original_recommendations:
            # 使用CCI模块识别因果贡献
            causal_support_set, contributions = self.cci.identify_causal_contributions(
                item.item(), num_epochs=30  # 进一步减少训练轮数
            )
            item_contributions[item.item()] = (causal_support_set, contributions)

        # 处理每个遗忘请求
        for forget_item in forget_items:
            # 调整推荐分数
            adjusted_scores = rating_original[self.user_idx].clone()
            
            for item in original_recommendations:
                item_idx = item.item()
                causal_support_set, contributions = item_contributions[item_idx]
                
                # 检查遗忘物品是否在因果支撑集中
                if forget_item in causal_support_set:
                    # 获取该遗忘物品的贡献度
                    contribution_idx = causal_support_set.index(forget_item)
                    contribution_value = contributions[contribution_idx]
                    
                    # 调整分数：score'(i) = score(i) * (1 - wi(h_f))
                    original_score = adjusted_scores[item_idx]
                    adjusted_scores[item_idx] = original_score * (1 - contribution_value)

            # 基于调整后的分数重新排序
            _, adjusted_indices = torch.topk(adjusted_scores, min(top_k, len(adjusted_scores)))
            adjusted_recommendations = adjusted_indices

            results[forget_item] = {
                'original': original_recommendations.tolist(),
                'adjusted': adjusted_recommendations.tolist()
            }

        return results


# 示例使用函数
def example_usage():
    """
    URA模块使用示例
    """
    # 注意：这只是一个示例，实际使用时需要传入真实的模型和数据
    print("URA模块使用示例")
    print("请在实际应用中传入预训练的推荐模型和相关数据")


if __name__ == "__main__":
    example_usage()