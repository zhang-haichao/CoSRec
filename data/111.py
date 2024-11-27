import pandas as pd
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict


class ml_100k_DataLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.file_path = self.get_file_path(dataset_name)

    def get_file_path(self, dataset_name: str) -> str:
        # 获取当前文件的目录
        base_dir = os.path.dirname(__file__)
        if dataset_name == "ml-100k":
            # 使用os.path.join构建跨平台的路径
            return os.path.join(base_dir, "ml-100k", "u.data")
        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

    def load_data(self):
        if self.dataset_name == "ml-100k":
            # 读取数据，使用'\t'作为分隔符，列名分别是user_id, item_id, score, time
            df = pd.read_csv(self.file_path, sep='\t', header=None, names=['user_id', 'item_id', 'score', 'time'])
            return df
        else:
            raise ValueError("Unsupported dataset")

    def filter_data_by_user_interaction(self, df):
        # 按用户交互次数进行统计
        user_interaction_count = df.groupby('user_id').size()

        # # 计算交互数量的阈值
        # lower_bound = user_interaction_count.quantile(0.1)  # 交互最少的10%的阈值
        # upper_bound = user_interaction_count.quantile(0.9)  # 交互最多的10%的阈值
        lower_bound = 50
        upper_bound = 200

        # 过滤掉交互次数在阈值外的用户
        valid_users = user_interaction_count[
            (user_interaction_count >= lower_bound) & (user_interaction_count <= upper_bound)].index

        # 过滤掉这些用户的数据
        filtered_df = df[df['user_id'].isin(valid_users)]

        return filtered_df, user_interaction_count

    def split_data_by_user_interaction(self, df):
        # 按用户划分数据集
        train_data = []
        validation_data = []
        test_data = []

        # 按用户ID分组
        grouped = df.groupby('user_id')

        for user_id, group in grouped:
            # 确保按时间顺序排列
            group = group.sort_values(by='time')  # 根据评分或时间排序

            # 计算用户交互数据的大小
            train_size = int(len(group) * 0.8)
            validation_size = int(len(group) * 0.1)

            # 划分数据
            train_data.append(group.iloc[:train_size])
            validation_data.append(group.iloc[train_size:train_size + validation_size])
            test_data.append(group.iloc[train_size + validation_size:])

        # 合并所有用户的训练集、验证集和测试集
        train_data = pd.concat(train_data)
        validation_data = pd.concat(validation_data)
        test_data = pd.concat(test_data)

        return train_data, validation_data, test_data

    def get_user_item_dict(self, df):
        # 获取用户-商品字典，格式为：user_id -> [item_id1, item_id2, ...]
        user_item_dict = defaultdict(list)
        for _, row in df.iterrows():
            user_item_dict[row['user_id']].append(row['item_id'])
        return user_item_dict

    def get_user_item_feature_matrix(self, df):
        # 生成用户-物品特征矩阵
        user_item_dict = self.get_user_item_dict(df)
        user_feature_matrix = {}
        item_feature_matrix = {}

        for user_id, items in user_item_dict.items():
            user_feature_matrix[user_id] = len(items)  # 示例：用户特征为其交互的物品数
            for item_id in items:
                if item_id not in item_feature_matrix:
                    item_feature_matrix[item_id] = 0
                item_feature_matrix[item_id] += 1  # 示例：物品特征为与之交互的用户数

        return user_feature_matrix, item_feature_matrix

    def get_dataset(self):
        df = self.load_data()

        # 数据过滤：过滤掉交互次数最多和最少的10%的用户
        filtered_df, user_interaction_count = self.filter_data_by_user_interaction(df)

        print(filtered_df, user_interaction_count)

        # 数据拆分：每个用户的交互数据按80%：10%：10%比例划分
        train_data, validation_data, test_data = self.split_data_by_user_interaction(filtered_df)

        # 用户-物品特征矩阵
        user_feature_matrix, item_feature_matrix = self.get_user_item_feature_matrix(filtered_df)

        return train_data, validation_data, test_data, user_feature_matrix, item_feature_matrix


# 示例使用
if __name__ == "__main__":
    data_loader = ml_100k_DataLoader(dataset_name='ml-100k')
    train_data, validation_data, test_data, user_feature_matrix, item_feature_matrix = data_loader.get_dataset()

    # 输出数据集的一些信息
    print("训练集样本数:", len(train_data))
    print("验证集样本数:", len(validation_data))
    print("测试集样本数:", len(test_data))

    # 输出用户特征矩阵和物品特征矩阵的部分内容
    print("用户特征矩阵:", {k: user_feature_matrix[k] for k in list(user_feature_matrix.keys())[:5]})
    print("物品特征矩阵:", {k: item_feature_matrix[k] for k in list(item_feature_matrix.keys())[:5]})
