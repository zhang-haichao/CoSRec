import pandas as pd
from sklearn.model_selection import train_test_split
import os

class DataLoader:
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
            df = pd.read_csv(self.file_path, sep='\t', header=None, names=['user_id', 'item_id', 'score', 'time'])
            df.drop(columns=['time'], inplace=True)
            return df
        else:
            pass

    def split_data(self, df):
        train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
        validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        return train_data, validation_data, test_data

    def get_dataset(self):
        df = self.load_data()
        return self.split_data(df)

# 示例使用
if __name__ == "__main__":
    data_loader = DataLoader(dataset_name='ml-100k')
    train_data, validation_data, test_data = data_loader.get_dataset()
    print("训练集:\n", train_data)
    print("验证集样本数:", len(validation_data))
    print("测试集样本数:", len(test_data))
