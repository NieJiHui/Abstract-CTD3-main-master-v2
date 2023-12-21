from sklearn.cluster import KMeans
import joblib
import numpy as np
import pandas as pd

import numpy as np
import pickle


class MiniBatchKMeansCustom:
    def __init__(self, n_clusters, batch_size=100, max_iters=100):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, data):
        # 随机初始化聚类中心
        self.centroids = data[np.random.choice(len(data), self.n_clusters, replace=False)]

        for _ in range(self.max_iters):
            # 随机抽样小批量数据
            batch_indices = np.random.choice(len(data), self.batch_size, replace=False)
            batch_data = data[batch_indices]

            # 计算小批量数据到聚类中心的距离
            distances = np.linalg.norm(batch_data[:, np.newaxis, :] - self.centroids, axis=2)

            # 分配每个样本到最近的聚类中心
            labels = np.argmin(distances, axis=1)

            # 更新聚类中心（采用均值更新）
            for i in range(self.n_clusters):
                if np.sum(labels == i) > 0:
                    self.centroids[i] = np.mean(batch_data[labels == i], axis=0)

    def predict(self, data):
        distances = np.linalg.norm(data[:, np.newaxis, :] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def get_cluster_centers(self, label=None):
        if label is not None:
            return self.centroids[label]
        else:
            return self.centroids

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)



if __name__ == '__main__':
    # 示例用法
    # 生成一些示例数据
    data_test = np.random.rand(1000, 2)
    # Load the dataset
    file_path = '/Users/akihi/Downloads/coding?/Abstract-CTD3-main-master/data_analysis/mdp/first_stage_abstract/first_abstract_pro_raw_data_test.csv'  # Replace with your actual file path
    data = pd.read_csv(file_path)

    # Convert string representations of lists into actual lists
    data['State'] = data['State'].apply(eval)
    data['NextState'] = data['NextState'].apply(eval)

    # Combine 'State' and 'NextState' into a single list and remove duplicates
    combined_states = data['State'].tolist() + data['NextState'].tolist()
    combined_states = [list(x) for x in set(tuple(x) for x in combined_states)]
    combined_states = np.array(combined_states)

    # 创建并训练 Mini-Batch KMeans 模型
    mini_batch_kmeans = MiniBatchKMeansCustom(n_clusters=3, batch_size=10)
    mini_batch_kmeans.fit(combined_states)

    # 保存模型
    model_file_path = 'mini_batch_kmeans_model.pkl'
    mini_batch_kmeans.save_model(model_file_path)

    # 加载模型
    loaded_mini_batch_kmeans = MiniBatchKMeansCustom.load_model(model_file_path)

    # 使用加载的模型进行预测
    new_data = np.random.rand(10, 2)
    predictions = loaded_mini_batch_kmeans.predict(new_data)
    print("Predictions:", predictions)
