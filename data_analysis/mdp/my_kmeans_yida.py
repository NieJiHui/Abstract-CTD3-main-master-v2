import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import jensenshannon

import sys

from data_analysis.mdp.construct_mdp import generate_graph_from_csv

# sys.path.append("F:\桌面\Abstract-CTD3-main-master\data_analysis\\acc_td3")


# 用于计算交集
class MySet:
    # 根据元组创建集合
    def __init__(self, edge):
        self.mins = edge.action[0][0]
        self.maxs = edge.action[0][1]
        self.reward = edge.reward

    def getMin(self):
        return self.mins

    def getMax(self):
        return self.maxs

    def getReward(self):
        return self.reward


def getIntersection(l1, l2):  # 求交集方法,同时计算奖励的最大差值->由于粒度一致
    result = []  # 用来存储l1和l2的所有交集

    # 对输入的两个列表进行排序
    l1.sort(key=lambda x: x.getMin())
    l2.sort(key=lambda x: x.getMin())

    i = 0
    j = 0
    while i < len(l1) and j < len(l2):
        s1 = l1[i]  # 在方法里调用MySet类
        s2 = l2[j]
        if s1.getMin() < s2.getMin():
            if s1.getMax() < s2.getMin():  # 第一种时刻，交集为空，不返回
                i += 1
            elif s1.getMax() <= s2.getMax():  # 第二种时刻
                result.append(MySet(l2[j]))
                i += 1
            else:  # 第三种时刻第二种情况
                result.append(MySet(l2[j]))
                j += 1
        elif s1.getMin() <= s2.getMax():
            if s1.getMax() <= s2.getMax():  # 第三种时刻第一种情况
                result.append(MySet(l1[j]))
                i += 1
            else:  # 第四种时刻
                result.append(MySet(l1[j]))
                j += 1
        else:  # 第五种时刻
            j += 1
    return result


def is_equal(a, b, tol=None):
    if tol is None:
        tol = sys.float_info.epsilon
    return abs(a - b) < tol

class CustomKMeans(KMeans):
    def __init__(self, n_clusters=8, init='k-means++', max_iter=300, tol=1e-4, random_state=None):
        super().__init__(n_clusters=n_clusters, init=init, max_iter=max_iter, tol=tol, random_state=random_state)
        self.cr = 0.5
        self.cd = 0.5
        self.cp = 0.5

    def _transform(self, X):
        # 自定义距离函数
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.array([self.distance(x, self.cluster_centers_[i]) for x in X])
        return distances

    def distance(self, x, y):
        # 判断两个状态是否相同
        if x.state == y.state:
            return 0

        lx, ly = [], []  # 统计动作区间
        prob_x, prob_y = [], []  # 统计概率
        action_x, action_y = [], [] # action具体值
        max_reward_difference = 0 # 奖励的最大差值

        # 读取动作元组,概率分布
        for edge_x in x.edges:
            lx.append(MySet(edge_x))
            prob_x.append(edge_x.prob)
            action_x.append((edge_x.action[0][0] + edge_x.action[0][1]) / 2)
        for edge_y in y.edges:
            lx.append(MySet(edge_y))
            prob_y.append(edge_y.prob)
            action_y.append((edge_y.action[0][0] + edge_y.action[0][1]) / 2)
        result = getIntersection(lx, ly)  # 动作区间交集

        # 需要保证动作有交集，才有最大奖励差
        if len(result) != 0:
            # 计算奖励的最大差值
            min_reward = result[0].getReward()
            max_reward = result[0].getReward()
            for customSet in result:
                if min_reward > customSet.getReward():
                    min_reward = customSet.getReward()
                if max_reward < customSet.getReward():
                    max_reward = customSet.getReward()
            max_reward_difference = max_reward - min_reward

        # 状态本身之间的距离
        state_distance = np.linalg.norm(np.array(x.state) - np.array(y.state))

        # 后继状态分布之间的距离——詹森-香农距离：衡量两个概率分布之间差异的距离度量，KL散度的拓展
        max_len = max(len(prob_x), len(prob_y))
        if len(prob_x) == max_len:
            for _ in range(max_len - len(prob_x)):
                prob_x.append(0)
        else:
            for _ in range(max_len - len(prob_y)):
                prob_y.append(0)
        distribution_difference = jensenshannon(prob_x, prob_y)

        max_action_difference = max(abs(max(action_x) - min(action_y)), abs(min(action_x) - max(action_y)))
        print(self.cr * state_distance + max_reward_difference + self.cd * max_action_difference + self.cd * distribution_difference)
        return self.cr * state_distance + max_reward_difference + self.cd * max_action_difference + self.cd * distribution_difference


graph = generate_graph_from_csv("res1.csv")
print(graph[(-0.41, -0.4), (-0.05, -0.04)])
print(len(graph))

# 创建自定义的K-means模型并进行训练
num_clusters = 10
kmeans = CustomKMeans(n_clusters=num_clusters)
# 计算自定义距离矩阵
distance = kmeans.distance(graph[(-0.05, -0.04), (0.15, 0.16)],graph[(-0.39, -0.38), (0.05, 0.06)])
print("成功！！")


# # 自定义数据点类，这里使用construct_mdp.node和edge
# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#
# # 创建数据点
# points = [Point(1, 2), Point(3, 4), Point(5, 6), Point(7, 8), Point(9, 10)]
# data = np.array([[p.x, p.y] for p in points])
#
# # 创建自定义的K-means模型并进行训练
# num_clusters = 2
# kmeans = CustomKMeans(n_clusters=num_clusters)
# kmeans.fit(data)
#
# # 保存模型到文件
# model_file = 'kmeans_model.pkl'
# with open(model_file, 'wb') as f:
#     pickle.dump(kmeans, f)
#
# # 加载模型
# with open(model_file, 'rb') as f:
#     loaded_kmeans = pickle.load(f)
#
# # 预测新数据
# new_data = np.array([[2, 3], [6, 7]])
# labels = loaded_kmeans.predict(new_data)
#
# print("数据点集合：")
# for point in points:
#     print(f"({point.x}, {point.y})")
#
# print("\n聚类标签：")
# print(labels)
