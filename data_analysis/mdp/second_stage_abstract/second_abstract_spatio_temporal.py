import copy
import json
import math
import csv
import ast
import os

import joblib
import numpy as np
import pickle
from joblib import Parallel, delayed

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.samples.definitions import SIMPLE_SAMPLES

from scipy.spatial.distance import jensenshannon
import networkx as nx
import matplotlib.pyplot as plt
import sys
import data_analysis.mdp.second_stage_abstract.findOptimalK as fOK
import utils
from data_analysis.mdp.MDP import Node, Edge, Attr


#   sys.path.append("F:\桌面\Abstract-CTD3-main-master\data_analysis\\acc_td3")


#   把图还原成2darray
def graph2arr(graph):
    lists = []
    for key, value in graph.items():
        ls = []
        #   处理state
        for it in value.state:
            ls = ls + [*it]

        for edge in value.edges:
            #   处理next_state
            ls1 = copy.deepcopy(ls)
            for it in edge.next_node.state:
                ls1 = ls1 + [*it]
            #   处理action reward cost done prob
            for i in range(len(edge.action)):
                ls2 = copy.deepcopy(ls1)
                ls2 = ls2 + [*edge.action[i]]
                ls2 = ls2 + [*edge.reward[i]]
                ls2 = ls2 + [*edge.cost[i]]
                ls2.append(1 if edge.done is True else 0)
                ls2.append(edge.prob)

                lists.append(ls2)

    ret = np.array(lists)
    return ret


def get_mdp_map(input_file):
    mdp_dic = {}
    with open(input_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            state_center = ast.literal_eval(row['State_Center'])
            act_center = ast.literal_eval(row['Act_Center'])
            reward_center = ast.literal_eval(row['Reward_Center'])
            next_state_center = ast.literal_eval(row['NextState_Center'])
            done = ast.literal_eval(row['Done'])
            cost_center = ast.literal_eval(row['Cost_Center'])
            weight = float(row['Weight'])
            probability = float(row['Probability'])

            current_state_key = tuple(state_center)
            next_state_key = tuple(next_state_center)

            if current_state_key in mdp_dic:
                current_state = Node.from_dict(mdp_dic[current_state_key])
                if next_state_key in mdp_dic:
                    next_state = Node.from_dict(mdp_dic[next_state_key])
                else:
                    next_state = Node(next_state_center)
                    mdp_dic[next_state_key] = next_state.to_dict()
                current_state.add_edge(next_state, act_center, reward_center, done, cost_center, weight, probability)
                mdp_dic[current_state_key] = current_state.to_dict()
            else:
                current_state = Node(state_center)
                if next_state_key in mdp_dic:
                    next_state = Node.from_dict(mdp_dic[next_state_key])
                else:
                    next_state = Node(next_state_center)
                # if current_state == next_state:
                #     print(True)
                current_state.add_edge(next_state, act_center, reward_center, done, cost_center, weight, probability)
                mdp_dic[current_state_key] = current_state.to_dict()
                if next_state_key not in mdp_dic:
                    mdp_dic[next_state_key] = next_state.to_dict()

    attr_dic = {}
    for state_key, state in mdp_dic.items():
        attr_dic[state_key] = Attr(Node.from_dict(state)).to_dict()

    return mdp_dic, attr_dic


def visualize_mdp(mdp_dic, save_path=None):
    G = nx.DiGraph()

    for state_key, state_data in mdp_dic.items():
        state_node = Node.from_dict(state_data)
        G.add_node(str(state_key))

        for edge in state_node.edges:
            next_state_key = tuple(edge.next_node.state)
            G.add_node(str(next_state_key))
            G.add_edge(str(state_key), str(next_state_key), action=edge.action, reward=edge.reward,
                       done=edge.done, cost=edge.cost, weight=edge.weight, prob=edge.prob)

    pos = nx.spring_layout(G, seed=42)  # You can choose a different layout if needed
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black',
            font_size=1, edge_color='gray', width=1, alpha=0.7, arrowsize=10)

    edge_labels = {(state,
                    next_state): f"{edge['action']}, R={edge['reward']}, Done={edge['done']}, Cost={edge['cost']}, Weight={edge['weight']}, Prob={edge['prob']:.2f}"
                   for state, next_state, edge in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def convert_to_list(input_data):
    result = []
    for item in input_data:
        if isinstance(item, str):
            # 如果元素是字符串形式的元组，使用 ast.literal_eval 解析
            item = ast.literal_eval(item)
        if isinstance(item, tuple):
            # 如果元素是元组，直接转换为列表
            result.append(list(item))
        else:
            raise ValueError("Invalid element format in the input list")
    return result


def get_mdp_states(mdp_dic, decimal_places=4):
    states = list(mdp_dic.keys())
    datas = convert_to_list(states)
    # datas = [list(ast.literal_eval(item)) for item in states]
    return np.array(datas)


def manhattan_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return abs((vec1 - vec2)).sum()


class SpatioTemporalKMeans(kmeans):
    def __init__(self, config, graph, datas, cr=0.5, cd=0.5, cp=0.5, cs=0.5, k=8, initial_centers=None,
                 tolerance=0.001, ccore=True):
        """
        # config 一阶段抽象的参数conf/eval/...yaml
        # graph dict一阶段抽象后生成的mdp图 key:((upperbound, lowerbound), ...) value:node
        # data 2d-array 把graph.keys()里面的tuple变成list 然后2d列表转数组
        """
        # 初始化中心点和距离函数
        if not initial_centers:
            initial_centers = kmeans_plusplus_initializer(datas, k).initialize()
        metric = distance_metric(type_metric.USER_DEFINED, func=self.distance)

        super().__init__(datas, initial_centers, tolerance, ccore, metric=metric)
        self.cr = cr
        self.cd = cd
        self.cp = cp
        self.cs = cs
        self.config = config
        self.state_dim = config["dim"]["state_dim"]
        self.graph = graph
        self.states = set(graph.keys())
        self.precs = self.get_prec(config)

    #   获取保留几位小数的精度
    def get_prec(self, config):
        """
        输入config是一阶段抽象里面的参数，用里面的粒度来计算保留精度
        如果gran[0] 是 0.1，那么 math.ceil(math.log10(0.1) * -1) 将得到 1，表示这个维度上的状态可以精确到小数点后 1 位。
        """
        ret = []
        gran = config["granularity"]["state_gran"]
        for i in range(self.state_dim):
            ret.append(math.ceil(math.log10(gran[i]) * -1))
        return ret

    #   为输入匹配mdp状态最相近的节点
    def match_node(self, state):
        """
        data 待匹配状态 1darray 从init的data里面提取出来的一行
        return 匹配到的节点 node
        """

        data_tup = tuple(state)

        if data_tup in self.graph.keys():
            return self.graph[data_tup]

        min_state = None
        min_distance = 100
        for k, item in enumerate(self.states):
            #   把元组形式的key变回list
            data_ls = list(item)

            dis = manhattan_distance(data_ls, state)

            if dis < min_distance:
                min_state = item
                min_distance = dis

        return self.graph[min_state]

    #   输入两个概率分布，返回差异
    def compute_distribution_difference(self, prob_x, prob_y):
        max_len = max(len(prob_x), len(prob_y))

        #   保证长度一样，不一样补0到一样
        if len(prob_x) == max_len:
            for _ in range(max_len - len(prob_y)):
                prob_y.append(0)
        else:
            for _ in range(max_len - len(prob_x)):
                prob_x.append(0)

        # 后继状态分布之间的距离——詹森-香农距离：衡量两个概率分布之间差异的距离度量，KL散度的拓展
        return jensenshannon(prob_x, prob_y)

    #   输入两个attr奖励的差值
    def get_reward_distance(self, attr_x, attr_y):
        sum_x = 0
        sum_y = 0
        for i in range(len(attr_x['rewards'])):
            sum_x += attr_x['probs'][i] * attr_x['rewards'][i][0]

        for i in range(len(attr_y['rewards'])):
            sum_y += attr_y['probs'][i] * attr_y['rewards'][i][0]

        return abs(sum_x - sum_y)

    def get_action_distance(self, attr_x, attr_y):
        sum_x = 0
        sum_y = 0
        for i in range(len(attr_x['actions'])):
            for j in range(len(attr_x['actions'][i])):
                sum_x += attr_x['probs'][i] * attr_x['actions'][i][j]

        for i in range(len(attr_y['actions'])):
            for j in range(len(attr_y['actions'][i])):
                sum_y += attr_y['probs'][i] * attr_y['actions'][i][j]

        return abs(sum_x - sum_y)

    #   计算mdp中两个节点距离
    def distance(self, data1, data2):
        # 判断两个状态是否相同
        x = self.match_node(data1)
        y = self.match_node(data2)
        if x['state'] == y['state']:
            return 0

        # 状态本身之间的距离
        state_distance = np.linalg.norm(np.array(x['state']['state']) - np.array(y['state']['state']))

        #   如果有一个是终止节点
        if not x['next_states'] or not y['next_states']:
            return self.cs * state_distance

        #   动作区间交集奖励的最大差异
        max_reward_difference = self.get_reward_distance(x, y)  # 动作区间交集

        #   后继状态分布的差异
        distribution_difference = self.compute_distribution_difference(x['probs'], y['probs'])

        #   动作的最大差异
        max_action_difference = self.get_action_distance(x, y)

        return self.cs * state_distance + self.cr * max_reward_difference + self.cd * max_action_difference + self.cp * distribution_difference

    # 将聚类得到的中心点变成mdp模型中结点
    def revised_centers(self, center):
        """
        centers/centroids: list[list[float]]，精度已经保留好
        """
        centroids = []
        for item in center:
            node = self.match_node(item)
            centroids.append(node['state']['state'])
        return centroids

    def compute_inertia(self, datas):
        """
        datas: 原始数据 2D 数组，与 'initialize' 输入相同。
        """
        inertia = 0
        centroids = self.get_centers()
        centroids = np.array(self.revised_centers(centroids))
        clusters = self.get_clusters()

        # 矢量化计算距离
        distances = np.linalg.norm(datas[:, np.newaxis, :] - centroids[clusters], axis=2)

        # 使用矢量化操作计算惯性
        for k, item in enumerate(clusters):
            inertia += np.sum(distances[k, item])

        return inertia

    def pairwise_distance(self, datas):
        """
        datas: 输入数据 2D 数组。
        ret: 具有对称性质的距离矩阵，2D 数组 (n_sample, n_sample)。
        """
        # 矢量化计算距离
        distances = np.linalg.norm(datas[:, np.newaxis, :] - datas, axis=2)

        # 返回对称距离矩阵
        return distances

    # 可以使用矢量化操作优化其他方法

    def transform(self, datas):
        """
        datas: 输入数据 2D 数组。
        ret: 距离矩阵 2D 数组 (n_sample, n_center)，表示每个输入数据到中心点的距离。
        """
        centroids = self.get_centers()
        centroids = np.array(self.revised_centers(centroids))

        # 矢量化计算距离
        distances = np.linalg.norm(datas[:, np.newaxis, :] - centroids, axis=2)

        return distances


# 并行计算 pairwise_distance 方法
def parallel_pairwise_distance(args):
    i, datas, obj = args
    return obj.pairwise_distance(datas[i:i + 1, :])


if __name__ == '__main__':
    #   获得config
    path = "/Users/akihi/Downloads/coding?/Abstract-CTD3-main-master/conf/eval/highway_acc_eval.yaml"
    eval_config = utils.load_yml(path)

    #   获得图和状态
    graph, attr_dic = get_mdp_map("/Users/akihi/Downloads/coding?/Abstract-CTD3-main-master/data_analysis/mdp"
                                  "/first_stage_abstract/acc_datasets/first_abstract_pro_center_data_test.csv")
    graph_str_keys = {str(key): value for key, value in graph.items()}
    # 将字典保存到文件
    with open('kmeans_model/acc_td3_risk/spatio_temporal_graph_test.json', 'w') as file:
        json.dump(graph_str_keys, file, indent=2)

    attr_dic_str_keys = {str(key): value for key, value in attr_dic.items()}
    # 将字典保存到文件
    with open('kmeans_model/acc_td3_risk/spatio_temporal_attr_dic_test.json', 'w') as file:
        json.dump(attr_dic_str_keys, file, indent=2)

    visualize_mdp(graph, 'mdp_pic/acc_td3/spartio_3_Node.png')
    states = get_mdp_states(graph)
    kmeans_instance = SpatioTemporalKMeans(config=eval_config, datas=states, graph=attr_dic, cr=0.25, cd=0.25, cp=0.25,
                                           cs=0.25, k=3)
    #   进行聚类
    kmeans_instance.process()

    #   模型保存
    joblib.dump(kmeans_instance, 'kmeans_model/acc_td3_risk/spatio_temporal_Kmeans_test.pkl')
