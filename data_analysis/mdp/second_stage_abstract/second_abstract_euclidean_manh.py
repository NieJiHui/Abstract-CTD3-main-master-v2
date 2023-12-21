import csv

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib
import findOptimalK as fok


def get_second_stage_mdp(data, kmeans_model, output_file):
    # 计算概率
    transition_probabilities = {}
    for row in data.itertuples(index=False):
        state = np.array(row.State).reshape(1, 2)
        next_state = np.array(row.NextState).reshape(1, 2)
        action = row.Act
        reward = row.Reward
        done = row.Done
        cost = row.Cost
        state_id_np = kmeans_model.predict(state)
        state_id = state_id_np[0]
        next_state_id_np = kmeans_model.predict(next_state)
        next_state_id = next_state_id_np[0]

        weight = row.Weight

        # 更新或添加概率
        if state_id in transition_probabilities:
            if next_state_id in transition_probabilities[state_id]:
                transition_probabilities[state_id][next_state_id]['Weight'] += weight
                transition_probabilities[state_id][next_state_id]['action'] = \
                    [x + y for x, y in zip(transition_probabilities[state_id][next_state_id]['action'], action)]
                transition_probabilities[state_id][next_state_id]['reward'] = \
                    [x + y for x, y in zip(transition_probabilities[state_id][next_state_id]['reward'], reward)]
                transition_probabilities[state_id][next_state_id]['cost'] = \
                    [x + y for x, y in zip(transition_probabilities[state_id][next_state_id]['cost'], cost)]
                transition_probabilities[state_id][next_state_id]['count'] += 1
            else:
                transition_probabilities[state_id][next_state_id] = {
                    'Weight': weight,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'cost': cost,
                    'count': 1
                }
        else:
            transition_probabilities[state_id] = {next_state_id: {'Weight': weight,
                                                                  'action': action,
                                                                  'reward': reward,
                                                                  'done': done,
                                                                  'cost': cost,
                                                                  'count': 1}}

    cluster_centers = kmeans_model.cluster_centers_
    # 归一化权重，计算概率
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['State_ID', 'Next_State_ID', 'Probability', 'Action', 'Reward', 'Cost', 'State_Center_Data', 'Next_State_Center_Data']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入表头
        csv_writer.writeheader()

        # 遍历每一行并写入 CSV 文件
        for state_id, successors in transition_probabilities.items():
            for next_state_id, weight_info in successors.items():
                weight = weight_info['Weight']
                total_weight = sum(weight_info['Weight'] for weight_info in successors.values())
                probability = weight / total_weight
                count = weight_info['count']
                act = [x / count for x in weight_info['action']]
                reward = [x / count for x in weight_info['reward']]
                cost = [x / count for x in weight_info['cost']]

                # 将数据写入 CSV 文件
                csv_writer.writerow({
                    'State_ID': state_id,
                    'Next_State_ID': next_state_id,
                    'Probability': probability,
                    'Action': act,
                    'Reward': reward,
                    'Cost': cost,
                    'State_Center_Data': cluster_centers[[state_id]].tolist()[0],
                    'Next_State_Center_Data': cluster_centers[[next_state_id]].tolist()[0]
                })

    return transition_probabilities



def perform_kmeans(data, optimal_k):
    # 曼哈顿距离
    # kmeans = KMeans(n_clusters=optimal_k, algorithm='full')
    # 欧式距离（即L2范数）
    kmeans = KMeans(n_clusters=optimal_k)
    kmeans.fit(data)
    return kmeans


if __name__ == '__main__':
    # Load the dataset
    file_path = 'first_abstract_pro_raw_data_test.csv'
    data = pd.read_csv(file_path)

    # Convert string representations of lists into actual lists
    data['State'] = data['State'].apply(eval)
    data['NextState'] = data['NextState'].apply(eval)
    data['Act'] = data['Act'].apply(eval)
    data['Reward'] = data['Reward'].apply(eval)
    data['Cost'] = data['Cost'].apply(eval)
    # Combine 'State' and 'NextState' into a single list and remove duplicates
    combined_states = data['State'].tolist() + data['NextState'].tolist()
    combined_states = [list(x) for x in set(tuple(x) for x in combined_states)]

    # Find optimal k using elbow method
    # optimal_k = fok.find_optimal_k_elbow(combined_states)

    # # Perform KMeans clustering with the optimal number of clusters
    # kmeans_model = perform_kmeans(combined_states, 3)
    #
    # # Display the cluster centers
    # cluster_centers = kmeans_model.cluster_centers_
    # print("Cluster Centers:")
    # print(cluster_centers)
    #
    # # Save the KMeans model
    model_save_path = 'kmeans_model/Euclidean_kmeans_model.pkl'
    # joblib.dump(kmeans_model, model_save_path)
    # print(f"KMeans model saved at: {model_save_path}")

    loaded_kmeans_model = joblib.load(model_save_path)

    get_second_stage_mdp(data, loaded_kmeans_model, 'test.csv')
    # # Prepare new data
    # new_data = np.array([[1.0, 2.0]])
    # print(new_data.shape)
    #
    # # Use KMeans model for prediction
    # # Load the KMeans model
    # loaded_kmeans_model = joblib.load(model_save_path)
    #
    # # Use the loaded KMeans model for prediction
    # loaded_predictions = loaded_kmeans_model.predict(new_data)
    # print("Loaded Predictions:", loaded_predictions)
    #
    # # Get the cluster center for each data point
    # predicted_centers = cluster_centers[loaded_predictions]
    # print("Predicted Centers:")
    # print(predicted_centers)
