import csv

# def get_state(csv_path):
#     combined_data = []
#
#     with open(csv_path, 'r') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             rel_dis = float(row['rel_dis'])
#             rel_speed = float(row['rel_speed'])
#             next_rel_dis = float(row['next_rel_dis'])
#             next_rel_speed = float(row['next_rel_speed'])
#
#             combined_data.append([rel_dis, rel_speed])
#             combined_data.append([next_rel_dis, next_rel_speed])
#
#     return combined_data

import csv
import re

import joblib
import numpy as np


def get_csv_info(csv_path, frequency, *columns):
    """
        读取CSV文件，按照指定的频率和列提取数据。

        Parameters:
        - csv_path (str): CSV文件的路径。
        - frequency (int): 读取数据的频率。
        - *columns (str): 要提取的列的名称。

        Returns:
        - List[List[float]]: 提取的数据列表。
    """
    combined_data = []
    counter = 0  # 计数器变量，用于控制每隔 freq 读取一次数据
    line_number = 0  # 当前行号

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            line_number += 1  # 增加行号计数
            #   print(row)
            try:
                if counter % frequency == 0:  # 判断是否是需要读取的行
                    data = []
                    for column in columns:
                        value = row[column]
                        try:
                            # 尝试将值转换为浮点数
                            value = float(value)
                        except ValueError:
                            # 转换失败时保留为字符串
                            pass
                        data.append(value)
                    combined_data.append(data)
                counter += 1  # 计数器增加1
            except KeyError as e:
                print(f"KeyError at line {line_number}: {e}")
    return combined_data


def get_designated_data(row, header, target_fields):
    """
    从CSV行数据中获取指定列的数据。

    Parameters:
    - row (dict): CSV行数据。
    - header (List[str]): CSV文件的表头。
    - target_fields (List[str]): 需要获取的列的名称。

    Returns:
    - List[float]: 获取的数据列表。
    """
    field_data = []

    # Check if the row is a dictionary (DictReader) and not an ordered tuple (standard Reader)
    if isinstance(row, dict):
        for field in target_fields:
            field_data.append(row.get(field, None))
    else:
        # Assuming the order of target fields matches the order of columns in the CSV
        for index in target_fields:
            field_data.append(str2float(row[header.index(index)]))

    return field_data


def str2float(value):
    """
        将字符串转换为浮点数。

        Parameters:
        - value (str): 要转换的字符串。

        Returns:
        - float: 转换后的浮点数。
    """
    try:
        float_value = float(value)
        return float_value
    except ValueError:
        # 如果转换为浮点数失败，则尝试转换为布尔值
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            raise ValueError("Cannot convert to float or bool")


def eliminate_similar(state, *thresholds):
    """
    通过比较每个维度，消除相似的状态。

    Parameters:
    - state (List[List[float]]): 状态值的列表。
    - *thresholds (float): 状态下每个维度的粒度阈值，小于这个阈值则认为相似。

    Returns:
    - List[List[float]]: 消除相似状态后的状态列表。
    """
    simplified_lst = []

    for sub_lst in state:
        if len(simplified_lst) > 0:
            diff = False
            for i in range(len(sub_lst)):
                if abs(sub_lst[i] - simplified_lst[-1][i]) >= thresholds[i]:
                    diff = True
                    break
            if not diff:
                simplified_lst[-1] = [(sub_lst[i] + simplified_lst[-1][i]) / 2 for i in range(len(sub_lst))]
            else:
                simplified_lst.append(sub_lst)
        else:
            simplified_lst.append(sub_lst)

    return simplified_lst


def real_state_2_abstract_state(model, state):
    """
        将真实状态映射到抽象状态，使用预训练的模型。

        Parameters:
        - model (object): 预训练的聚类模型。
        - state (List[List[float]]): 真实状态的列表。

        Returns:
        - Tuple[np.ndarray, np.ndarray]: 预测的标签和对应的聚类中心。
    """
    # 预测新数据点的簇
    new_data = np.array(state)
    predicted_label = model.predict(new_data)
    # 获取聚类中心
    cluster_center = model.cluster_centers_[predicted_label]
    return predicted_label, cluster_center

def log2csv(log_file, csv_file):
    """
        从日志文件中提取数据并写入CSV文件。

        Parameters:
        - log_file (str): 日志文件的路径。
        - csv_file (str): 要写入的CSV文件的路径。

        Returns:
        - None
    """
    def is_valid_string(input_string):
        pattern = r'^[-0-9 .]+$'
        return re.match(pattern, input_string) is not None

    with open(log_file, 'r',encoding='utf8') as file:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE, escapechar=',')

            # 写入CSV文件的标题行
            headline = file.readline()
            headline = headline.strip('\n').split(' ')
            writer.writerow(headline)
            # 逐行读取日志文件并写入CSV文件
            for line in file:
                # 解析日志文件中的数据
                data = line.strip().replace('[', '').replace(']', '').split(', ')
                if len(data) >= 2 and is_valid_string(data[1]):
                    write_line = []
                    for item in data:
                        #   中间有空格的是多维数据
                        if ' ' in item:
                            ls = item.split(' ')
                            ls = [it for it in ls if it]
                            write_line = write_line + ls
                        else:
                            write_line.append(item)
                    #   print(write_line)
                    writer.writerow(write_line)


def output2csv(data, csv_file):
    header1 = ["tran", "attr", "num", "pro"]
    with open(csv_file, 'w', encoding='utf-8', newline='') as file:
        Writer = csv.writer(file, header1)
        Writer.writerow(header1)
        Writer.writerows(data)


if __name__ == '__main__':
    # loaded_model = joblib.load('./acc_td3/kmeans_model.pkl')
    # state = []
    # state = get_csv_info('./acc_td3/test.csv', 1, 'rel_dis', 'rel_speed', 'cost')
    # print(state)
    # states = get_csv_info('./acc_td3/td3_risk_acc_logs.csv', 1, 'rel_dis', 'rel_speed', 'acc', 'reward', 'next_rel_dis',
    #                      'next_rel_speed', 'cost')
    # states = get_csv_info('./acc_td3/td3_risk_acc_logs.csv', 1, 'rel_dis', 'rel_speed')
    # # 表头
    # # header = ['rel_dis', 'rel_speed', 'acc', 'reward', 'next_rel_dis', 'next_rel_speed', 'cost']
    # header = ['rel_dis', 'rel_speed']
    # filename = 'gra_10.csv'
    # # 使用CSV模块写入CSV文件
    # with open(filename, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(header)  # 写入表头
    #     writer.writerows(states)
    # print(len(state))
    #
    # state = eliminate_similar(state, 0.0009, 0.0009)
    # print(len(state))
    file_path = 'gra_10.csv'
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)

        # Assuming the first row is the header
        header = next(csv_reader)

        # Example: Get 'Name' and 'Age' from the first row
        row_data = next(csv_reader)
        result = get_designated_data(row_data, header, ['rel_speed'])

        print(result)



