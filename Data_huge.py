#电动汽车模型数据
import json
import random
import time
import find_nearest_neighbors as fnn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

battery_consumption=1.30# kwh/1km
# 配送站数量和客户数量
num_stations = int(3)#配送站数量
num_customers = int(30)#工地客户数量
station_powor=250#所有厂内部充电功率
station_powor_km=station_powor*battery_consumption#所有厂内部充电功率，里程
station_powor_out=600#所有厂外部充电功率
station_powor_out_km=station_powor_out*battery_consumption#所有厂外部充电功率，里程
kache_zaizhong=36000 #kg
battery_capacity=1000#kwh
battery_yuzhi=0.3#电池电量到达某一节点后，如果低于这个阈值，需要寻找充电桩进行充电
kache_xuhang=800#km
goujian_w=1100#kg
punish_1=10
punish_2=10
bat_statiom_n = [31, 32, 33]
chongdian_jiage1=4.5#站外充电价格
chongdian_jiage2=1.2#站内充电价格
speed=80#km平均时速
# driver_num=6#可同时出动的最大车辆数

# 配送站坐标
stations ={
    i+31: {
        'location': station['location'],
        'carNum': station['carNum'],
    }
    for i, station in enumerate([
        {'location': (96, 21),'carNum': 9},
        {'location': (28, 28),'carNum': 10},
        {'location': (37, 87),'carNum': 8},
    ])
}
# print(stations)

# 客户坐标
customers = [
    (33, 42),
    (97, 18),
    (25, 34),
    (82, 16),
    (88, 66),
    (8, 19),
    (33, 78),
    (61, 34),
    (84, 2),
    (54, 43),
    (99, 31),
    (58, 45),
    (69, 29),
    (94, 88),
    (30, 38),
    (18, 5),
    (45, 37),
    (40, 84),
    (99, 47),
    (11, 11),
    (3, 14),
    (86, 5),
    (72, 72),
    (58, 69),
    (32, 41),
    (17, 94),
    (12, 45),
    (59, 51),
    (77, 12),
    (86, 97)
]



# 充电站数据
charging_stations = {
    i+1: {
        'location': station['location'],
        'slotsNum': station['slotsNum'],
        'powerRate': station['powerRate']
    }
    for i, station in enumerate([
        {'location': (84, 66), 'slotsNum': 6, 'powerRate': 60},
        {'location': (54, 30), 'slotsNum': 7, 'powerRate': 60},
        {'location': (57, 46), 'slotsNum': 6, 'powerRate': 30},
        {'location': (82, 39), 'slotsNum': 10, 'powerRate': 7},
        {'location': (10, 96), 'slotsNum': 7, 'powerRate': 60}
    ])
}

# 卡车数据
trucks = {
    i+1: {
        'capacity': truck['capacity'],
        'battery_capacity': truck['battery_capacity']
    }
    for i, truck in enumerate([
        {'capacity': 14911, 'battery_capacity': 746},
        {'capacity': 15194, 'battery_capacity': 370},
        {'capacity': 14494, 'battery_capacity': 950},
        {'capacity': 13077, 'battery_capacity': 449},
        {'capacity': 19132, 'battery_capacity': 978},
        {'capacity': 17612, 'battery_capacity': 352},
        {'capacity': 14217, 'battery_capacity': 844},
        {'capacity': 12183, 'battery_capacity': 583},
        {'capacity': 19346, 'battery_capacity': 389},
        {'capacity': 7841, 'battery_capacity': 926}
    ])
}

# 构件数据
components = {
    i+1: {
        'weight': component['weight']
    }
    for i, component in enumerate([
        {'weight': 652},
        {'weight': 796},
        {'weight': 896},
        {'weight': 782},
        {'weight': 859},
        {'weight': 287},
        {'weight': 430},
        {'weight': 135},
        {'weight': 719},
        {'weight': 911},
        {'weight': 197},
        {'weight': 746},
        {'weight': 343},
        {'weight': 309},
        {'weight': 932},
        {'weight': 213},
        {'weight': 468},
        {'weight': 916},
        {'weight': 437},
        {'weight': 837},
        {'weight': 730},
        {'weight': 122},
        {'weight': 206},
        {'weight': 482},
        {'weight': 395},
        {'weight': 376},
        {'weight': 876},
        {'weight': 305},
        {'weight': 954},
        {'weight': 951}
    ])
}

# 客户需求数据
# 客户需求数据，包含坐标
customer_demands = {
    i+39: {
        # 'location': customers[i],
        'demand': demand['demand'],
        'time_window': demand['time_window']
    }
    for i, demand in enumerate([
        {'demand': 300, 'time_window': (12, 18)},
        {'demand': 500, 'time_window': (9, 14)},
        {'demand': 400, 'time_window': (10, 13)},
        {'demand': 300, 'time_window': (9, 16)},
        {'demand': 500, 'time_window': (8, 15)},
        {'demand': 800, 'time_window': (11, 16)},
        {'demand': 400, 'time_window': (8, 16)},
        {'demand': 100, 'time_window': (10, 17)},
        {'demand': 100, 'time_window': (8, 15)},
        {'demand': 700, 'time_window': (12, 18)},
        {'demand': 700, 'time_window': (8, 18)},
        {'demand': 900, 'time_window': (9, 15)},
        {'demand': 200, 'time_window': (8, 16)},
        {'demand': 700, 'time_window': (9, 16)},
        {'demand': 200, 'time_window': (8, 18)},
        {'demand': 300, 'time_window': (11, 18)},
        {'demand': 900, 'time_window': (11, 15)},
        {'demand': 700, 'time_window': (10, 18)},
        {'demand': 100, 'time_window': (9, 14)},
        {'demand': 200, 'time_window': (12, 18)},
        {'demand': 500, 'time_window': (11, 17)},
        {'demand': 800, 'time_window': (8, 16)},
        {'demand': 400, 'time_window': (10, 13)},
        {'demand': 400, 'time_window': (12, 18)},
        {'demand': 100, 'time_window': (12, 13)},
        {'demand': 100, 'time_window': (9, 16)},
        {'demand': 200, 'time_window': (9, 14)},
        {'demand': 800, 'time_window': (9, 17)},
        {'demand': 700, 'time_window': (8, 15)},
        {'demand': 200, 'time_window': (10, 13)}
    ])
}
bat_station_n = [34, 35, 36, 37, 38]#充电站编号
bigen_station_n=[31, 32, 33]#编号
road_point_n=[i+1 for i in range(30)]
client_point_n=[i+39 for i in range(30)]
# print(client_point_n)

# 定义读取 JSON 文件的函数
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 读取 JSON 数据并转换为 Python 数据结构
    return data


# 定义转换为结构数据的函数
def convert_to_structured_data(data):
    structured_data = []
    for item in data:
        node = {
            "id": item["id"],
            "x": item["location"][0],
            "y": item["location"][1],
            "node_class": item["classification"],
            "elements": item["elements"]
        }
        structured_data.append(node)
    return structured_data
json_file_path = 'coordinates.json'  # 替换为你的 JSON 文件路径
json_data = read_json_file(json_file_path)  # 读取 JSON 文件
structured_nodes = convert_to_structured_data(json_data)  # 转换为结构数据
locations = [{'location': (node['x'], node['y'])} for node in structured_nodes]
# 节点数据,1到30为路网节点，31号到33号为配送站，34号到38号为充电站，39-68的为客户。
road_network_nodes = {
    i+1: {
        'location': node['location']
    }
    for i, node in enumerate(locations)
}
# print(road_network_nodes)
# 计算两个点之间的欧氏距离
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_distance(point1, point2):
    """计算两点之间的欧氏距离"""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def create_adjacency_matrix(connections, num_nodes):
    """
    生成连通性矩阵（索引从1开始）
    :param connections: 字典，键为节点编号（从1开始），值为与该节点联通的节点编号列表（从1开始）
    :param num_nodes: 节点总数
    :return: 连通性矩阵（邻接矩阵）
    """
    # 初始化矩阵为全0
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for node, connected_nodes in connections.items():
        for connected_node in connected_nodes:
            adjacency_matrix[node - 1][connected_node - 1] = 1  # 将节点编号减1适配矩阵索引
            adjacency_matrix[connected_node - 1][node - 1] = 1  # 对称

    return adjacency_matrix
def find_nearest_neighbors(nodes, k=4):
    """
    找到每个节点最近的k个节点
    :param nodes: 字典，包含节点编号和位置信息
    :param k: 最近邻节点数量
    :return: 字典，每个节点对应k个最近邻节点
    """
    connections = {}

    for node_id, node_data in nodes.items():
        distances = []

        # 计算当前节点到其他节点的距离
        for other_id, other_data in nodes.items():
            if node_id != other_id:
                dist = calculate_distance(node_data['location'], other_data['location'])
                distances.append((other_id, dist))

        # 按距离排序并取前k个节点
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = [neighbor_id for neighbor_id, _ in distances[:k]]

        # 存储到connections字典中
        connections[node_id] = nearest_neighbors

    return connections

# 定义连通性阈值（例如50）
connectivity_threshold = 20

# 初始化连通性矩阵
connections =find_nearest_neighbors(road_network_nodes, k=4)
adj_matrix = create_adjacency_matrix(connections, len(connections))
connectivity_matrix=adj_matrix
#打印连通性矩阵
# print(connectivity_matrix)
# dataname2 = "connectivity_matrix_" + str(time.time()) + ".csv"
# df2 = pd.DataFrame(connectivity_matrix)
# df2.to_csv(dataname2, index=False, header=False)
# print("customer_demands",customer_demands)
# print("components",components)
# print("trucks",trucks)
# print("charging_stations",charging_stations)
# print("stations",stations)
if __name__ == '__main__':
    # 创建图形
    plt.figure(figsize=(10, 10))

    # 绘制每个节点
    for node_id, node_info in road_network_nodes.items():
        x, y = node_info['location']

        # 路网节点（1-30）
        if 1 <= node_id <= 30:
            plt.scatter(x, y, color='blue', label='Road Network' if node_id == 1 else "")
            plt.text(x, y, str(node_id), color='black', fontsize=8)

        # 配送站（31-33）
        elif 31 <= node_id <= 33:
            plt.scatter(x, y, color='green', label='Distribution Station' if node_id == 31 else "")
            plt.text(x, y, str(node_id), color='black', fontsize=8)

        # 充电站（34-38）
        elif 34 <= node_id <= 38:
            plt.scatter(x, y, color='red', label='Charging Station' if node_id == 34 else "")
            plt.text(x, y, str(node_id), color='black', fontsize=8)

        # 客户（39-68）
        elif 39 <= node_id <= 68:
            plt.scatter(x, y, color='purple', label='Customer' if node_id == 39 else "")
            plt.text(x, y, str(node_id), color='black', fontsize=8)

    # 添加图例
    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Node Map with Different Categories")

    # 显示图形
    plt.grid(True)
    plt.show()


