import numpy as np
import re
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_intra(clt, data):
    total_dist = 0
    for i in clt:
        for j in clt:
            dist = (data[i] - data[j]) ** 2
            dist = dist.sum()
            dist = dist ** 0.5
            total_dist = total_dist + dist
    
    return total_dist / (len(clt) * len(clt))

def get_inter(clt1, clt2, data):
    total_dist = 0
    for i in clt1:
        for j in clt2:
            dist = (data[i] - data[j]) ** 2
            dist = dist.sum()
            dist = dist ** 0.5
            total_dist = total_dist + dist
    
    return total_dist / (len(clt1) * len(clt2))

def isExist(collection_list, point):#判断该集合中是否存在core point
    collection_included = []
    for i in range(0, len(collection_list)):
        collection = collection_list[i]
        if point in collection:
            collection_included.append(collection)

    if len(collection_included) == 0:
        return []
    else:
        return collection_included


def DBSCAN(data, radius, minPts):
    collection_list = []
    noise = []
    for i in range(0, len(data)):
        #计算距离点i不超过radius的点的个数
        point = data[i]
        collection = []
        dist = (point - data) ** 2
        dist = dist.sum(axis = 1)
        dist = dist ** 0.5
        for j in range(0, len(dist)):
            if dist[j] < radius:
                collection.append(j)
        
        #如果范围内点的个数超过minPts个，且集合里有core point，那么这些聚类都合并起来
        if len(collection) >= minPts:
            collection_included = isExist(collection_list, i)
            collection_merged = []
            remove_list = []
            if len(collection_included) != 0:
                for k in collection_included:
                    collection_merged.extend(k)
                    collection_list.remove(k)
                collection_merged.extend(collection)
                collection_merged = set(collection_merged)
                collection_list.append(collection_merged)
            else:#集合离没有core point，那么添加新的聚类
                collection_list.append(collection)

    result = []
    for i in collection_list:
        result.extend(i)
    for i in range(0, len(data)):
        if i not in result:
            noise.append(i)
    return collection_list, noise



if __name__ == "__main__":
    k = 5
    with open('clusterling\cluster_data.txt') as f:
        data = []
        while True:
            rows = f.readline()
            if not rows:
                break
            reList = rows.strip().split(' ')
            data.append(reList)
        data = np.array([[float(x) for x in row] for row in data])
    collection_list, noise = DBSCAN(data, 7, 15)
    # collection_list, noise = DBSCAN(data, 7, 25)

    model = TSNE(learning_rate=100)
    transformed = model.fit_transform(data)
    for collection in collection_list:
        result = []
        xs = []
        ys = []
        for i in collection:
            xs.append(transformed[i][0]) 
            ys.append(transformed[i][1])
        plt.scatter(xs,ys,c=[np.random.rand(3,)])

    xs = []
    ys = []
    for i in noise:
        xs.append(transformed[i][0]) 
        ys.append(transformed[i][1])
    plt.scatter(xs,ys,c='black')
    plt.title("DBSCAN")

    inter_class = []
    intra_class = []

    #可视化聚类结果
    for i in range(0, len(collection_list)):
        for j in range(i, len(collection_list)):
            if i == j:
                intra_class.append(get_intra(collection_list[j], data))
            else:
                inter_class.append(get_inter(collection_list[i], collection_list[j], data))
    print("intra = ", intra_class, np.mean(intra_class), np.var(intra_class), np.std(intra_class))
    print("inter = ", inter_class, np.mean(inter_class), np.var(inter_class), np.std(inter_class))
    plt.show()
    print(len(collection_list))
    print(len(noise))
