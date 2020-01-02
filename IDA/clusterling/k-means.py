import numpy as np
import re
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def naive_kmeans(k, data):
    center_index = random.sample(range(0, len(data)), k)
    cls_center = []
    cls_res = np.zeros(len(data), dtype='int')
    for i in center_index:
        cls_center.append(data[i])

    while True:
        is_changed = 0
        cls_count = np.zeros(k, dtype = float)
        cls_avg = np.zeros([k,7])
        for i in range(0, len(data)):
            #计算点i到其他点的所有距离，并把点i分类到距离最近的聚类当中
            point = data[i]
            dist = (point - cls_center) ** 2
            dist = dist.sum(axis = 1)
            dist = dist ** 0.5
            min = dist.argsort()[0]
            cls_avg[min] = cls_avg[min] + data[i]
            cls_res[i] = min
            cls_count[min] = cls_count[min] + 1
        
        for i in range(0, k):#获取新的中心点，并判断该聚类的中心点是否有变化
            avg = cls_avg[i] / cls_count[i]
            if not(np.array_equal(np.array(cls_center[i]), avg)):
                cls_center[i] = avg
                is_changed = 1
        
        if is_changed == 0:
            break

    return cls_center, cls_res 




def kmeans(k, data_idx, data, cls_res, center_idx, cluster_number):
    
    center_list = random.sample( data_idx, k)
    cls_center = []
    temp_cls_res = []
    temp_cls_res.extend(cls_res)

    for i in center_list:
        cls_center.append(data[i])

    while True:
        is_changed = 0
        cls_count = np.zeros(k, dtype = float)
        cls_avg = np.zeros([k,7])
        sse_idx1 = []
        sse_idx2 = []
        for i in data_idx:
            #计算点i到其他点的所有距离，并把点i分类到距离最近的聚类当中
            point = data[i]
            dist = (point - cls_center) ** 2
            dist = dist.sum(axis = 1)
            dist = dist ** 0.5
            min = dist.argsort()[0]
            cls_avg[min] = cls_avg[min] + data[i]
            if min == 0:
                temp_cls_res[i] = center_idx
            else:
                temp_cls_res[i] = cluster_number

            cls_count[min] = cls_count[min] + 1
        
        for i in range(0, k):
            avg = cls_avg[i] / cls_count[i] #获取新的中心点，并判断该聚类的中心点是否有变化
            if not(np.array_equal(np.array(cls_center[i]), avg)):
                cls_center[i] = avg
                is_changed = 1
        
        if is_changed == 0:
            break
    
    for i in data_idx:
        if temp_cls_res[i] == center_idx:
            sse_idx1.append(i)
        else:
            sse_idx2.append(i)
    sse1 = get_SSE(sse_idx1, data, cls_center[0])
    sse2 = get_SSE(sse_idx2, data, cls_center[1])
    return cls_center, temp_cls_res , [sse1, sse2]

def get_SSE(data_idx, data, center):
    part_data = []
    for i in data_idx:
        part_data.append(data[i])
    sse = (center - part_data) ** 2
    sse = sse.sum(axis = 1)
    sse = sse.sum(axis = 0)
    return sse

def bisecting(k, data):
    cls_center = []
    cls_SSE = np.zeros(k, dtype = 'float')
    cls_start = np.zeros(7, dtype = 'float')
    cls_res = np.zeros(len(data), dtype = 'int')
    cls_count = np.zeros(k, dtype = 'int')
    for i in data:
        cls_start = cls_start + i
    cls_start = cls_start / len(data)
    cls_center.append(cls_start)

    while len(cls_center) < k:#重复操作，直到得到k个聚类为止
        cls_count = np.zeros(k, dtype = float)
        cls_avg = np.zeros([k,7])

        if len(cls_center) == 1:
            for i in range(0, len(cls_center)):
                cmp_SSE = []
                data_idx = []
                for j in range(0, len(data)):
                    if cls_res[j] == i:
                        data_idx.append(j)
            center, temp_res, sse_list = kmeans(2, data_idx, data, cls_res, 0, 1)

            cls_center = center
            cls_res = temp_res
            cls_SSE[0] = sse_list[0]
            cls_SSE[1] = sse_list[1]
        else:
            temp_SSE = []
            temp_center = []
            cmp_SSE = []
            res = []
            for i in range(0, len(cls_center)):
                data_idx = []
                for j in range(0, len(data)):
                    if cls_res[j] == i:
                        data_idx.append(j)
            
                center, temp_res, sse_list = kmeans(2, data_idx, data, cls_res, i, len(cls_center))
                sse_list.append(cls_SSE[i])
                res.append(temp_res)
                temp_center.append(center)
                temp_SSE.append(sse_list)
            for i in temp_SSE:#对所有已有的聚类进行k=2的k-means计算之后，选取一个能降低SSE最多的一个计算结果
                cmp_SSE.append(i[2] - i[0] - i[1])
            min = np.array(cmp_SSE).argsort()[len(cmp_SSE) - 1]
            cls_center[min] = temp_center[0]
            cls_center.append(temp_center[1])
            cls_res = res[min]
            cls_SSE[min] = temp_SSE[min][0]
            cls_SSE[len(cls_center) - 1] = temp_SSE[min][1]
        # print("RES = ", count_result(cls_res))
    for i in range(0, len(data)):
        cls_count[cls_res[i]] = cls_count[cls_res[i]] + 1      
    
    return cls_res

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

def analyze_k(k, data):
    cls_res = bisecting(k, data)
    collection_list = []
    intra_class = []
    inter_class = []
    sse_list = []
    for i in range(0, k):
        result = []

        for j in range(0, len(data)):
            if cls_res[j] == i:
                result.append(j)
        collection_list.append(result)

    for i in range(0, len(collection_list)):
        sse_list.append(get_SSE(collection_list[i], data, get_center(collection_list[i], data)))
    
    print("SSE = ", np.mean(sse_list))
    return np.mean(sse_list)


def get_center(data_idx, data):
    center = np.zeros(7, dtype='float')
    for i in data_idx:
        center = center + data[i]

    return center / len(data_idx)

if __name__ == "__main__":
    k = 4
    with open('clusterling\cluster_data.txt') as f:
        data = []
        while True:
            rows = f.readline()
            if not rows:
                break
            reList = rows.strip().split(' ')
            data.append(reList)
        data = np.array([[float(x) for x in row] for row in data])
 
#######bisecting_kmeans
    cls_res = bisecting(k, data)
    cls_count = np.zeros(k, dtype = 'int')
    for i in range(0, len(data)):
        cls_count[cls_res[i]] = cls_count[cls_res[i]] + 1

    model = TSNE(learning_rate=100)
    transformed = model.fit_transform(data)

    collection_list = []
    intra_class = []
    inter_class = []
    sse_list = []
    for i in range(0, k):#可视化计算结果
        result = []
        xs = []
        ys = []
        for j in range(0, len(data)):
            if cls_res[j] == i:
                xs.append(transformed[j][0]) 
                ys.append(transformed[j][1])
                result.append(j)
        plt.title("bisecting_kmeans")
        plt.scatter(xs,ys,c=[np.random.rand(3,)])

        collection_list.append(result)
    plt.show()

    for i in range(0, len(collection_list)):
        for j in range(i, len(collection_list)):
            if i == j:
                intra_class.append(get_intra(collection_list[j], data))
                sse_list.append(get_SSE(collection_list[i], data, get_center(collection_list[i], data)))
            else:
                inter_class.append(get_inter(collection_list[i], collection_list[j], data))
    
    print("intra = ", intra_class, np.mean(intra_class), np.var(intra_class), np.std(intra_class))
    print("inter = ", inter_class, np.mean(inter_class), np.var(inter_class), np.std(inter_class))
    print("SSE = ", sse_list, np.mean(sse_list), np.var(sse_list), np.std(sse_list))

##########naive kmeans
    collection_list = []
    intra_class = []
    inter_class = []
    sse_list = []
    _ , cls_res = naive_kmeans(k, data)
    for i in range(0, k):
        result = []
        xs = []
        ys = []
        for j in range(0, len(data)):
            if cls_res[j] == i:
                xs.append(transformed[j][0]) 
                ys.append(transformed[j][1])
                result.append(j)
        plt.title("naive_kmeans")
        plt.scatter(xs,ys,c=[np.random.rand(3,)])
        collection_list.append(result)
    for i in range(0, len(collection_list)):
        for j in range(i, len(collection_list)):
            if i == j:
                intra_class.append(get_intra(collection_list[j], data))
                sse_list.append(get_SSE(collection_list[i], data, get_center(collection_list[i], data)))
            else:
                inter_class.append(get_inter(collection_list[i], collection_list[j], data))

    print("intra = ", intra_class, np.mean(intra_class), np.var(intra_class), np.std(intra_class))
    print("inter = ", inter_class, np.mean(inter_class), np.var(inter_class), np.std(inter_class))
    print("SSE = ", sse_list, np.mean(sse_list), np.var(sse_list), np.std(sse_list))
    plt.show()
