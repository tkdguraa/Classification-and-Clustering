import numpy as np
import csv
import os
import math
import time

def kNN(n, index, input, folded_data, label, k, cf_matrix):
    """
    n = 五块数据集中设哪一个数据集为测试集
    index = 该数据在数据集中的index
    input = 测试数据
    folded_data = 五块数据
    label = 标签
    k = kNN中 k 的大小
    """
    head = 0
    if n == 0:
        head = head + 1
    remain_data = folded_data[head]

    count_label = np.zeros((5,), dtype=float)
    for i in range(0, 5):
        if i != head and i != n:
            remain_data = np.concatenate((remain_data, folded_data[i]), axis = 0)
        
    dist = (input - remain_data) ** 2
    dist = dist.sum(axis = 1)
    dist = dist ** 0.5

    sorted_dist = dist.argsort()[:k]
    for i in sorted_dist:
        j = i

        if int(i / 2300) >= n:
            i = i + 2300
        count_label[int(label[i]) - 1] = count_label[int(label[i]) - 1] + float(1 / (1 + dist[j]))
    result = count_label.argsort()[4]
    cf_matrix[int(label[index]) - 1][result] = cf_matrix[int(label[index]) - 1][result] + 1
    return cf_matrix

def five_fold():
    cf_matrix = np.zeros((5,5), dtype = int)
    cf = np.zeros((5,), dtype = int) #[tp fp fn tn]
    for i in range(0, 11500):
        cf_matrix = kNN( int(i / 2300), i, data[i], folded_data, label, 5, cf_matrix)
    correct = 0
    for i in range(0, 5):
        correct = correct + cf_matrix[i][i]
    print("five_accuracy: ", float( correct / 11500 ) )
    fault = 0
    for i in range(1, 5):
        fault = fault + cf_matrix[i][0]
        fault = fault + cf_matrix[0][i]

    print("two_accuracy: ", float( (11500 - fault) / 11500 ) )
    print(cf_matrix)
    


if __name__ == "__main__":
    data = []
    label = []
    test = []
    cnt = -1
    with open('classify\classification_data.csv') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            del(row[0])
            if cnt != -1:
                label.append(row[len(row) -1])
                # label.append(row)
            cnt = cnt + 1
            del(row[len(row) - 1])
            data.append(row)
    del(data[0])
    data = np.array([[float(x) for x in row] for row in data])

    folded_data = np.split(np.array(data), 5)
    five_fold()


# print(data[3])
# print(label[3])
# kNN(0, 0, data[0], folded_data, label, 0)