import matplotlib.pyplot as plt

import numpy as np
from random import uniform, seed, shuffle ,sample
import math
import logging

# from random import random

'''JY_Toolkit.py'''


class Jy_makeDataset(object):

    def random_state(random_seed):
        seed(int(random_seed))

    def draw_HalfMoon(n_sample: int = 1000,       # 样本点个数，两个分类一共 n_sample
                      w: float = 1,              # 半月的线宽
                      radius: float = 4,         # 半月的半径
                      hor_distance: float = 4,   # Horizontal direction distance for two point
                      ver_distance: float = 0,   # Vertical direction distance for two point
                      slope: float = 0,          # 半月倾斜的角度  [0 ~ 180]
                      positive_val: int = 1,
                      negative_val: int = -1,
                      ):

        slope %= 180            # make the `slope`  between 0 and 180
        # 将 n_sample 和样本分为两类每个样本 n_sample / 2 类
        each_m = n_sample//2
        # circle origin point of positive moon [x , y]
        p_origin = [1 + w/2 + radius, 1 + w/2 + radius + ver_distance]
        # circle origin point of negative moon [x , y]
        n_origin = [p_origin[0] + hor_distance, p_origin[1] - ver_distance]

        # product positive point
        p_sample = []
        n_sample = []
        for i in range(each_m):
            # Randomly generate l
            temp_l = radius + uniform(-(w/2), w/2)
            # Randomly generate angle i.e. theta
            temp_angle = uniform(slope, slope + 180)
            point_x = p_origin[0] + temp_l*math.cos(math.pi/180*temp_angle)
            point_y = p_origin[1] + temp_l*math.sin(math.pi/180*temp_angle)
            p_sample.append([point_x, point_y, positive_val])

        for i in range(each_m):
            # Randomly generate l
            temp_l = radius + uniform(-(w/2), w/2)
            # Randomly generate angle i.e. theta , but the angle of negative point should between `slope + 180` and `slope + 360`
            temp_angle = uniform(slope + 180, slope + 360)
            point_x = n_origin[0] + temp_l*math.cos(math.pi/180*temp_angle)
            point_y = n_origin[1] + temp_l*math.sin(math.pi/180*temp_angle)
            n_sample.append([point_x, point_y, negative_val])

        sample_points = p_sample + n_sample
        shuffle(sample_points)
        sample_points = np.array(sample_points)
        return sample_points[:, 0:2], sample_points[:, 2]

    pass


class Jy_dataSetProcess(object):

    def Jy_train_test_split(X,
                            y,
                            test_size : 0.2,
                            ):
        data = np.column_stack((X,y))
        if test_size >= 1 and test_size <= 0:
            logging.exception('test_size must be greater than 0 less than 1, we will assign test_size value of 0.2')
            test_size = 0.2

        sample_count = int(len(data)*test_size)

        '''
        分离思路：
        先将输入的数据集打乱，然后取前 test_size 部分为测试集，后部分为训练集
        '''
        shuffle(data)
        X_test = data[0:sample_count-1]
        X_train = data[sample_count:]

        return X_train[:,0:2],  X_test[:,0:2] ,X_train[:,2] , X_test[:,2]

    pass


if __name__ == '__main__':
    random_seed = 52

    Jy_makeDataset.random_state(random_seed)

    np_data, label = Jy_makeDataset.draw_HalfMoon(n_sample=2000)

    p_point_x1 = [np_data[i][0] for i in range(len(np_data)) if label[i] == 1]
    p_point_x2 = [np_data[i][1] for i in range(len(np_data)) if label[i] == 1]

    n_point_x1 = [np_data[i][0] for i in range(len(np_data)) if label[i] == -1]
    n_point_x2 = [np_data[i][1] for i in range(len(np_data)) if label[i] == -1]

    fig = plt.figure(num="HalfMoons", figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    ax1.scatter(p_point_x1, p_point_x2, c='red')
    ax1.scatter(n_point_x1, n_point_x2, c='blue')
    plt.show()

    print(np_data)
