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

        mid_origin_point = [p_origin[0]/2 + n_origin[0]/2, p_origin[1]/2 + n_origin[1]/2]

        p_origin[0] = mid_origin_point[0] + (hor_distance/2) * math.cos(math.pi/180*slope) 
        p_origin[1] = mid_origin_point[1] + (hor_distance/2) * math.sin(math.pi/180*slope) 

        p_origin[0] = p_origin[0] + (ver_distance/2) * math.cos(math.pi/180 * (slope + 90))
        p_origin[1] = p_origin[1] + (ver_distance/2) * math.sin(math.pi/180 * (slope + 90))

        n_origin[0] = mid_origin_point[0] + (hor_distance/2) * math.cos(math.pi/180*(slope + 180)) 
        n_origin[1] = mid_origin_point[1] + (hor_distance/2) * math.sin(math.pi/180*(slope + 180)) 

        n_origin[0] = n_origin[0] + (ver_distance/2) * math.cos(math.pi/180*(slope + 270)) 
        n_origin[1] = n_origin[1] + (ver_distance/2) * math.sin(math.pi/180*(slope + 270)) 

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

class Jy_dataShow(object):
    def __init__(self):
        pass
    def show_TwoDimData(X,y):
        y_type_array = []
        for sample in y:
            if sample not in y_type_array:
                y_type_array.append(sample)
        y_type_count = len(y_type_array)
        # all_X = [[],[],[]]
        all_X = []
        for n in range(y_type_count):
            temp_point_x1 = [X[i][0] for i in range(len(X)) if y[i] == y_type_array[n]]
            temp_point_x2 = [X[i][1] for i in range(len(X)) if y[i] == y_type_array[n]]
            all_X.append([temp_point_x1.copy(),temp_point_x2.copy()])
        fig = plt.figure(num = 'DataShow',figsize=(8,8))
        plot_1 = fig.add_subplot(111)
        plot_1.set_xlabel('x1')
        plot_1.set_ylabel('x2')
        plot_1.set_title('DataShow')
        for i in range(y_type_count):
            plot_1.scatter(all_X[i][0],all_X[i][1],label=y_type_array[i])
        # plot_1.scatter(p_point_x1, p_point_x2, c='red')
        # plot_1.scatter(n_point_x1, n_point_x2, c='blue')
        plot_1.legend(loc=3)
        plt.show()
    pass

class Jy_dataSetProcess(object):

    def Jy_train_test_split(X,
                            y,
                            *
                            ,
                            test_size = 0.2,
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

    np_data, label = Jy_makeDataset.draw_HalfMoon(n_sample=1000,slope= 30,ver_distance=-2)
    Jy_dataShow.show_TwoDimData(np_data, label)

