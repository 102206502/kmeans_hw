# -*- coding: utf-8 -*-
import scipy.spatial.distance as dist
import numpy as np
import matplotlib.pyplot as plt
import random

#(使用Mahalanobis distance計算距離)
#step 0. scale the coordinate
#step 1. 選取k個點
#step 2. 歸類選取點的附屬點
#step 3. 計算邊界中的點集合的中心
#step 4. 更新選取點
#step 5. 回到step 2


class Kmeans(object):
    """
        parameter
        ----------
        data : the points to be cluster
        k_num : how many part
    """
    def __init__(self, data, k_num):
        self.points = self.data2points(data)
        self.k_num = k_num
        self.boundary = self.find_boundary()
        # print('points', self.points)
        #step 1
        self.k_points = self.select_ponits()
        # print('k_points:', self.k_points)
        #step 2
        self.co_vec_inv = np.linalg.inv(np.cov(self.points.T))
        self.do_kmeans()



    def data2points(self, data):
        # X = np.vstack([data[:, 0], data[:, 1]])
        # print('vstack:\n', X)
        temp_list = []
        for xy in data:
            point = [xy[0], xy[1]]
            temp_list.append(point)

        return np.array(temp_list)

    '''找出資料邊界'''
    def find_boundary(self):
        x_max, y_max = self.points.max(axis=0)
        # print('x_max, y_max:', x_max, y_max)
        x_min, y_min = self.points.min(axis=0)
        # print('x_min, y_min:', x_min, y_min)
        alist = [x_max, y_max, x_min, y_min]
        return alist

    '''選取k個點'''
    def select_ponits(self):
        x1 = self.boundary[2]
        x2 = self.boundary[0]
        y1 = self.boundary[3]
        y2 = self.boundary[1]

        x_k_points = gen_uniq_floats(x1, x2, self.k_num)
        y_k_points = gen_uniq_floats(y1, y2, self.k_num)
        k_points = []
        for i in range(self.k_num):
            temp_list = [x_k_points[i], y_k_points[i]]
            k_points.append(temp_list)
        return np.array(k_points)

    '''do k-means'''
    def do_kmeans(self):
        cluster_assment = np.zeros(shape=(len(self.points), 2))
        samples = self.points
        centroids = self.k_points
        cluster_changed = True

        while True:
            # cluster_changed = False
            for i_p in range(len(self.points)):
                min_dist = 10000.0

                min_idx = 0
                #step 2 : find centroid of samples
                for i_c in range(self.k_num):
                    distance = dist.mahalanobis(centroids[i_c,:], samples[i_p,:], self.co_vec_inv)
                    if distance < min_dist:
                        min_dist = distance
                        min_idx = i_c
                    #update it cluster
                    if cluster_assment[i_p, 0] != min_idx:
                        # cluster_changed = True
                        cluster_assment[i_p,:] = [min_idx, min_dist]

            #step 3 : update centroid
            centroids, cluster_changed = self.update_centroids(cluster_assment, centroids)
            if not centroid_changed:
                break

        print('cluster complete!')
        return centroids, cluster_assment

    def update_centroids(self, cluster_assment, centroids):
        samples = self.points
        centroid_changed = False
            points_in_cluster_list = []
            for i_c in range(self.k_num):
                points_in_cluster = samples[cluster_assment[:, 0] == i_c, :]
                points_in_cluster_list.append(points_in_cluster)

            for i_c in range(self.k_num):
                if len(points_in_cluster) == 0:
                    self.spilt_cluster()
                point_mean = np.mean(points_in_cluster, axis=0)
                if point_mean[0]!=centroids[i_c, 0] and point_mean[1]!=centroids[i_c, 1]:
                    cluster_changed = True
                    centroids[i_c, :] = np.mean(points_in_cluster, axis=0)
                # self.show_centoids(centroids)
                # print('centroids after\n', centroids)
            # print(points_in_cluster_list)
            print('cluster_assment:\n', cluster_assment)
            self.show_cluster(centroids, points_in_cluster_list)
        return centroids, centroid_changed
    '''選取下一個中心點的群組，並分為兩群

        psuedo code
        -----------
        pick next group of cluster
        split the group by x(or y)
        take the mean of the two groups as new centoids

    '''
    def spilt_cluster(self):
        pass

    def show_centoids(self, centroids):
        plt.xlabel('petal width')
        plt.ylabel('petal length')
        plt.scatter(self.points[:,0], self.points[:, 1], color='green', marker='x')
        plt.scatter(centroids[:,0], centroids[:, 1], color='blue', marker='o')
        plt.show();

    def show_cluster(self, centroids, cluster_p):
        if self.k_num > 5:
            print('k_num too large!')
            return
        marker_list = ['x', 'v', '*', 's', 'p']
        plt.xlabel('petal width')
        plt.ylabel('petal length')
        plt.scatter(centroids[:,0], centroids[:, 1], color='blue', marker='o')
        for i in range(2):
            plt.scatter(cluster_p[i][:, 0], cluster_p[i][:, 1], color='green', marker=marker_list[i])
            print('cluster_p', i, ':', cluster_p[i])
            print(cluster_p[i].shape)
        # plt.figure(num='k = '+str(self.k_num))
        # fig = plt.gcf()
        # fig.canvas.set_window_title('k = '+str(self.k_num))
        plt.show()



def gen_uniq_floats(lo, hi, n):
    out = np.empty(n)
    needed = n
    while needed != 0:
        arr = np.random.uniform(lo, hi, needed)
        uniqs = np.setdiff1d(np.unique(arr), out[:n-needed])
        out[n-needed: n-needed+uniqs.size] = uniqs
        needed -= uniqs.size
    np.random.shuffle(out)
    return out.tolist()