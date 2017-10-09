# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt

import kmeans

class testmain(object):
    """docstring for main"""
    def __init__(self):
        df = pd.read_csv('iris.data.csv', names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
        self.df = df.iloc[:, 2:5]
    
    def main(self):
        #load petal length(3rd col), petal width(4th col)
        x = self.df.iloc[:, 0:2].values
        alg = kmeans.Kmeans(x, 3)
        alg.do_kmeans()

    def draw_lilis(self, df):
        x = df
        plt.xlabel('petal width')
        plt.ylabel('petal length')
        plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='x', label='Setosa')
        plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='o', label='Versicolour')
        plt.scatter(x[100:150, 0], x[100:150, 1], color='green', marker='v', label='Virginica')
        plt.legend(loc='upper left')
        plt.show()

if __name__ == '__main__':
    test = testmain()
    test.main()