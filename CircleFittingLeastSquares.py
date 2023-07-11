# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:26:07 2023

@author: Bestvision
"""

import numpy as np

def find_circle_center(points):
    # 转换点的数据类型为float
    points = np.array(points, dtype='float')

    # 构造A, B矩阵
    A = np.hstack((-2 * points, np.ones((points.shape[0], 1))))
    B = (points[:, 0]**2 + points[:, 1]**2).reshape(-1, 1)

    # 使用numpy的最小二乘法求解
    res = np.linalg.lstsq(A, B, rcond=None)[0].flatten()

    return (res[0], res[1])

# 测试
points = [
          (-3972.8547, 474.1851), 
          (-3947.765, 478.1046), 
          (-3922.5412, 481.0396),
          (-3897.2218,482.9856),
          (-3871.8457,483.9396),
          (-3647.2994,448.1457),
          (-3623.4776,439.4493)          
          ]
center = find_circle_center(points)
print("The center of the circle is: ", center)
