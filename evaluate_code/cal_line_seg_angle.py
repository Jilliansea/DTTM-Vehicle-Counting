# coding=utf-8

"""
Time:2020-04-05 13:45
Author: XingTengfei
File:cal_line_seg_angle.py
Email:xingtengfei@didiglobal.com
Company: Computer Vision at DiDi Map
"""

import os
import sys
import json
import numpy as np
import pprint


#两条线段的夹角的计算
#向量思维

import math

#得到向量的坐标以及向量的模长
class Point(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def vector(self):
        c = (self.x1 - self.x2, self.y1 - self.y2)
        return c

    def length(self):
        d = math.sqrt(pow((self.x1 - self.x2), 2) + pow((self.y1 - self.y2), 2))
        return d

#计算向量夹角
class Calculate(object):
    def __init__(self, x, y, m, n):
        self.x = x
        self.y = y
        self.m = m
        self.n = n

    def Vector_multiplication(self):
        self.mu = np.dot(self.x, self.y)
        return self.mu

    def Vector_model(self):
        self.de = self.m * self.n
        return self.de

    def cal(self):
        result = Calculate.Vector_multiplication(self) / Calculate.Vector_model(self)
        if abs(result) > 1:
            # print(result)
            if result > 1.0:
                result = 1.0
            elif result < -1.0:
                result = -1.0
        angle = math.acos(result)*180/3.141
        # if not angle:
        #     print(angle)
        return angle


def line_seg_angle(seg1, seg2):
    first_point = Point(seg1[0][0], seg1[0][1], seg1[1][0], seg1[1][1])
    two_point = Point(seg2[0][0], seg2[0][1], seg2[1][0], seg2[1][1])
    if first_point.length()==0 or two_point.length() == 0:
        return 0
    ca = Calculate(first_point.vector(), two_point.vector(), first_point.length(), two_point.length())
    # print(ca.cal())
    return ca.cal()
