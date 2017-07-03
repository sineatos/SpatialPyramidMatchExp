# -*- encoding:UTF-8 -*-
"""
该模块包含了空间金字塔的实现
"""
import cv2
import numpy as np


class SpatialPyramidMatch:
    """
    标准的空间金字塔匹配
    """

    # sift描述子对应的一些关键字
    DESCRIPTORS = "DESCRIPTORS"
    X = "X"
    Y = "Y"
    WIDTH = "WIDTH"
    HEIGHT = "HEIGHT"

    def __init__(self, train_set, pyramid_level=3):
        """
        初始化方法
        :param train_set: 图片训练集，需要是一个可迭代对象，每一个元素都是一个像素矩阵
        :param pyramid_level: 空间金字塔的层数，默认为3层
        """
        self._train_set = train_set
        self._pyramid_level = pyramid_level

    @staticmethod
    def calculate_sift(images):
        """
        计算每一张图片的sift特征值
        :param images 一个图片集合的可迭代对象
        :return: 一个列表
                [
                    {
                        SpatialPyramidMatch.DESCRIPTORS: 当前图片的各个关键点的sift特征,
                        SpatialPyramidMatch.X: 当前图片的各个关键点的x,
                        SpatialPyramidMatch.Y: 当前图片的各个关键点的y,
                        SpatialPyramidMatch.HEIGHT: 当前图片的高,
                        SpatialPyramidMatch.WIDTH: 当前图片的宽,
                    },
                    ...
                ]
        """
        answer = []
        for image in images:
            sift_detector = cv2.xfeatures2d.SIFT_create()  # 调用SIFT
            key_points = sift_detector.detect(image)
            key_points, descriptors = sift_detector.compute(image, key_points)  # 求出sift描述子
            xs = np.empty((len(key_points), 1))
            ys = np.empty((len(key_points), 1))
            for i, kp in enumerate(key_points):
                xs[i], ys[i] = kp.pt
            answer.append({
                SpatialPyramidMatch.DESCRIPTORS: descriptors,
                SpatialPyramidMatch.X: xs,
                SpatialPyramidMatch.Y: ys,
                SpatialPyramidMatch.HEIGHT: image.shape[0],
                SpatialPyramidMatch.WIDTH: image.shape[1],
            })
        return answer


__all__ = ['SpatialPyramidMatch', ]
