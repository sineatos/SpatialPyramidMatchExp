# -*- encoding:UTF-8 -*-
"""
该模块包含了空间金字塔的实现
"""
from math import ceil, floor
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq


def calculate_sift(images, show_msg=False):
    """
    计算每一张图片的sift特征值
    :param images 一个图片集合的可迭代对象
    :param show_msg 显示操作信息
    :return: 一个列表
            [
                SIFTFeature(),
                ...
            ]
    """
    answer = []
    if show_msg:
        print("calculating sift ...")
    for image in images:
        sift_detector = cv2.xfeatures2d.SIFT_create()  # 调用SIFT
        key_points = sift_detector.detect(image)
        key_points, descriptors = sift_detector.compute(image, key_points)  # 求出sift描述子
        xs = np.empty((len(key_points), 1))
        ys = np.empty((len(key_points), 1))
        for i, kp in enumerate(key_points):
            xs[i], ys[i] = kp.pt
        answer.append(SIFTFeature(descriptors, xs, ys, image.shape[0], image.shape[1]))
    return answer


def generate_vocabulary_dictionary(features_list, keyword_amounts=None, iters=20, thresh=1e-5,
                                   dictionary_size=200, show_msg=False):
    """
    生成视觉词典，暂时不考虑优化
    该方法会修改SIFTFeature对象的textons属性，赋予其最近的聚类中心编号
    :param features_list: 图像的sift特征列表，每一个元素为一个SIFTFeature对象
    :param keyword_amounts: 聚类以后的关键字数目，若为None则聚类数目被设置为总的sift数目的五分之一
    :param iters: 聚类的最大迭代次数
    :param thresh: 聚类的时候的误差阈值
    :param dictionary_size: 词典大小，默认200
    :param show_msg: 显示操作信息
    :return: 统计量矩阵(一行为一张图片的视觉词典统计量直方图向量),视觉词典的关键字(即聚类中心)
    """
    # train_indices = np.random.choice(len(features_list), len(features_list), replace=False)
    sift_all = features_list[0].descriptors
    for features in features_list[1:]:
        sift_all = np.row_stack((sift_all, features.descriptors))
    if keyword_amounts is None:
        keyword_amounts = ceil(sift_all.shape[0] / 5)

    if show_msg:
        print("running kmeans ...")
    centers = kmeans(sift_all, k_or_guess=keyword_amounts, iter=iters, thresh=thresh)[0]

    if show_msg:
        print("building dictionary ...")
    # 获取图像中各个关键点最近的聚类中心，并赋值给features
    for features in features_list:
        features.textons = vq(features.descriptors, centers)[0]
    hist_all, hist_edges = np.histogram(features_list[0].textons, bins=dictionary_size)
    for features in features_list[1:]:
        hist, hist_edges = np.histogram(features, bins=dictionary_size)
        hist_all = np.row_stack((hist_all, hist))

    return hist_all, centers


def compile_pyramid(features, level=2, dictionary_size=200, show_msg=False):
    """
    使用一副图像得到的特征构建空间金字塔
    :param features: 一副图像的特征信息
    :param level: 构建的金字塔层数
    :param dictionary_size: 字典的大小
    :param show_msg: 显示操作信息
    :return: 一个长向量，该向量包含了一张图在各个尺度上的特征信息
    """
    if show_msg:
        print("building pyramid ...")
    width = features.width
    height = features.height
    bin_num = 2 ** level
    all_match_result = []
    match_result = np.empty((bin_num, bin_num, dictionary_size))
    for i in range(0, bin_num):
        for j in range(0, bin_num):
            x_lo = floor(width / bin_num * i)
            x_hi = floor(width / bin_num * (i + 1))
            y_lo = floor(height / bin_num * j)
            y_hi = floor(height / bin_num * (j + 1))
            indices = (features.xs > x_lo) * (features.xs <= x_hi) \
                      * (features.ys > y_lo) * (features.ys <= y_hi)  # 这里相当于进行与运算
            texton_patch = features.textons[indices[:, 0]]
            # np.histogram的range是左开右闭的
            match_result[i, j, :] = np.histogram(texton_patch
                                                 , bins=dictionary_size
                                                 , range=(0, dictionary_size))[0]  # / len(features.textons)
    all_match_result.append(match_result)
    pre_match_result = match_result
    for m_level in range(level - 1, 0, -1):
        bin_num = 2 ** m_level
        match_result = np.empty((bin_num, bin_num, dictionary_size))
        for i in range(0, bin_num):
            for j in range(0, bin_num):
                # 左上，右上，左下，右下
                match_result[i, j, :] = pre_match_result[2 * i, 2 * j, :] + \
                                        pre_match_result[2 * i, 2 * j + 1, :] + \
                                        pre_match_result[2 * i + 1, 2 * j, :] + \
                                        pre_match_result[2 * i + 1, 2 * j + 1, :]
        all_match_result.append(match_result)
        pre_match_result = match_result
    if level == 0:
        return match_result.flatten()
    for i, mr in enumerate(all_match_result):
        p = level if i == 0 else level - i + 1
        all_match_result[i] = mr.flatten() * 1 / (2 ** p)
    result = np.concatenate(all_match_result)
    return result


class SpatialPyramidMatch:
    """
    标准的空间金字塔匹配
    """

    def __init__(self, train_set, pyramid_level=2):
        """
        初始化方法
        :param train_set: 图片训练集，需要是一个可迭代对象，每一个元素都是一个像素矩阵
            :param pyramid_level: 空间金字塔的层数，默认为3层
            """
        self._features_list = calculate_sift(train_set)
        self._hists, self._centers = generate_vocabulary_dictionary(self._features_list)
        self._pyramid_level = pyramid_level


class SIFTFeature:
    """
    SIFT描述子特征
    """

    def __init__(self, descriptors, xs, ys, height, width):
        """
        :param descriptors: 描述子矩阵，第i个元素为第i个的关键点的特征向量
        :param xs: 关键点的x坐标序列
        :param ys: 关键点的y坐标序列
        :param height: 图像的高
        :param width: 图像的宽
        """
        self.descriptors = descriptors
        self.xs = xs
        self.ys = ys
        self.height = height
        self.width = width
        self.textons = None  # 记录关键点对应的聚类中心


__all__ = ['SpatialPyramidMatch', 'SIFTFeature', 'calculate_sift', 'generate_vocabulary_dictionary', 'compile_pyramid']
