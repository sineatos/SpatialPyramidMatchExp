# -*- encoding:UTF-8 -*-
"""
该模块包含了空间金字塔的实现
"""
from math import ceil, floor
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans


def calculate_sift(images):
    """
    计算每一张图片的sift特征值
    :param images 一个图片集合的可迭代对象
    :return: 一个列表
            [
                SIFTFeature(),
                ...
            ]
    """
    answer = []
    for image in images:
        sift_detector = cv2.xfeatures2d.SURF_create()  # 调用SIFT
        key_points = sift_detector.detect(image)
        key_points, descriptors = sift_detector.compute(image, key_points)  # 求出sift描述子
        xs = np.empty((len(key_points), 1))
        ys = np.empty((len(key_points), 1))
        for i, kp in enumerate(key_points):
            xs[i], ys[i] = kp.pt
        answer.append(SIFTFeature(descriptors, xs, ys, image.shape[0], image.shape[1]))
    return answer


def generate_vocabulary_dictionary(features_list, keyword_amounts=None, iters=100, thresh=0,
                                   dictionary_size=200, get_histogram_all=False):
    """
    生成视觉词典，暂时不考虑优化
    该方法会修改SIFTFeature对象的textons属性，赋予其最近的聚类中心编号
    :param features_list: 图像的sift特征列表，每一个元素为一个SIFTFeature对象
    :param keyword_amounts: 聚类以后的关键字数目，若为None则聚类数目被设置为总的sift数目的五分之一
    :param iters: 聚类的最大迭代次数
    :param thresh: 聚类的时候的误差阈值
    :param dictionary_size: 词典大小，默认200
    :param get_histogram_all: 是否返回所有图片的的视觉词典统计量组成的矩阵，默认为否
    :return: if get_histogram_all == True : 视觉词典的关键字(即聚类中心),统计量矩阵(一行为一张图片的视觉词典统计量直方图向量)
            else: 视觉词典的关键字(即聚类中心)
    """
    # train_indices = np.random.choice(len(features_list), len(features_list), replace=False)
    sift_all = []
    for features in features_list:
        sift_all.append(features.descriptors)
    sift_all = np.concatenate(sift_all, axis=0)
    if keyword_amounts is None:
        if sift_all.shape[0] < dictionary_size:
            keyword_amounts = sift_all.shape[0]
        else:
            keyword_amounts = dictionary_size
            # keyword_amounts = ceil(sift_all.shape[0] / 10)
    init_size = 300
    if dictionary_size > init_size:
        init_size = dictionary_size + 100
    m_kmeans = MiniBatchKMeans(init='k-means++', n_clusters=keyword_amounts, max_iter=iters, tol=thresh,
                               init_size=init_size)
    m_kmeans.fit(sift_all)
    centers = m_kmeans.cluster_centers_
    # centers = kmeans(sift_all, k_or_guess=keyword_amounts, iter=iters, thresh=thresh)[0]

    # 获取图像中各个关键点最近的聚类中心，并赋值给features
    for features in features_list:
        features.textons = vq(features.descriptors, centers)[0]
    if get_histogram_all:
        hist_all = []
        for features in features_list:
            hist, hist_edges = np.histogram(features.textons, bins=dictionary_size)
            hist_all.append(hist)
        hist_all = np.array(hist_all)
        return centers, hist_all
    else:
        return centers


def compile_pyramid(features, level=2, dictionary_size=200):
    """
    使用一副图像得到的特征构建空间金字塔
    :param features: 一副图像的特征信息
    :param level: 构建的金字塔层数
    :param dictionary_size: 字典的大小
    :return: 一个长向量，该向量包含了一张图在各个尺度上的特征信息
    """
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
                                                 , range=(0, dictionary_size))[0] / len(features.textons)
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
        p = level if i == 0 else (level - i + 1)
        all_match_result[i] = mr.flatten() * 1 / (2 ** p)
    result = np.concatenate(all_match_result)
    return result


# 直方图交核函数
def histogram_intersection(hist_m, hist_n):
    """
    求直方图的交，其中每一个直方图都是一个矩阵，第i行为第i张图片的长向量
    :param hist_m: 直方图m
    :param hist_n: 直方图n
    :return:
    """
    m = hist_m.shape[0]
    n = hist_n.shape[0]
    result = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            # hist_m的第i张图片与hist_n的第j张图片的交(对应bin的最小值求和，已加权)
            result[i][j] = np.sum(np.minimum(hist_m[i], hist_n[j]))
    return result


class SpatialPyramidMatch:
    """
    标准的空间金字塔匹配
    """

    def __init__(self, train_set, train_label, pyramid_level=2, svm_kernel='linear', keep_redundancy=False,
                 show_msg=False):
        """
        初始化方法
        :param train_set: 图片训练集，需要是一个可迭代对象，每一个元素都是一个像素矩阵
        :param train_label: 标签集(array-like)，长度必须要与训练集一样，每个元素表示着对应样本的标签
        :param pyramid_level: 空间金字塔的层数，默认为2层
        :param svm_kernel: svm的核函数，默认是rbf，可以是'linear','poly','rbf','sigmoid','precomputed','hi'
        :param keep_redundancy: 是否保留冗余数据，默认否
        :param show_msg: 是否显示操作信息，默认否
        """
        self._features_list = None  # 特征列表，每一个元素包含一张图片的特征
        self._centers = None  # 视觉词典聚类中心
        self._label_set = None  # 标签集合，是一个dict((标签编号,对应属性))
        self._label_list = None  # 图片对应的标签列表
        self._pyramid_level = None  # 金字塔的层数
        self._pyramid_matrix = None  # 金字塔长向量矩阵，每一行为一张图片的长向量
        self._svm_kernel = None  # svm的核函数
        self._svc_clf = None  # svm分类器
        self._keep_redundancy = keep_redundancy  # 是否保留冗余数据
        self._show_msg = show_msg
        self._calculate_sift(train_set)
        self._generate_vocabulary_dictionary()
        self._init_label_set_and_label_list(train_set, train_label)
        self._build_pyramid(pyramid_level)
        self._train_classificator(svm_kernel)

        if not self._keep_redundancy:  # 删除冗余数据
            del self._features_list
            del self._label_list

    @property
    def svm_kernel(self):
        return self._svm_kernel

    # 获取指定标签编号的标签
    def get_label(self, label_id):
        return self._label_set[int(label_id)]

    # 获取标签信息
    def get_label_info(self):
        return dict(self._label_set)

    # 计算图片的sift描述子
    def _calculate_sift(self, train_set):
        self._show_operate_msg("calculate sift")
        self._features_list = calculate_sift(train_set)

    # 生成视觉词典并构建直方图
    def _generate_vocabulary_dictionary(self):
        self._show_operate_msg("generate vocabulary dictionary")
        self._centers = generate_vocabulary_dictionary(self._features_list)

    # 初始化标签集
    def _init_label_set_and_label_list(self, train_set, train_label):
        self._show_operate_msg("init label set and label list")
        self._label_set = dict(enumerate((set(train_label))))
        tmp_ls = dict((v, k) for k, v in self._label_set.items())
        self._label_list = np.empty(len(train_set))
        for i in range(len(self._label_list)):
            self._label_list[i] = tmp_ls[train_label[i]]
        del tmp_ls

    # 构建空间金字塔
    def _build_pyramid(self, pyramid_level):
        self._show_operate_msg("build pyramid")
        self._pyramid_level = pyramid_level
        self._pyramid_matrix = []
        for features in self._features_list:
            self._pyramid_matrix.append(compile_pyramid(features, level=self._pyramid_level))
        self._pyramid_matrix = np.array(self._pyramid_matrix)  # 保存训练集中所有图片的空间金字塔长向量

    # 显示操作信息
    def _show_operate_msg(self, operation):
        if self._show_msg:
            print(operation, "is running ...")

    # SVM训练
    def _train_classificator(self, svm_kernel):
        self._show_operate_msg("train classificator")
        self._svm_kernel = svm_kernel
        if svm_kernel == 'hi' or svm_kernel == 'precomputed':
            train_matrix = histogram_intersection(self._pyramid_matrix, self._pyramid_matrix)
            self._svc_clf = svm.SVC(kernel="precomputed")
            self._svc_clf.fit(train_matrix, self._label_list)
        else:
            self._svc_clf = svm.SVC(kernel=self._svm_kernel)
            self._svc_clf.fit(self._pyramid_matrix, self._label_list)

    def predict(self, test_matrix):
        """
        预测图片的类别
        :param test_matrix: 金字塔长向量矩阵，每一行就是一张图片的金字塔长向量
        :return: 一个列向量，每一个元素就是代表图片属于哪一类
        """
        results = None
        if self._svm_kernel == 'hi' or self._svm_kernel == 'precomputed':
            predict_martix = histogram_intersection(test_matrix, self._pyramid_matrix)
            results = self._svc_clf.predict(predict_martix)
        else:
            results = self._svc_clf.predict(test_matrix)
        return results

    def predict_images(self, test_images):
        """
        预测图片的类别
        :param test_images: 一个图像对象列表，里面每一个对象都是一个numpy.ndarray对象
        :return: 一个列向量，每一个元素就是代表图片属于哪一类
        """
        test_features_list = calculate_sift(test_images)
        test_matrix = []
        for features in test_features_list:
            features.textons = vq(features.descriptors, self._centers)[0]
            test_matrix.append(compile_pyramid(features, level=self._pyramid_level))
        test_matrix = np.array(test_matrix)
        return self.predict(test_matrix)


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


__all__ = ['SpatialPyramidMatch', 'SIFTFeature', 'calculate_sift', 'generate_vocabulary_dictionary', 'compile_pyramid',
           'histogram_intersection']
