# -*- encoding:UTF-8 -*-
"""
该模块包含一些读写相关的函数
"""

import pickle
import os
import os.path as ospath
import numpy as np
from PIL import Image
from cvtools import conf


def save_data(data, path_prefix="data", filename="data.bin", mode="wb"):
    """保存数据
    :param data: 数据对象
    :param path_prefix: 保存的目录名
    :param filename: 数据的文件名
    :param mode: 写模式
    :return: 如果保存成功返回文件路径
    """
    os.makedirs(path_prefix, exist_ok=True)
    full_filename = ospath.join(path_prefix, filename)
    with open(full_filename, mode) as f:
        pickle.dump(data, f)
    return full_filename


def load_data(path_prefix="data", filename="data.bin", mode="rb"):
    """导入数据
    :param path_prefix: 保存的目录名
    :param filename: 数据的文件名
    :param mode: 读模式
    :return: 返回数据对象
    """
    full_filename = ospath.join(path_prefix, filename)
    with open(full_filename, mode) as f:
        return pickle.load(f)


def load_images(folder_path, suffixes=('jpg', 'png',)):
    """迭代读入一个目录中的所有图片，但是不会递归读取。
    :param folder_path: 要读取图片的目录
    :param suffixes: 接受图片的后缀名元组
    :return: 一个迭代器，迭代的时候，每一次返回一个PIL.Image对象
    """
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            pre, suf = ospath.splitext(file)
            if suf in suffixes:
                image = Image.open(file)
                images.append(image)
        break
    return images


def get_images_name(folder_path, suffixes=('.jpg', '.png',)):
    """迭代读入一个目录中的所有图片的路径，但是不会递归读取。
        :param folder_path: 要读取图片的目录
        :param suffixes: 接受图片的后缀名元组
        :return: 一个迭代器，迭代的时候，每一次返回一个图片的路径名
        """
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            pre, suf = ospath.splitext(file)
            if suf in suffixes:
                yield ospath.join(folder_path, file)
        break


def extract_sift_feature(image_path, size=30, steps=15, force_orientation=False, resize=None):
    """ 抽取图片的sift特征
    :param image_path: 图片文件的路径
    :param size:
    :param steps:
    :param force_orientation:
    :param resize:
    :return: sift保存的文件位置
    """
    im = Image.open(image_path).convert('L')
    if resize is not None:
        im = im.resize(resize)
    m, n = im.size

    conf.make_tmp_path()

    if ospath.splitext(image_path)[1] != '.pgm':
        # 创建一个pgm文件
        pgm = ospath.join(conf.TMP_PATH, 'tmp.pgm')
        im.save(pgm)
        image_path = pgm

    # 构建框并保存在临时文件里面
    tmp_frame = ospath.join(conf.TMP_PATH, 'tmp.frame')
    scale = size / 3.0
    x, y = np.meshgrid(range(steps, m, steps), range(steps, n, steps))
    xx, yy = x.flatten(), y.flatten()
    frame = np.array([xx, yy, scale * np.ones(xx.shape[0]), np.zeros(xx.shape[0])])
    np.savetxt(tmp_frame, frame.T, fmt='%03.3f')

    result_path = ospath.join(conf.TMP_PATH, 'tmp.sift')
    cmmd = ["sift", image_path, "--output=%s --read-frames=%s" % (result_path, tmp_frame,)]
    if force_orientation:
        cmmd.append("--orientations")
    if conf.VLFEAT_LOCATION not in os.environ['PATH']:
        os.environ['PATH'] += conf.VLFEAT_LOCATION
        print("Extract SIFT features of " + image_path)
    # print(' '.join(cmmd).replace('/', os.sep))
    os.system(' '.join(cmmd).replace('/', os.sep))
    return result_path


def read_features_from_file(filename):
    """ 从文件中读取特征属性并以矩阵的形式返回"""
    f = np.loadtxt(filename)
    return f[:, :4], f[:, 4:]  # 特征位置(前四列), 描述子(从第四列以后的所有列)

__all__ = ['save_data', 'load_data', 'load_images', 'get_images_name', 'extract_sift_feature','read_features_from_file']
