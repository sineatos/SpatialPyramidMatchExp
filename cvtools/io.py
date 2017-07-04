# -*- encoding:UTF-8 -*-
"""
该模块包含一些读写相关的函数
"""

import pickle
import os
import os.path as ospath
import re
from PIL import Image
import cv2


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


def load_pil_images(folder_path, suffixes=('jpg', 'png',), recursive=False):
    """迭代读入一个目录中的所有图片，但是不会递归读取。
    :param folder_path: 要读取图片的目录
    :param suffixes: 接受图片的后缀名元组
    :param recursive: 是否递归读取，默认否
    :return: 一个迭代器，迭代的时候，每一次返回一个PIL.Image对象
    """
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            pre, suf = ospath.splitext(file)
            if suf in suffixes:
                image = Image.open(file)
                images.append(image)
        if not recursive:
            break
    return images


def get_images_name(folder_path, suffixes=('.jpg', '.png',), recursive=False):
    """迭代读入一个目录中的所有图片的路径。
        :param folder_path: 要读取图片的目录
        :param suffixes: 接受图片的后缀名元组
        :param recursive: 是否递归读取，默认否
        :return: 一个迭代器，迭代的时候，每一次返回一个图片的路径名
        """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            pre, suf = ospath.splitext(file)
            if suf in suffixes:
                yield ospath.join(root, file)
        if not recursive:
            break


def get_image_label_in_filename(paths, label_re=r'^(.*)_.*$'):
    """
    从文件命中获取图像的标签，该方法首先会使用os.path.basename获取文件路径中的文件名，然后使用正则表达式获取文件名中的标签
    注意：提取的标签为正则表达式中的第一个括号里的内容。
    :param paths: 文件路径列表
    :param label_re: 正则表达式字符串，默认为文件名以"标签_其他文字和后缀名"作为名称
    :return: 返回图像的标签列表
    """
    labels = []
    for path in paths:
        filename = ospath.basename(path)
        mo = re.match(label_re, filename)
        labels.append(mo.group(1))
    return labels


def load_image2ndarray(paths):
    """
    根据图像路径并将图像转化为一个numpy.ndarray的对象返回，接收的输入为一个可迭代对象
    :param paths: 图像路径列表
    :return: numpy.ndarray对象列表
    """
    return [cv2.imread(path) for path in paths]


__all__ = ['save_data', 'load_data', 'load_pil_images', 'get_images_name', 'get_image_label_in_filename',
           'load_image2ndarray']
