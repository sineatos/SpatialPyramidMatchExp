# -*- encoding:UTF-8 -*-
"""
该模块保存一些与配置相关的变量和方法
"""
import os
from os import path as ospath

# 临时目录
TMP_PATH = './tmp'

# vlfeat的路径，用于提取sift特征
VLFEAT_LOCATION = r'D:\vlfeat-0.9.20-bin\vlfeat-0.9.20\bin\win64'


def make_tmp_path():
    """创建临时目录"""
    os.makedirs(TMP_PATH, exist_ok=True)


__all__ = ['TMP_PATH', 'make_tmp_path', 'VLFEAT_LOCATION']
