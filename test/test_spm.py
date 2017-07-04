# -*- encoding:UTF-8 -*-

from os import path as ospath

from cvtools import io
from cvtools import spm
from cvtools import conf

train_data_path = '../dataset/training'
test_data_path = '../dataset/testing'

# 导入训练数据和测试数据数据
train_data = tuple(io.get_images_name(train_data_path, recursive=True))
test_data = tuple(io.get_images_name(test_data_path, recursive=True))
train_images = io.load_image2ndarray(train_data)
train_labels = io.get_image_label_in_filename(train_data)

mspm = spm.SpatialPyramidMatch(train_images, train_labels)

fullname = io.save_data(mspm, filename="spm.pkl")
print(fullname)
