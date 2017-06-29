# -*- encoding:UTF-8 -*-
from cvtools import io
from pprint import pprint

images = io.get_images_name(folder_path='../dataset/testing/Phoning')
imgs = list(images)
result_path = io.extract_sift_feature(imgs[0])
a,b = io.read_features_from_file(result_path)
print("****************** a ****************")
print(a)
print("****************** b ****************")
print(b)
