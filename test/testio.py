# -*- encoding:UTF-8 -*-
from cvtools import io
from pprint import pprint

images = io.get_images_name(folder_path='../dataset/testing/Phoning')
imgs = list(images)
io.extract_sift_feature(imgs[0])
