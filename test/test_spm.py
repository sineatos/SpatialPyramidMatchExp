# -*- encoding:UTF-8 -*-

import cv2
from cvtools import *

ii = r'H:\Pictures\Saved Pictures\2.jpg'  # '../dataset/testing/Phoning/Phoning_0041.jpg'
im = cv2.imread(ii)
features_list = calculate_sift([im, ], show_msg=True)
hist_all, centers = generate_vocabulary_dictionary(features_list, show_msg=True)
for features in features_list:
    vec = compile_pyramid(features)
    print(vec)
