# -*- encoding:UTF-8 -*-

import cv2
from cvtools import SpatialPyramidMatch

ii = r'H:\Pictures\Saved Pictures\2.jpg'  # '../dataset/testing/Phoning/Phoning_0041.jpg'
im = cv2.imread(ii)
aa = SpatialPyramidMatch.calculate_sift([im, ])

print(len(aa))
print(aa[0])
