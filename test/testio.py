# -*- encoding:UTF-8 -*-
from cvtools import io
import cv2
from pprint import pprint

images = io.get_images_name(folder_path='../dataset/testing/Phoning')
imgs = list(images)
result_path = io.extract_sift_feature(imgs[0])
a, b = io.read_features_from_file(result_path)

im = cv2.imread(imgs[0])
cv2.imshow('original', im)
for k in range(a.shape[0]):
    cv2.circle(im, (int(a[k][0]), int(a[k][1])), 1, (0, 255, 0), -1)

cv2.imshow('SURF_features', im)
cv2.waitKey()
cv2.destroyAllWindows()
