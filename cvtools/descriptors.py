# -*- encoding:UTF-8 -*-

import numpy as np

class SIFTDescriptor:
    """
    SIFT 描述子
    """

    def __init__(self, width, height, descriptor):
        self.width = width
        self.height = height
        self.descriptor = SIFTDescriptor._normalize_sift(descriptor)

    @staticmethod
    def _normalize_sift(descriptor):
        descriptor = np.array(descriptor)
        norm = np.linalg.norm(descriptor)
        if norm > 1.0:
            descriptor /= float(norm)
        return descriptor


class ImageDescriptors:
    """
    图片描述子
    """

    def __init__(self, descriptors, label, width, height):
        self.descriptors = descriptors
        self.label = label
        self.width = width
        self.height = height

__all__ = ['SIFTDescriptor', 'ImageDescriptors']
