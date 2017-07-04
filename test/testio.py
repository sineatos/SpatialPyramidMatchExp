# -*- encoding:UTF-8 -*-

from cvtools import io


def test_get_image_name_and_label(path):
    for image_name in io.get_images_name(path, recursive=True):
        label = io.get_image_label_in_filename(image_name)
        print(image_name, label)


if __name__ == '__main__':
    test_get_image_name_and_label('../dataset/')
