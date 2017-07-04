# -*- encoding:UTF-8 -*-

from cvtools import io


def test_get_image_name_and_label(path):
    image_names = tuple(io.get_images_name(path, recursive=True))
    labels = io.get_image_label_in_filename(io.get_images_name(path, recursive=True))
    for image_name, label in zip(image_names, labels):
        print(image_name, label)


if __name__ == '__main__':
    test_get_image_name_and_label('../dataset/')
