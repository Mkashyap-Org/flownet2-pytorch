#!/usr/local/bin/python3

import cv2
import os
import glob
import timeit
import numpy as np
from PIL import Image

#def fast_pencil(input_image, element):
#    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
#    pencil_image = cv2.dilate(gray, element)
    # 65026 = 255**2 + 1 <=> the division will always be < 1.0 and thus casted to 0
#    pencil_image[pencil_image == 0] = 65026
    # NEED to write 255.0 and not 255 because we multiply 2 uint8 and thus the value overflows
#    return np.expand_dims(255.0 * gray / pencil_image, axis=2).astype(np.uint8)


def orig_pencil(input_image, element):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    pencil_image = cv2.dilate(input_image, element)

    valorig = input_image.copy()
    valmax = pencil_image.copy()
    valout = pencil_image.copy()
    alpha_image = pencil_image.copy()

    valout[valmax == 0] = 255

    alpha_image[valmax > 0] = (
        valorig[valmax > 0]*float(255.0)) / valmax[valmax > 0]

    valout[valout < 255] = alpha_image[valout < 255]

    pencil_image = valout.copy()
    pencil_image = np.expand_dims(pencil_image, axis=2)
    return pencil_image


if __name__ == "__main__":
    path = "C:\\Users\\Mahesh\\Desktop\\Kaiserslatern\\Sem3\\Project\\Imp\\FlyingChairs1_Samp\\"
    penciled_path = "C:\\Users\\Mahesh\\Desktop\\Kaiserslatern\\Sem3\\Project\\Imp\\Penciled\\"
    for file in os.listdir(path):
        # c = cv2.imread(file)
        im = Image.open(file)
        print(file)
        # im = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        # cv2.imshow('image', im)
        # cv2.waitKey(0)
        # NOTE: the 120 is a parameter which should be adapted to the input image size
        dilate_size = int(im.shape[0] / 120)
        kernel_size = 2*dilate_size + 1

        element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size), (dilate_size, dilate_size))

#    fast = fast_pencil(im, element)
        orig = orig_pencil(im, element)
#    print(np.array_equal(orig, fast))

#    times = timeit.repeat(lambda: fast_pencil(
#        im, element), number=10000, repeat=1)
#    print('Fast pencil time: {}'.format(min(times)))

        times = timeit.repeat(lambda: orig_pencil(
        im, element), number=10000, repeat=1)
        print('Orig pencil time: {}'.format(min(times)))

        cv2.imwrite(os.path.join(penciled_path, file), orig)
        cv2.imshow("penciled", orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()