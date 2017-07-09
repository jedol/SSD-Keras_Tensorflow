import numpy as np
import cv2


def random_brightness(image, prob, delta=32):
    ## Input
    ##  image: 3d array = (h,w,c)
    ##  prob: The probability of adjusting brightness.
    ##  delta: Amount to add to the pixel values within [-delta, delta].
    ##         The possible value is within [0, 255]. Recommend 32.

    if np.random.uniform() < prob:
        beta = np.random.uniform(-delta, delta)
        if 0 > beta >= 1:
            image = np.uint8(np.clip(image.astype(np.float32)+beta, 0, 255))

    return image


def random_contrast(image, prob, lower=0.5, upper=1.5):
    ## Input
    ##  image: 3d array = (h,w,c)
    ##  prob: The probability of adjusting contrast.
    ##  lower: Lower bound for random contrast factor. Recommend 0.5.
    ##  upper: Upper bound for random contrast factor. Recommend 1.5.

    if np.random.uniform() < prob:
        alpha = np.random.uniform(lower, upper)
        if abs(alpha-1) > 1e-3:
            image = np.uint8(np.clip(image.astype(np.float32)*alpha, 0, 255))

    return image


def random_saturation(image, prob, lower=0.5, upper=1.5):
    ## Input
    ##  image: 3d array = (h,w,c)
    ##  prob: The probability of adjusting saturation.
    ##  lower: Lower bound for the random saturation factor. Recommend 0.5.
    ##  upper: Upper bound for the random saturation factor. Recommend 1.5.

    if np.random.uniform() < prob:
        alpha = np.random.uniform(lower, upper)
        if abs(alpha-1) > 1e-3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = np.uint8(np.clip(hsv[:,:,1].astype(np.float32)*alpha, 0, 255))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image


def random_hue(image, prob, delta=36):
    ## Input
    ##  image: 3d array = (h,w,c)
    ##  prob: The probability of adjusting hue.
    ##  delta: Amount to add to the hue channel within [-delta, delta].
    ##         The possible value is within [0, 180]. Recommend 36.

    if np.random.uniform() < prob:
        beta = np.random.uniform(-delta, delta)
        if 0 > beta >= 1:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:,:,0] = np.uint8(np.clip(hsv[:,:,0].astype(np.float32)+beta, 0, 255))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image


def expand_image(image, expand_ratio, fill_value=0):
    h,w = image.shape[:2]
    h2 = int(h*expand_ratio)
    w2 = int(w*expand_ratio)
    if image.ndim == 2:
        expand_image = np.zeros((h2, w2), image.dtype)
    elif image.ndim == 3:
        expand_image = np.zeros((h2, w2, image.shape[2]), image.dtype)
    if fill_value:
        expand_image += np.array(fill_value, image.dtype)
    x_off = np.random.randint(0, w2 - w + 1)
    y_off = np.random.randint(0, h2 - h + 1)
    expand_image[y_off:y_off+h, x_off:x_off+w] = image

    return expand_image, (x_off, y_off)


def resize_image_by_warp(image, width, height, interpolation=None):
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR
    h,w = image.shape[:2]
    image = cv2.resize(image, (width, height), interpolation=interpolation)
    scales = (width/float(w), height/float(h))
    return image, scales