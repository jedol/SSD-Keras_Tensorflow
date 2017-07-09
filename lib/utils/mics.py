import numpy as np
import json


def save_json(file_path, data, indent=None):
    ## Input
    ##  file_path: string
    ##  data: python data

    ## save data
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(file_path):
    ## Input
    ##  file_path: string

    ## load data
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


def grid_image(images, row = 0, col = 0, padsize = 1, padval = 0):
    ## Input
    ##  images: 3d or 4d array=(n,h,w,c)

    ## take an image of shape (n, height, width) or (n, height, width, channels)
    n = images.shape[0]
    if row*col == 0:
        row = int(np.ceil(np.sqrt(n)))
        col = row
    assert n <= row*col, '[ERROR] n({}) > row*col({})'.format(n, row*col)

    ## Add padding
    padding = ((0, row*col - images.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (images.ndim - 3)
    images = np.pad(images, padding, mode='constant', constant_values=(padval, padval))

    ## tile the image
    images = images.reshape((row,col)+images.shape[1:]).transpose((0,2,1,3)+tuple(range(4, images.ndim + 1)))

    return images.reshape((row*images.shape[1],col*images.shape[3])+images.shape[4:])