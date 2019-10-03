import mxnet as mx
from mxnet import nd, image
import numpy as np


def extracting_features_with_symbol(symbol, weights, output_blob, pre_processing_func, datasets, out_dir):
    print('extracting features with symbol')


def extracting_features_with_gluon(inference, weights, output_blog, pre_precssing_function, datasets, out_dir):
    print('extracting features with gluon')


def pre_processing(im_path, data_shape, resize=128, mean_rgb=(0, 0, 0), std_rgb=(1, 1, 1)):
    try:
        with open(im_path, 'rb') as f:
            im = image.imdecode(f.read())

        augmenter = image.CreateAugmenter(data_shape, resize=resize, mean=np.array(mean_rgb), std=np.array(std_rgb))

        for aug_func in augmenter:
            im = aug_func(im)

        im = im.transpose((2, 0, 1)).expand_dims(axis=0)

        return im
    except Exception as e:
        print(e)
        return None
