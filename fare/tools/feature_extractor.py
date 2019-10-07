import os
import mxnet as mx
from mxnet import nd, image, gluon
import numpy as np
import argparse
import timeit
import datetime

from fare.io import glob_files, Signature


def image_processing(im_path):
    try:
        with open(im_path, 'rb') as f:
            im = image.imdecode(f.read())

        for aug_func in augmenter:
            im = aug_func(im)

        im = im.transpose((2, 0, 1)).expand_dims(axis=0)

        return im
    except IOError as e:
        print(e)
        return None


def extracting_features():
    # glob images
    im_paths = glob_files(args.db_dir, args.im_exts)

    feature_paths = [im_path.replace(args.db_dir, args.ft_dir).replace(os.path.splitext(os.path.basename(im_path))[1], args.ft_ext) for im_path in im_paths]

    # generate feature directory
    [os.makedirs(os.path.dirname(feature_path)) for feature_path in feature_paths if not os.path.exists(os.path.dirname(feature_path))]

    # loading model
    use_symbol_block = False
    if len(args.symbol_prefix) > 0:
        use_symbol_block = True
        sym, arg_params, aux_params = mx.model.load_checkpoint(args.symbol_prefix, 0)
        internals = sym.get_internals()
        inputs = internals['data']
        outputs_blobs = [internals[layer_name + '_output'] for layer_name in args.output_blobs.split(',')]
        inference = gluon.SymbolBlock(outputs_blobs, inputs)
    else:
        inference = gluon.model_zoo.vision.get_model(args.arch, classes=args.num_classes)

    inference.load_params(args.weights, ctx=ctx)

    if len(args.symbol_prefix) == 0:
        inference.hybridize()

    # extracting features
    global_start_time = timeit.default_timer()
    valid_counts = 0
    for im_idx, (im_path, feature_path) in enumerate(zip(im_paths, feature_paths)):
        if not os.path.exists(feature_path):
            start_time = timeit.default_timer()
            im = image_processing(im_path)
            if im is not None:
                im = im.as_in_context(ctx)
                elapsed_time = timeit.default_timer()
                processing_time = elapsed_time - start_time

                start_time = timeit.default_timer()
                if use_symbol_block:
                    features = inference(im)
                else:
                    features = inference.features(im)
                feature_extracting_time = timeit.default_timer() - start_time

                features = features.asnumpy().flatten()
                feature = Signature(features)
                feature.save_features(feature_path)

                valid_counts += 1

                print('Presssed [%d/%d]: %s \t Pre-processing time: %.2f ms\t Extracting features time: %.2f ms' %
                      (im_idx, len(im_paths), im_path, processing_time * 1000, feature_extracting_time * 1000))
        else:
            print('The feature file exists, skip the file')

    global_elapsed_time = timeit.default_timer() - global_start_time
    print('Total elapsed time: %s \t Processed [%d] images \t Avg. time: %.2f ms' %
          (str(datetime.timedelta(seconds=global_elapsed_time)), valid_counts,
           global_elapsed_time * 1000 / valid_counts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', help='dataset directory', type=str)
    parser.add_argument('ft_dir', help='output features directory', type=str)
    parser.add_argument('--im-exts', default='.jpg,.jpeg,.png,.bmp', type=str, help='image extensions')
    parser.add_argument('--ft-ext', default='.txt', type=str, help='extension')
    parser.add_argument('--data-shape', default='3, 224, 224', type=str, help='data shape')
    parser.add_argument('--mean', default='127.5,127.5,127.5', type=str, help='mean (r, g, b)')
    parser.add_argument('--std', default='128,128,128', type=str, help='mean (r, g, b)')
    parser.add_argument('--resize', default=256, type=int, help='resize images')

    parser.add_argument('--weights', default='', type=str, help='model weights path')
    parser.add_argument('--gpu', default=0, type=int, help='gpu')

    sgroup = parser.add_argument_group()
    sgroup.add_argument('--symbol-prefix', default='', type=str, help='symbol path')
    sgroup.add_argument('--output-blobs', default='', type=str, help='output blocks name')

    igroup = parser.add_argument_group()
    igroup.add_argument('--arch', default='resnet101_v2', help='architecture')
    igroup.add_argument('--num_classes', default=1000, type=int, help='num classes: WebFace: 10575, VGG-Face2: 9131')

    args = parser.parse_args()

    args.im_exts = [ext.strip() for ext in args.im_exts.split(',')]

    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()

    data_shape = tuple(map(int, args.data_shape.split(',')))
    mean_rgb = tuple(map(float, args.mean.split(',')))
    std_rgb = tuple(map(float, args.std.split(',')))

    augmenter = image.CreateAugmenter(data_shape, resize=args.resize, mean=np.array(mean_rgb), std=np.array(std_rgb))

    extracting_features()
