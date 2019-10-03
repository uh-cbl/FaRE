"""
This script is used to align 2D images according to the landmarks
CSV file defined the bounding box and landmarks files
"""

import argparse
import os
import tqdm
import cv2

from eugena.geometry import Align2D
from eugena.io import load_csv, save_csv, load_lm2d
from multiprocessing import Pool

import matplotlib.pyplot as plt

IMG_IDX, ID_IDX, BOX_IDX, LM_IDX = 'FILE', 'SUBJECT_ID', 'ROI', 'LM2D'
UR2D_LM7 = [4, 6, 7, 9, 11, 13, 14]


def align_image(item):
    """
    align image
    :param item
    :return: None
    """
    img_path, lm_path = item['FILE'], item['LM2D']

    output_path = os.path.join(args.out_dir, img_path)
    align_processor = Align2D(args.im_size, n_landmarks=7)

    if os.path.exists(output_path):
        return output_path

    try:
        if os.path.exists(os.path.join(args.db_dir, img_path)):
            im = cv2.imread(os.path.join(args.db_dir, img_path))

            lm = load_lm2d(lm_path)

            lm = lm[UR2D_LM7]
            if args.plot:
                plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                plt.scatter(lm[:, 0], lm[:, 1])
                plt.show()

            thumbnail = align_processor.align(im, lm)
            if args.plot:
                plt.imshow(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
                plt.show()

            cv2.imwrite(output_path, thumbnail)

            return img_path

    except Exception as e:
        print(e)
        print(img_path)


def main():

    contents = load_csv(args.list, read_header=True)

    # create out dir
    for row in contents:
        if not os.path.exists(os.path.join(args.out_dir, os.path.dirname(row['FILE']))):
            os.makedirs(os.path.join(args.out_dir, os.path.dirname(row['FILE'])))

    with Pool() as p:
        contents_post = list(tqdm.tqdm(p.imap_unordered(align_image, contents), total=len(contents)))

    contents_post = list(filter(None, contents_post))

    if len(args.out_path) > 0:
        save_csv(args.out_path, contents_post, header=['FILE'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('list', help='input csv file')
    parser.add_argument('db_dir', help='dataset directory')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--im-size', default=112, type=int, help='image dimension')
    parser.add_argument('--out-path', default='', type=str, help='output path')
    parser.add_argument('--plot', action='store_true', help='plot')
    args = parser.parse_args()

    main()
