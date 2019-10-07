import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import numpy as np
from fare.io import glob_files
import matplotlib.pyplot as plt


def crop_image(im_path):
    try:
        out_path = im_path.replace('._texture', '').replace(args.db_dir, args.out_dir)
        if not os.path.exists(out_path):
            im = cv2.imread(im_path)
            im = cv2.resize(im, (256, 256))
            im_crop = im[48: 208, 48: 208]

            if args.flip or args.half:
                im_mask = cv2.imread(im_path.replace('texture', 'mask'), 0)
                im_mask = im_mask[48: 208, 48: 208]
                # mask sum
                mask1 = np.sum(im_mask[:, 0: 80])
                mask2 = np.sum(im_mask[:, 80: 160])

                im_flip = cv2.flip(im_crop, 1)

                if mask1 > mask2:
                    im_crop[:, 0: 80] = im_flip[:, 0: 80]
                else:
                    im_crop[:, 80: 160] = im_flip[:, 80: 160]

                if args.half:
                    im_crop = im_crop[:, 0:80]

            im_crop = cv2.flip(im_crop, 0)

            if args.half:
                im_crop = cv2.resize(im_crop, (args.im_size // 2, args.im_size))
            else:
                im_crop = cv2.resize(im_crop, (args.im_size, args.im_size))

            cv2.imwrite(out_path, im_crop)

            if args.plot:
                plt.imshow(cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB))
                plt.show()
    except Exception as e:
        print(e)


def generate_training_list():

    file_list = glob_files(args.db_dir, ['.png'])
    file_list = [f for f in file_list if 'texture' in os.path.basename(f)]

    [os.makedirs(os.path.dirname(f.replace(args.db_dir, args.out_dir))) for f in file_list if not
    os.path.exists(os.path.dirname(f.replace(args.db_dir, args.out_dir)))]

    with Pool() as p:
        content = list(tqdm(p.imap_unordered(crop_image, file_list), total=len(file_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', help='data directory')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--im-size', default=112, help='image size')
    parser.add_argument('--half', action='store_true', help='save half face')
    parser.add_argument('--flip', action='store_true', help='flip')
    parser.add_argument('--plot', action='store_true', help='plot')
    args = parser.parse_args()

    generate_training_list()
