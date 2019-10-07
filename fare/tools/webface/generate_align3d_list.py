import os
import csv
import random
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import numpy as np
from fare.io import save_csv
import matplotlib.pyplot as plt


def crop_image(item):
    identity = item[0]
    im_path = item[1]
    out_path = im_path.replace('._texture', '')
    if os.path.exists(os.path.join(args.out_dir, out_path)):
        return identity, out_path
    else:
        im = cv2.imread(os.path.join(args.db_dir, im_path))

        # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # plt.show()
        im_crop = im[48: 208, 48: 208]

        if args.flip:
            im_mask = cv2.imread(os.path.join(args.db_dir, im_path.replace('texture', 'mask')), 0)
            im_mask = im_mask[48: 208, 48: 208]
            # mask sum
            mask1 = np.sum(im_mask[:, 0: 80])
            mask2 = np.sum(im_mask[:, 80: 160])

            im_flip = cv2.flip(im_crop, 1)

            if mask1 > mask2:
                im_crop[:, 0: 80] = im_flip[:, 0: 80]
            else:
                im_crop[:, 80: 160] = im_flip[:, 80: 160]

        im_crop = cv2.flip(im_crop, 0)
        im_crop = cv2.resize(im_crop, (args.im_size, args.im_size))

        cv2.imwrite(os.path.join(args.out_dir, out_path), im_crop)

        if args.plot:
            plt.imshow(cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB))
            plt.show()

        return identity, out_path


def generate_training_list():

    subject_id_list = os.listdir(args.db_dir)

    subject_id_list.sort()
    subject_id = {}

    for index, subject_dir in enumerate(subject_id_list):
        if os.path.isdir(os.path.join(args.db_dir, subject_dir)):
            subject_id[subject_dir] = index

            if not os.path.exists(os.path.join(args.out_dir, subject_dir)):
                os.makedirs(os.path.join(args.out_dir, subject_dir))

    final_contents = []

    p_bar = tqdm(total=len(subject_id_list))

    for sub_dir in subject_id_list:
        identity = subject_id[sub_dir]
        cur_dir = os.path.join(args.db_dir, sub_dir)

        im_files = os.listdir(cur_dir)
        im_files = [im_file for im_file in im_files if 'texture' in im_file]
        items = [[identity, os.path.join(sub_dir, im_file)] for im_file in im_files]

        final_contents.extend(items)
        p_bar.update()

    p_bar.close()

    with Pool() as p:
        final_contents = list(tqdm(p.imap_unordered(crop_image, final_contents), total=len(final_contents)))

    final_contents = list(filter(None, final_contents))

    if args.shuffle:
        random.shuffle(final_contents)

    if len(args.out_file) > 0:
        save_csv(args.out_file, final_contents, header=['SUBJECT_ID', 'FILE', ])

        with open(args.out_file.replace(os.path.splitext(args.out_file)[1], '.lst'), 'w') as f:
            csv_writer = csv.writer(f, delimiter='\t')
            for i, item in enumerate(final_contents):
                csv_writer.writerow([i, item[0], item[1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', help='vgg face directory')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--im-size', default=112, help='image size')
    parser.add_argument('--flip', action='store_true', help='flip')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')
    parser.add_argument('--out-file', default='', help='output file')
    parser.add_argument('--plot', action='store_true', help='plot')
    args = parser.parse_args()

    generate_training_list()
