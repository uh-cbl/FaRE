import os
import csv
import random
from tqdm import tqdm
import argparse
import sys

from fare.io import save_csv


def merge_train_test_sets():

    sub_dirs = os.listdir(args.db_dir)

    contents = {}
    subject_id_list = []

    for sub_dir in sub_dirs:
        l = os.listdir(os.path.join(args.db_dir, sub_dir))
        l.sort()
        contents[sub_dir] = l
        subject_id_list.extend(l)

    subject_id_list.sort()
    subject_id = {}

    for index, subject_dir in enumerate(subject_id_list):
        subject_id[subject_dir] = index

    final_contents = []

    p_bar = tqdm(total=len(subject_id_list))

    for sub_dir, contents in contents.items():
        for ss_dir in contents:
            identity = subject_id[ss_dir]
            cur_dir = os.path.join(args.db_dir, sub_dir, ss_dir)
            im_files = os.listdir(cur_dir)

            items = [[identity, os.path.join(sub_dir, ss_dir, im_file)] for im_file in im_files]

            final_contents.extend(items)
            p_bar.update()

    p_bar.close()

    if args.shuffle:
        random.shuffle(final_contents)

    save_csv(os.path.join(args.db_dir, 'VGG-FACE2.csv'), final_contents, header=['FILE', 'SUBJECT_ID'])

    with open(os.path.join(args.db_dir, 'VGG-FACE2.lst'), 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        for i, item in enumerate(final_contents):
            csv_writer.writerow([i, item[0], item[1]])

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', help='vgg face directory')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')
    args = parser.parse_args()

    merge_train_test_sets()
