import os
import csv
import random
from tqdm import tqdm
import argparse

from eugena_io import save_csv


def generate_training_list():

    subject_id_list = os.listdir(args.db_dir)

    subject_id_list.sort()
    subject_id = {}

    for index, subject_dir in enumerate(subject_id_list):
        subject_id[subject_dir] = index

    final_contents = []

    p_bar = tqdm(total=len(subject_id_list))

    for sub_dir in subject_id_list:
        identity = subject_id[sub_dir]
        cur_dir = os.path.join(args.db_dir, sub_dir)
        im_files = os.listdir(cur_dir)

        items = [[identity, os.path.join(sub_dir, im_file)] for im_file in im_files]

        final_contents.extend(items)
        p_bar.update()

    p_bar.close()

    if args.shuffle:
        random.shuffle(final_contents)

    save_csv(os.path.join(args.db_dir, 'WebFace.csv'), final_contents, header=['SUBJECT_ID', 'FILE', ])

    with open(os.path.join(args.db_dir, 'WebFace.lst'), 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        for i, item in enumerate(final_contents):
            csv_writer.writerow([i, item[0], item[1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', help='vgg face directory')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')
    args = parser.parse_args()

    generate_training_list()