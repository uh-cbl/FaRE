import os
from eugena.io import glob_files
import csv
import argparse


def merge_im_box():
    im_files = glob_files(args.db_dir, ['.jpg', '.jpeg', '.png', '.bmp'])
    box_files = [im_file.replace(args.db_dir, args.box_dir).replace(os.path.splitext(im_file)[1], args.box_ext)
                 for im_file in im_files]

    content = [[im_file.replace(args.db_dir, ''), box_file] for im_file, box_file in zip(im_files, box_files)]
    with open(args.out_csv, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['FILE', 'ROI'])
        for line in content:
            csv_writer.writerow(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', help='Dataset directory')
    parser.add_argument('box_dir', help='Dataset directory')
    parser.add_argument('out_csv', help='output csv file')
    parser.add_argument('--box-ext', default='.roi', help='box extension')

    args = parser.parse_args()

    if args.db_dir[-1] != '/':
        args.db_dir = args.db_dir + '/'

    if args.box_dir[-1] != '/':
        args.box_dir = args.box_dir + '/'

    merge_im_box()
