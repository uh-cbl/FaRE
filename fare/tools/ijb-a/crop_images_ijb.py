import os
import csv
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import cv2

from eugena.io import glob_files
from eugena.geometry.rectangles import Rectangle


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from collections import OrderedDict


def crop_image(line):
    im_path, template_id, subject_id, x, y, w, h, out_im_path, out_box_path = line

    if os.path.exists(out_im_path) and os.path.exists(out_box_path):
        return None
    else:
        try:
            # enlarge box
            rect1 = Rectangle(x, y, x+w, y+h)
            rect2 = Rectangle(x, y, x+w, y+h)
            rect2.enlarge_bbox(args.enlarge_factor)

            im = cv2.imread(os.path.join(args.db_dir, im_path))

            im_pad = cv2.copyMakeBorder(im, rect2.height, rect2.height, rect2.width, rect2.width, cv2.BORDER_CONSTANT, value=127)

            rect1.translation(rect2.width, rect2.height)
            rect2.translation(rect2.width, rect2.height)

            # crop image
            im_crop = im_pad[rect2.top: rect2.bottom, rect2.left: rect2.right]
            rect1.translation(-rect2.left, -rect2.top)

            # plot images
            if args.plot:
                plt.imshow(cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB))
                plt.gca().add_patch(patches.Rectangle((rect1.left, rect1.top), rect1.width, rect1.height, fill=False,
                                                      color='g'))
                plt.show()

            # save image
            cv2.imwrite(out_im_path, im_crop)

            # save bbox
            if args.save_box:
                with open(out_box_path, 'w') as f:
                    f.write('%d %d %d %d' % (rect1.left, rect1.top, rect1.width, rect1.height))

            return out_im_path.replace(args.out_dir, ''), template_id, subject_id, out_box_path
        except IOError as e:
            print(e)
            print(line)
            return None


def write_csv(file_path, contents):
    with open(file_path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['FILE', 'TEMPLATE_ID', 'SUBJECT_ID', 'ROI'])
        for row in contents:
            csv_writer.writerow([row[0], row[1], row[2], os.path.realpath(row[-1])])


def write_final_csv():
    files = glob_files(args.out_dir, ['.png'])

    p_bar = tqdm(total=len(files))

    with open(os.path.join(args.out_dir, 'IJB-%s.csv' % args.set), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['FILE', 'ROI'])

        for file_path in files:
            box_path = os.path.realpath(file_path).replace('.png', '.roi')

            csv_writer.writerow([file_path.replace(args.out_dir, ''), box_path])

            p_bar.update()

    p_bar.close()


def remove_duplicates(contents):
    dicts = OrderedDict()
    for row in contents:
        if row[0] not in dicts:
            dicts[row[0]] = row[1:]

    contents = [[k, *v] for k, v in dicts.items()]
    return contents


def main():
    if args.set == 'A':
        files = glob_files(os.path.join(args.db_dir, 'IJB-A_1N_sets'), ['.csv'])
    elif args.set == 'B':
        files = [os.path.join(args.db_dir, 'protocol/ijbb_face_detection.csv')]
    elif args.set == 'C':
        files = [os.path.join(args.db_dir, 'protocol/ijbc_face_detection_ground_truth.csv')]
    else:
        raise NotImplementedError

    contents = []
    idx_tmp_id, idx_sub_id, idx_file, idx_x, idx_y, idx_w, idx_h, idx_reye_x, idx_reye_y, idx_leye_x, idx_leye_y, \
    idx_nose_x, idx_nose_y = 0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

    for file_ind, file_path in enumerate(files):
        print('Processing [%d]: %s...' % (file_ind, file_path))

        with open(file_path) as f:
            csv_reader = csv.reader(f)
            for idx, row in enumerate(csv_reader):

                if idx == 0:
                    idx_tmp_id = row.index('TEMPLATE_ID')
                    idx_sub_id = row.index('SUBJECT_ID')
                    idx_file = row.index('FILE')
                    idx_x = row.index('FACE_X')
                    idx_y = row.index('FACE_Y')
                    idx_w = row.index('FACE_WIDTH')
                    idx_h = row.index('FACE_HEIGHT')
                    idx_reye_x = row.index('RIGHT_EYE_X')
                    idx_reye_y = row.index('RIGHT_EYE_Y')
                    idx_leye_x = row.index('LEFT_EYE_X')
                    idx_leye_y = row.index('LEFT_EYE_Y')
                    idx_nose_x = row.index('NOSE_BASE_X')
                    idx_nose_y = row.index('NOSE_BASE_Y')
                else:
                    template_id = row[idx_tmp_id]
                    subject_id = row[idx_sub_id]
                    im_path = row[idx_file]

                    x, y, w, h = float(row[idx_x]), float(row[idx_y]), float(row[idx_w]), float(row[idx_h])

                    reye_x, reye_y, leye_x, leye_y, nose_x, nose_y = row[idx_reye_x], row[idx_reye_y], row[idx_leye_x],\
                                                                     row[idx_leye_y], row[idx_nose_x], row[idx_nose_y]

                    if len(reye_x) > 0 and len(reye_y) > 0 and len(leye_x) > 0 and len(leye_y) > 0 and len(nose_x) > 0 and len(nose_y) > 0:
                        reye_x, reye_y, leye_x, leye_y, nose_x, nose_y = float(reye_x), float(reye_y), float(leye_x), \
                                                                         float(leye_y), float(nose_x), float(nose_y)
                        c_x, c_y = nose_x, nose_y
                        radius = math.sqrt(max((reye_x - nose_x)**2 + (reye_y - nose_y) ** 2, (leye_x - nose_x)**2 + (leye_y - nose_y) ** 2))
                        radius *= 1.1
                        x, y = c_x - radius, c_y - radius
                        w, h = 2 * radius, 2 * radius

                    im_name = os.path.splitext(os.path.basename(im_path))[0]
                    dir_name = os.path.dirname(im_path)

                    out_im_path = os.path.join(args.out_dir, dir_name, '%s.png' % im_name)
                    if not os.path.exists(os.path.dirname(out_im_path)):
                        os.makedirs(os.path.dirname(out_im_path))

                    if len(args.box_dir) > 0:
                        out_box_path = os.path.join(args.box_dir, dir_name, '%s.roi' % im_name)
                        if not os.path.exists(os.path.dirname(out_box_path)):
                            os.makedirs(os.path.dirname(out_box_path))
                    else:
                        out_box_path = os.path.join(args.out_dir, dir_name, '%s.roi' % im_name)

                    contents.append([im_path, template_id, subject_id, int(x), int(y), int(w), int(h), out_im_path,
                                     out_box_path])

    with Pool() as p:
        processed_contents = list(tqdm(p.imap(crop_image, contents), total=len(contents)))

    processed_contents = list(filter(None, processed_contents))

    processed_contents = remove_duplicates(processed_contents)

    if len(args.out_path) > 0:
        if not os.path.exists(os.path.dirname(args.out_path)):
            os.makedirs(os.path.dirname(args.out_path))
        write_csv(args.out_path, processed_contents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is used to crop images in IJB from scratch')
    parser.add_argument('db_dir', help='dataset directory')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('--set', default='A', help='A: IJB-A, B: IJB-B, C: IJB-C')
    parser.add_argument('--enlarge-factor', default=1.0, type=float, help='enlarge bounding box')
    parser.add_argument('--save-box', action='store_true', help='store bounding box')
    parser.add_argument('--box-dir', default='', help='bounding box directory')
    parser.add_argument('--out-path', default='', help='save final csv directory')
    parser.add_argument('--plot', action='store_true', help='plot')
    args = parser.parse_args()

    if not args.out_dir[-1] == '/':
        args.out_dir = args.out_dir + '/'

    main()
