import argparse
import dlib
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm
import csv
import numpy as np

from eugena.io import glob_files


def gen_bbox_from_landmarks(file_landmark):
    lms = np.loadtxt(file_landmark, delimiter=args.lm_delimiter)

    c_x, c_y = np.mean(lms, axis=0)
    x_min, y_min = np.min(lms, axis=0)
    x_max, y_max = np.max(lms, axis=0)
    box_size = int(max(x_max - x_min, y_max - y_min))

    x = int(c_x - box_size // 2)
    y = int(c_y - box_size // 2)
    w = box_size
    h = box_size

    return [x, y, w, h]


def center_crop(im):
    try:
        im_h, im_w = im.shape[:2]
        x = int(im_w * 0.0625)
        y = int(im_h * 0.0625)

        w = int(im_w - 2 * x)
        h = int(im_h - 2 * y)

        return [x, y, w, h]

    except Exception as e:
        print(e)
        return []


def detect_dlib(detector, im):
    try:
        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        enlarge_factor = 1
        while max(img.shape[0:2]) > 1000:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            enlarge_factor *= 2

        bbs = detector(img)

        bbs = [[b.rect.left() * enlarge_factor, b.rect.top() * enlarge_factor,
                b.rect.width() * enlarge_factor, b.rect.height() * enlarge_factor] for b in bbs]

        if len(bbs) > 0:
            return bbs[0]
        else:
            return []

    except Exception as e:
        print(e)
        return []


def detect_face():
    files = glob_files(args.db_dir, ['.jpg', '.jpeg', '.png', '.bmp'])

    detector = dlib.cnn_face_detection_model_v1(args.dlib_model)
    contents = []
    p_bar = tqdm.tqdm(total=len(files))
    for f in files:
        ext = os.path.splitext(f)[1]
        file_bbox = f.replace(args.db_dir, args.out_dir)
        file_bbox = file_bbox.replace(ext, '.roi')

        file_landmark = f.replace(args.db_dir, args.lm_dir)
        file_landmark = file_landmark.replace(ext, args.lm_ext)

        if os.path.exists(file_bbox):
            contents.append([f, file_bbox])
        else:
            if not os.path.exists(os.path.dirname(file_bbox)):
                os.makedirs(os.path.dirname(file_bbox))

            im = cv2.imread(f)

            if args.use_lm:
                box = gen_bbox_from_landmarks(file_landmark)
            else:
                box = detect_dlib(detector, f)

            if len(box) == 0:
                # center crop
                box = center_crop(im)

            if args.plot:
                plt.figure()
                plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                ax = plt.gca()
                ax.add_patch(patches.Rectangle((box[0], box[1]), box[2], box[3], fill=False, ec='g'))
                plt.show()

            with open(file_bbox, 'w') as fout:
                fout.write('%d %d %d %d' % (box[0], box[1], box[2], box[3]))

            contents.append([f, file_bbox])

        p_bar.update()

    p_bar.close()

    if len(args.out_path) > 0:
        with open(args.out_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['FILE', 'ROI'])
            for im_path, roi_path in contents:
                csv_writer.writerow([im_path.replace(args.db_dir, ''), os.path.realpath(roi_path)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dir', help='directory')
    parser.add_argument('out_dir', help='dlib bounding box dir')
    parser.add_argument('--dlib_model', default='deep_learning_models/dlib/mmod_human_face_detector.dat',
                        help='dlib model path')
    parser.add_argument('--plot', action='store_true', help='plotting')
    parser.add_argument('--use-lm', action='store_true', help='use landmarks to generate images')
    parser.add_argument('--lm-dir', default='', help='landmark directory')
    parser.add_argument('--lm-ext', default='.txt', help='landmark extension')
    parser.add_argument('--lm-delimiter', default=',', help='landmark delimiter')
    parser.add_argument('--out-path', default='', help='output path, default, does not save the path')
    args = parser.parse_args()

    assert len(args.db_dir) > 0, 'dataset directory cannot be empty'
    if args.db_dir[-1] != '/':
        args.db_dir += '/'

    assert len(args.out_dir) > 0, 'output directory cannot be empty'
    if args.out_dir[-1] != '/':
        args.out_dir += '/'

    if len(args.lm_dir) > 0 and len(args.lm_dir[-1]) != '/':
        args.lm_dir += '/'

    detect_face()
