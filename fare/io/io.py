import os
import csv
import numpy as np
import pickle as pkl
import scipy.io as sio
import logging
import json


def load_json(file_path, mode='r'):
    with open(file_path, mode) as f:
        json_dict = json.load(f)

    return json_dict


def save_json(file_path, object, mode='w'):
    with open(file_path, mode) as f:
        json.dump(object, f, indent=4)


def glob_files(d, exts=None):
    files = []
    for root, dirs, filenames in os.walk(d):
        for fname in filenames:
            if exts is not None:
                if os.path.splitext(fname)[1] in exts:
                    files.append(os.path.join(root, fname))
            else:
                files.append(os.path.join(root, fname))
    return files


def load_pkl(file_path, mode='rb'):
    with open(file_path, mode) as f:
        obj = pkl.load(f)

    return obj


def save_pkl(file_path, obj, mode='wb'):
    with open(file_path, mode) as f:
        pkl.dump(obj, f)


def count_lines(file_path):
    num_line = 0
    with open(file_path) as f:
        for _ in f:
            num_line += 1

    return num_line


def load_csv(file_path, read_header=True):
    with open(file_path) as f:
        csv_reader = csv.reader(f)
        content = []
        headers = None
        for i, row in enumerate(csv_reader):
            if read_header is True:
                if i == 0:
                    headers = row
                else:
                    item = {headers[idx]: item for idx, item in enumerate(row)}
                    content.append(item)
            else:
                content.append(row)

    return content


def save_csv(file_path, content, header=None):
    with open(file_path, 'w') as f:
        csv_writer = csv.writer(f)
        if header is not None:
            csv_writer.writerow(header)

        for i, row in enumerate(content):
            csv_writer.writerow([row])


def load_lm2d(file_path, delimiter=' '):
    lms = []

    with open(file_path) as f:
        for row in f:
            if '#' in row:
                continue
            item = row.split(delimiter)
            if len(item) == 3:
                item = item[1:]

            item = list(map(float, item))
            lms.append(item)

    return np.array(lms)


def save_ndarray(file_path, ndarray, fmode='txt'):
    try:

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        if fmode == 'numpy':
            np.save(file_path, ndarray)
            return True
        elif fmode == 'txt':
            np.savetxt(file_path, ndarray)
            return True
        else:
            raise NotImplementedError
    except Exception as e:
        logging.warning(e)
        return False


def load_ndarray(file_path, fmode='txt'):
    try:
        if fmode == 'numpy':
            ndarray = np.load(file_path)
        elif fmode == 'txt':
            ndarray = np.loadtxt(file_path)
        elif fmode == 'mat':
            ndarray = sio.loadmat(file_path)
        else:
            raise NotImplementedError

        return ndarray
    except Exception as e:
        logging.warning(e)
        return False
