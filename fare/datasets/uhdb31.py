import os
from tqdm import tqdm
from multiprocessing import Pool
from .imdb import IdentificationCloseSetDataset

from ..io import glob_files, File


class UHDB31(IdentificationCloseSetDataset):
    def __init__(self, gallery_regex='d31%02d.bmp', probe_regex='d31%02d.bmp', **kwargs):
        if 'num_folds' not in kwargs:
            kwargs['num_folds'] = 20
        super(UHDB31, self).__init__(**kwargs)

        self.gallery_regex = gallery_regex
        self.probe_regex = probe_regex

        self.parse_data_set()

    def parse_data_set(self):
        folds = []

        im_files = os.listdir(self.db_dir)
        num_iters = 0
        for i in range(self.num_folds):
            ind = i + 1 if i < 10 else i + 2
            list_a = [im_file for im_file in im_files if self.gallery_regex % 11 in im_file]
            list_b = [im_file for im_file in im_files if self.probe_regex % ind in im_file]
            list_a.sort()
            list_b.sort()
            list_a = [File(im_path, int(os.path.basename(im_path)[:5]) - 90001) for im_path in list_a]
            list_b = [File(im_path, int(os.path.basename(im_path)[:5]) - 90001) for im_path in list_b]

            folds.append({'gallery': list_a, 'probe': list_b})

            num_iters += len(list_a) + len(list_b)

        self.folds = folds
        self.num_iters = num_iters

