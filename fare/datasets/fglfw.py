import os
from .imdb import VerificationDataset
from ..io import File


class FGLFW(VerificationDataset):
    def __init__(self, **kwargs):
        super(FGLFW, self).__init__(**kwargs)
        self.dataset_file = os.path.join(self.cur_dir, 'FGLFW/pair_FGLFW.txt')
        self.num_folds = 10
        self.num_pos_pairs = 300
        self.num_neg_pairs = 300

        self.num_iters = (self.num_pos_pairs + self.num_neg_pairs) * self.num_folds

        self.parse_data_set()

    def parse_data_set(self):
        folds = []
        with open(self.dataset_file) as f:
            for i_fold in range(self.num_folds):
                list_a, list_b, labels = [], [], []
                for i_pos_pair in range(self.num_pos_pairs):
                    path1 = f.readline()
                    path2 = f.readline()

                    path1 = path1.replace('.jpg', self.im_ext)
                    path2 = path2.replace('.jpg', self.im_ext)

                    list_a.append(File(path1))
                    list_b.append(File(path2))
                    labels.append(1)

                for i_neg_pair in range(self.num_neg_pairs):
                    path1 = f.readline()
                    path2 = f.readline()

                    path1 = path1.replace('.jpg', self.im_ext)
                    path2 = path2.replace('.jpg', self.im_ext)

                    list_a.append(File(path1))
                    list_b.append(File(path2))
                    labels.append(0)

                folds.append({'list_a': list_a, 'list_b': list_b, 'labels': labels})

        self.folds = folds
