import os
from .imdb import VerificationDataset
from ..io import File


class LFW(VerificationDataset):
    def __init__(self, **kwargs):
        super(LFW, self).__init__(**kwargs)
        self.dataset_file = os.path.join(self.cur_dir, 'LFW/pairs_LFW.txt')

        self.num_folds = 10
        self.num_pos_pairs = 300
        self.num_neg_pairs = 300
        self.num_iters = (self.num_pos_pairs + self.num_neg_pairs) * self.num_folds

        self.parse_data_set()

    def parse_data_set(self):
        folds = []
        with open(self.dataset_file) as f:
            f.readline()

            for i_fold in range(self.num_folds):
                list_a, list_b, labels = [], [], []
                for i_pos_pair in range(self.num_pos_pairs):
                    line = f.readline()
                    tokens = line.strip('\n').split('\t')
                    person = tokens[0]
                    path1 = '%s/%s_%04d' % (person, person, int(tokens[1])) + self.im_ext
                    path2 = '%s/%s_%04d' % (person, person, int(tokens[2])) + self.im_ext

                    list_a.append(File(path1))
                    list_b.append(File(path2))
                    labels.append(1)

                for i_neg_pair in range(self.num_neg_pairs):
                    line = f.readline()
                    tokens = line.strip('\n').split('\t')
                    path1 = '%s/%s_%04d' % (tokens[0], tokens[0], int(tokens[1])) + self.im_ext
                    path2 = '%s/%s_%04d' % (tokens[2], tokens[2], int(tokens[3])) + self.im_ext

                    list_a.append(File(path1))
                    list_b.append(File(path2))
                    labels.append(0)

                folds.append({'list_a': list_a, 'list_b': list_b, 'labels': labels})

        self.folds = folds
