import os
from .imdb import VerificationDataset
from ..io import File


class CFP(VerificationDataset):
    def __init__(self, db_protocol='FP', **kwargs):
        super(CFP, self).__init__(**kwargs)
        assert db_protocol in ['FP', 'FF'], 'protocol should be FP or FF'

        self.pair_dict_frontal = self._read_pair_list(os.path.join(self.cur_dir, 'CFP', 'Pair_list_F.txt'))
        self.pair_dict_profile = self._read_pair_list(os.path.join(self.cur_dir, 'CFP', 'Pair_list_P.txt'))

        self._db_proto = db_protocol

        self.protocol_dir = os.path.join(self.cur_dir, 'CFP', 'Split', self._db_proto)

        self.num_folds = 10
        self.num_pos_pairs = 350
        self.num_neg_pairs = 350

        self.num_iters = (self.num_pos_pairs + self.num_neg_pairs) * self.num_folds

        self.parse_data_set()

    @staticmethod
    def _read_pair_list(file_path):
        im_pair_dict = {}
        with open(file_path) as f:
            for line in f:
                ind, path = line.split()
                im_pair_dict[int(ind)] = path

        return im_pair_dict

    def parse_data_set(self):
        folds = []

        for i, fold in enumerate(range(self.num_folds)):
            list_a, list_b, labels = [], [], []
            with open(os.path.join(self.protocol_dir, '%02d' % (i + 1), 'same.txt')) as f:
                for line in f:
                    ind1, ind2 = line.split(sep=',')
                    if self._db_proto == 'FF':
                        path1 = self.pair_dict_frontal[int(ind1)]
                        path2 = self.pair_dict_frontal[int(ind2)]
                    else:
                        path1 = self.pair_dict_frontal[int(ind1)]
                        path2 = self.pair_dict_profile[int(ind2)]

                    path1 = path1.replace('.jpg', self.im_ext)
                    path2 = path2.replace('.jpg', self.im_ext)

                    list_a.append(File(path1))
                    list_b.append(File(path2))
                    labels.append(1)

            with open(os.path.join(self.protocol_dir, '%02d' % (i + 1), 'diff.txt')) as f:
                for line in f:
                    ind1, ind2 = line.split(sep=',')
                    if self._db_proto == 'FF':
                        path1 = self.pair_dict_frontal[int(ind1)]
                        path2 = self.pair_dict_frontal[int(ind2)]
                    else:
                        path1 = self.pair_dict_frontal[int(ind1)]
                        path2 = self.pair_dict_profile[int(ind2)]

                    list_a.append(File(path1))
                    list_b.append(File(path2))
                    labels.append(0)

            folds.append({'list_a': list_a, 'list_b': list_b, 'labels': labels})

        self.folds = folds
