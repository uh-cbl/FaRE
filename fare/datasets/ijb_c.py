"""
Script for evaluation IJB-C dataset
"""

import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from .imdb import IdentificationOpenSetDataset, VerificationDataset
from .imdb import load_single_template, compute_paired_distances
from ..io import Signature, load_pkl
from ..metrics import BiometricCompareProtocol


class IJBCMixVerification(VerificationDataset):
    def __init__(self, **kwargs):
        super(IJBCMixVerification, self).__init__(**kwargs)
        self.protocol_dir = os.path.join(self.cur_dir, 'IJB-C')

        self.parse_data_set()

    def parse_data_set(self):
        file_gallery1_path = os.path.join(self.protocol_dir, 'IJBC_1N_G1.pkl')
        file_gallery2_path = os.path.join(self.protocol_dir, 'IJBC_1N_G2.pkl')
        file_probe_path = os.path.join(self.protocol_dir, 'IJBC_1N_Pb.pkl')
        file_matches_path1 = os.path.join(self.protocol_dir, 'IJBC_11_Matches-1.pkl')
        file_matches_path2 = os.path.join(self.protocol_dir, 'IJBC_11_Matches-2.pkl')
        file_matches_path3 = os.path.join(self.protocol_dir, 'IJBC_11_Matches-3.pkl')

        G1 = load_pkl(file_gallery1_path)
        G2 = load_pkl(file_gallery2_path)

        Pb = load_pkl(file_probe_path)

        matches1 = load_pkl(file_matches_path1)
        matches2 = load_pkl(file_matches_path2)
        matches3 = load_pkl(file_matches_path3)

        self.Templates = {**G1, **G2, **Pb}
        self.Matches = matches1 + matches2 + matches3

    def load_and_compute_similarities(self, metric='cosine', template_transform=None, **kwargs):
        # load templates
        items = []
        for k, t in self.Templates.items():
            items.append([os.path.join(self.ft_dir, str(t.template_id) + self.ft_ext), t.template_id,
                          self.fmode, self.flatten])

        total = len(items)
        with Pool() as p:
            content = list(tqdm(p.imap(load_single_template, items), total=total, desc='Loading the templates'))

        for ft, t_id in content:
            self.Templates[t_id].signature = Signature(ft)

        bs = kwargs['bs'] if 'bs' in kwargs else 1000

        similarity, binary_labels = [], []
        p_bar = tqdm(desc='Computing the similarity', total=len(self.Matches))

        for i in range(0, len(self.Matches), bs):
            # load two lists
            matches = self.Matches[i: min(i + bs, len(self.Matches))]
            templates_a, templates_b, labels = [], [], []
            for t_id1, t_id2 in matches:
                templates_a.append(self.Templates[t_id1].signature.features)
                templates_b.append(self.Templates[t_id2].signature.features)
                lb = 1 if self.Templates[t_id1].subject_id == self.Templates[t_id2].subject_id else 0
                labels.append(lb)

            templates_a, templates_b = np.concatenate(templates_a), np.concatenate(templates_b)

            sim = 1 - compute_paired_distances(templates_a, templates_b, metric=metric)
            lb = np.array(labels)

            similarity.append(sim)
            binary_labels.append(lb)

            step = len(self.Matches) - i if i + bs > len(self.Matches) else bs
            p_bar.update(step)

        p_bar.close()

        binary_labels = np.concatenate(binary_labels)
        similarity = np.concatenate(similarity)

        self.protocols.append(BiometricCompareProtocol(binary_labels, similarity))


class IJBCMixIdentification(IdentificationOpenSetDataset):
    def __init__(self, **kwargs):
        super(IJBCMixIdentification, self).__init__(**kwargs)
        self.protocol_dir = os.path.join(self.cur_dir, 'IJB-C')

        self.num_folds = 2
        self.folds = [{'gallery': [], 'probe': []} for _ in range(self.num_folds)]

        self.parse_data_set()

    def parse_data_set(self):
        file_gallery1_path = os.path.join(self.protocol_dir, 'IJBC_1N_G1.pkl')
        file_gallery2_path = os.path.join(self.protocol_dir, 'IJBC_1N_G2.pkl')
        file_probe_path = os.path.join(self.protocol_dir, 'IJBC_1N_Pb.pkl')

        gallery1 = list(load_pkl(file_gallery1_path).values())
        gallery2 = list(load_pkl(file_gallery2_path).values())

        probe = list(load_pkl(file_probe_path).values())

        self.folds[0]['gallery'] = gallery1
        self.folds[0]['probe'] = probe
        self.folds[1]['gallery'] = gallery2
        self.folds[1]['probe'] = probe

        self.num_iters = len(gallery1) + len(gallery2) + 2 * len(probe)
