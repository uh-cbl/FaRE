import os
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

import mxnet as mx
from mxnet.gluon.data import DataLoader
from mxnet.gluon.utils import split_and_load
from sklearn.metrics.pairwise import paired_distances as compute_paired_distances

from .imdb import IdentificationOpenSetDataset, VerificationDataset, load_single_template
from ..io import load_pkl, Signature
from ..metrics import BiometricCompareProtocol
from ..utils.dataloader import ImagePathDataset, ImagePathLabelDataset, PairedImagePathLabelDataset


class IJBAVerification(VerificationDataset):
    def __init__(self, **kwargs):
        super(IJBAVerification, self).__init__(**kwargs)

        self.protocol_dir = os.path.join(self.cur_dir, 'IJB-A')

        self.parse_data_set()

    def parse_data_set(self):

        self.Templates = load_pkl(os.path.join(self.protocol_dir, 'IJBA_11_Templates.pkl'))
        self.Matches = load_pkl(os.path.join(self.protocol_dir, 'IJBA_11_Matches.pkl'))
        self.num_folds = len(self.folds)

        self.num_iters = len(self.Templates)

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

        total = sum([len(match) for match in self.Matches])
        p_bar = tqdm(desc='Computing the similarity', total=total)

        for match in self.Matches:
            similarity, binary_labels = [], []
            for i in range(0, len(match), bs):
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

            binary_labels = np.concatenate(binary_labels)
            similarity = np.concatenate(similarity)

            self.protocols.append(BiometricCompareProtocol(binary_labels, similarity))

        p_bar.close()

    def generate_and_compute_similarities(self, inference, data_transform, bs, ctx=(mx.gpu(0)), norm_embeds=False,
                                          metric='cosine', output_transform=None, template_transform=None,
                                          save_template=False, **kwargs):

        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

        if isinstance(ctx, tuple):
            ctx = list(ctx)

        p_bar = tqdm(total=len(self.Templates), desc='Generating the templates')

        for k, t in self.Templates.items():
            if save_template:
                if os.path.exists(os.path.join(self.save_dir, str(t.template_id) + self.ft_ext)):
                    continue

            im_paths = [im_path for im_path in t.im_paths if os.path.exists(os.path.join(self.db_dir, im_path))]
            dataset = ImagePathDataset(self.db_dir, im_paths, transform=data_transform)
            dataloader = DataLoader(dataset, min(bs, len(im_paths)), last_batch='keep', pin_memory=False)

            template_embeds = []
            for im_batch in dataloader:
                im_lst = split_and_load(im_batch, ctx)

                for im in im_lst:
                    embeds = inference(im)

                    if isinstance(embeds, tuple) or isinstance(embeds, list):
                        embeds = output_transform(embeds)

                    if self.flatten:
                        embeds = mx.nd.flatten(embeds)

                    if norm_embeds:
                        embeds = mx.nd.L2Normalization(embeds, mode='instance')

                    embeds = embeds.asnumpy()

                    template_embeds.append(embeds)

            template_embeds = template_transform(template_embeds)

            self.Templates[k].load_signature(template_embeds)

            if save_template:
                assert t.template_id is not None, 'Template ID cannot be None'
                self.Templates[k].save_signature(os.path.join(self.save_dir, str(t.template_id) + self.ft_ext))

            p_bar.update(1)

        p_bar.close()

        bs = kwargs['bs'] if 'bs' in kwargs else 1000

        total = sum([len(match) for match in self.Matches])
        p_bar = tqdm(desc='Computing the similarity', total=total)

        for match in self.Matches:
            similarity, binary_labels = [], []
            for i in range(0, len(match), bs):
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

            binary_labels = np.concatenate(binary_labels)
            similarity = np.concatenate(similarity)

            self.protocols.append(BiometricCompareProtocol(binary_labels, similarity))

        p_bar.close()


class IJBAIdentification(IdentificationOpenSetDataset):
    def __init__(self, **kwargs):
        super(IJBAIdentification, self).__init__(**kwargs)

        self.protocol_dir = os.path.join(self.cur_dir, 'IJB-A/IJB-A_1N_sets')
        self.parse_data_set()

    def parse_data_set(self):
        self.folds = load_pkl(os.path.join(self.protocol_dir, 'IJBA_1N.pkl'))
        self.num_folds = len(self.folds)

        num_iters = 0
        for proto in self.folds:
            num_iters += len(proto['Gallery']) + len(proto['Probe'])

        self.num_iters = num_iters
