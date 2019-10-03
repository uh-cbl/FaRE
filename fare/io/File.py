import numpy as np
import logging
from .io import load_ndarray, save_ndarray


class Signature(object):
    """
    The class to read and save the features
    """
    def __init__(self, features, occ=None, fmode='txt', flatten=True):
        self.flatten = flatten
        self.fmode = fmode

        if isinstance(features, str):
            self.features = self.load_features(features, fmode)
        else:
            self.features = features

        if isinstance(occ, str):
            self.occ = self.load_features(occ, fmode)
        else:
            self.occ = occ

    def load_features(self, feature_path, fmode='txt'):
        features = load_ndarray(feature_path, fmode=fmode)

        if self.flatten:
            features = features.flatten()

        return features

    def save_features(self, file_path, occ_path=None):
        if self.features is not None:
            save_ndarray(file_path, self.features, fmode=self.fmode)
        else:
            logging.warning('Features is None, pls load')

        if self.occ and occ_path:
            save_ndarray(occ_path, self.occ, fmode=self.fmode)


class File(object):
    """
    The basic file
    """
    def __init__(self, im_path=None, subject_id=None, features=None, occ=None, fmode='txt', flatten=True, **kwargs):

        self.im_path = im_path
        self.subject_id = subject_id

        self.fmode = fmode
        self.flatten = flatten

        self.signature = Signature(features=features, occ=occ, fmode=fmode, flatten=flatten)

        self.auxiliary = {}
        for k, v in kwargs.items():
            self.auxiliary[k] = v

    def set_item(self, key, value):
        self.auxiliary[key] = value

    def get_item(self, key):
        return self.auxiliary[key]

    def load_signature(self, feature_path, occ=None):
        self.signature = Signature(features=feature_path, occ=occ, fmode=self.fmode, flatten=self.flatten)


class Template(object):
    """
    The class for set-base face recognition
    """
    def __init__(self, im_paths=None, subject_id=None, template_id=None, features=None, occ=None, fmode='txt',
                 flatten=True, **kwargs):

        self.im_paths = im_paths
        self.subject_id = subject_id
        self.template_id = template_id

        self.signature = Signature(features=features, occ=occ, fmode=fmode, flatten=flatten)

        self.fmode = fmode
        self.flatten = flatten

        self.auxiliary = {}
        for k, v in kwargs.items():
            self.auxiliary[k] = v

    def set_item(self, key, value):
        self.auxiliary[key] = value

    def get_item(self, key):
        return self.auxiliary[key]

    def load_signature(self, signature, occ=None):
        self.signature = signature if isinstance(signature, Signature) else \
            Signature(features=signature, occ=occ, fmode=self.fmode, flatten=self.flatten)

    def save_signature(self, feature_path, occ_path=None):
        assert self.signature is not None, 'Template features is None'

        self.signature.save_features(feature_path, occ_path)

    # def compute_template_signature(self, list_files=None, template_transform='mean', weights=None):
    #     features = []
    #     occs = []
    #
    #     if list_files is not None:
    #         self.im_paths = list_files
    #
    #     for file in self.im_paths:
    #         if file.signature is None:
    #             logging.warning('file: %s does not contains features, pls load' % file.im_path)
    #         else:
    #             features.append(file.signature.features)
    #             if file.signature.occ is not None:
    #                 occs.append(file.signature.occ)
    #
    #     if len(features) > 1:
    #         if template_transform == 'mean':
    #
    #             if len(occs) > 0:
    #                 # ur2d signature [len, 512, 8] and occlusion mask [len, 8] -> [512, 8], [8]
    #                 features, occs = np.array(features), np.array(occs)
    #                 # occs = np.expand_dims(occs, axis=1)
    #                 features_weighted = features * occs
    #                 features = np.mean(features_weighted, axis=0)
    #                 # norm
    #                 occ = (np.mean(occs, axis=0) > 0).astype(np.int)
    #
    #                 self.signature = Signature(features=features, occ=occ, fmode=self.fmode, flatten=self.flatten)
    #             else:
    #                 # common signature
    #                 template_signature = np.mean(features, axis=0)
    #                 self.signature = Signature(features=template_signature, fmode=self.fmode, flatten=self.flatten)
    #         else:
    #             self.signature = Signature(template_transform(features, weights=weights), fmode=self.fmode,
    #                                        flatten=self.flatten)
