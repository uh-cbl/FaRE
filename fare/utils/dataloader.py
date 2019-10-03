
import os
import mxnet as mx
from mxnet.gluon.data import Dataset, RecordFileDataset


class ImagePathDataset(Dataset):

    def __init__(self, im_dir, im_paths, transform=None):
        self.im_dir = im_dir
        if im_dir is None:
            self.items = [im_path for im_path in im_paths if os.path.exists(im_path)]
        else:
            self.items = [os.path.join(im_dir, im_path) for im_path in im_paths
                          if os.path.exists(os.path.join(im_dir, im_path))]
        assert len(self.items) >= 1, 'length of item cannot be zero'
        self._tform = transform

    def __getitem__(self, idx):
        im = mx.image.imread(self.items[idx], 1)

        if self._tform is not None:
            return self._tform(im)

        return im

    def __len__(self):
        return len(self.items)


class ImagePathLabelDataset(Dataset):
    def __init__(self, im_dir, im_path_labels, transform=None):
        self.im_dir = im_dir
        self.items = [[os.path.join(im_dir, im_path), lb] for im_path, lb in im_path_labels
                      if os.path.exists(os.path.join(im_dir, im_path))]
        assert len(self.items) >= 1, 'length of item cannot be zero'
        self._tform = transform

    def __getitem__(self, idx):
        im = mx.image.imread(self.items[idx][0], 1)

        lb = self.items[idx][1]

        if self._tform is not None:
            im = self._tform(im)

        return im, lb

    def __len__(self):
        return len(self.items)


class PairedImagePathLabelDataset(Dataset):
    def __init__(self, im_dir, im_path_label_pairs, transform=None):
        self.im_dir = im_dir
        self.item_pairs = [[os.path.join(im_dir, im_path1), os.path.join(im_dir, im_path2), lb]
                           for im_path1, im_path2, lb in im_path_label_pairs
                           if os.path.exists(os.path.join(im_dir, im_path1))
                           and os.path.exists(os.path.join(im_dir, im_path2))]
        assert len(self.item_pairs) >= 1, 'length of item cannot be zero'
        self._tform = transform

    def __getitem__(self, idx):
        im1 = mx.image.imread(self.item_pairs[idx][0], 1)
        im2 = mx.image.imread(self.item_pairs[idx][1], 1)

        lb = self.item_pairs[idx][2]

        if self._tform is not None:
            im1 = self._tform(im1)
            im2 = self._tform(im2)

        return im1, im2, lb

    def __len__(self):
        return len(self.item_pairs)
