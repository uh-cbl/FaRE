"""
The models functions
"""

import os
import yaml
import numpy as np

import mxnet as mx
from mxnet import autograd
from mxnet.metric import Accuracy, Loss
from mxnet.initializer import Xavier
from mxnet.gluon import trainer
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms, ImageRecordDataset
from mxnet.gluon.loss import SoftmaxCELoss
from mxnet.gluon.utils import split_and_load
from mxnet.gluon.model_zoo.vision import get_model

from .lr_schlr import msteps_lr_schlr
from ..logging.logger import TextLogger, SummaryLogger


class BasicModel(object):
    """
    Basic Face Recognition Model
    """
    def __init__(self, args):
        """
        Initialization of the basic model
        :param args:
        """
        self.args = args
        self.create_logger()

        # Loader
        self.load_cfg()
        self.create_loader()
        self.create_metrics()
        self.reset_metrics()

        self.create_context()

        self.cur_epoch = 0
        self.cur_iter = 0

        # records
        self.create_records()

    @staticmethod
    def check_dir_exists(dpath):
        """
        Check the direcotry exists
        :param dpath: directory path
        :return:
        """
        if not os.path.exists(dpath):
            return os.makedirs(dpath)
        else:
            return True

    def create_logger(self):
        """
        Create the logger
        :return:
        """
        # Logging
        remove_previous_log = self.args.start_epoch == 0 and self.args.mode.upper() == 'TRAIN'
        self.text_logger = TextLogger(self.args.log_dir, self.args.fname + '.log',
                                      remove_previous_log=remove_previous_log)
        self.check_dir_exists(self.args.ckpt_dir)

        if self.args.mxboard:
            self.summary_logger = SummaryLogger(self.args.log_dir, filename_suffix=self.args.fname)
        else:
            self.summary_logger = None

        if self.args.mode.upper() == 'TRAIN':
            args = vars(self.args)
            arg_str = '\n'
            for k, v in args.items():
                arg_str += '%s: %s\n' % (k, v)

            self.text_logger.add_str(arg_str)

    def load_cfg(self):
        """
        Load the configuration file to dictionary
        :return:
        """
        with open(self.args.cfg) as f:
            self.cfg = yaml.load(f)

    def create_metrics(self):
        """
        Create the metrics
        :return:
        """
        self.metrics = {'Train-Xent': Loss(),
                        'Train-Aux': Loss(),
                        'Train-Acc': Accuracy(),
                        'Val-Xent': Loss(),
                        'Val-Aux': Loss(),
                        'Val-Acc': Accuracy()
                        }

    def reset_metrics(self):
        """
        Reset the metrics
        :return:
        """
        for v in self.metrics.values():
            v.reset()

    def create_records(self):
        """
        Create records
        :return:
        """
        test_dbs = self.cfg['eval'].keys()

        self.records = {}

        for db in list(test_dbs):
            self.records[db] = []

    def reset_records(self):
        """
        Reset records
        :return:
        """
        for k in self.records.keys():
            self.records[k] = []

    def create_loader(self):
        """
        Create the data loader
        :return:
        """
        if self.args.mode.upper() == 'TRAIN':
            tforms = []
            tforms.append(transforms.Resize(self.args.resize))

            if self.args.flip:
                tforms.append(transforms.RandomFlipLeftRight())

            if self.args.random_crop:
                tforms.append(transforms.RandomResizedCrop(self.args.im_size, scale=(0.8, 1)))
            else:
                tforms.append(transforms.CenterCrop(self.args.im_size))

            if self.args.random_jitter:
                tforms.append(transforms.RandomColorJitter(0.4, 0.4, 0.4, 0.4))

            tforms.append(transforms.ToTensor())
            tforms.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

            tforms = transforms.Compose(tforms)

            tr_db = list(self.cfg['train'].values())[0]

            dataset = ImageRecordDataset(tr_db['rec'], transform=tforms)

            self.tr_loader = DataLoader(dataset, batch_size=self.args.bs, num_workers=8, pin_memory=True)

        else:
            tforms = transforms.Compose([
                transforms.Resize(self.args.resize),
                transforms.CenterCrop(self.args.im_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            self.eval_tforms = tforms

    def pre_processing(self, im_path):
        """
        Pre-processing of the image
        :param im_path:
        :return:
        """
        with open(im_path, 'rb') as f:
            im = mx.image.imdecode(f.read())

        im = self.eval_tforms(im)

        im = im.expand_dims(axis=0)

        im = im.as_in_context(self.ctx[0])

        return im

    def create_context(self):
        """
        Create the context
        :return:
        """
        self.ctx = [mx.gpu(int(gpu_id)) for gpu_id in self.args.gpus.split(',')] if len(self.args.gpus) > 0 else [mx.cpu()]

    def create_model(self):
        """
        Create the model
        :return:
        """
        tr_db = list(self.cfg['train'].values())[0]
        num_class = tr_db['num_cls']

        self.inference = get_model(self.args.arch, classes=num_class)

        if self.args.hybridize:
            self.inference.hybridize()

    def init_model(self, is_train=True):
        """
        Initialized the model
        :param is_train:
        :return:
        """
        if self.args.pretrained_mode == 0 and os.path.exists(self.args.model_path):
            # load imagenet pretrained model
            print('load pretrained model from %s...' % self.args.model_path)
            self.inference.init_imagenet_params(self.args.model_path, self.ctx)
        elif self.args.pretrained_mode == 1 and os.path.exists(self.args.model_path):
            # load webface pretrained model
            print('load pretrained model from %s...' % self.args.model_path)
            self.inference.init_convs_params(self.args.model_path, self.ctx)
        elif self.args.pretrained_mode == 2 and os.path.exists(self.args.model_path):
            print('load pretrained model from %s...' % self.args.model_path)
            self.inference.load_parameters(self.args.model_path, ctx=self.ctx, allow_missing=True)
            self.inference.init_extra_params(self.ctx)
        elif self.args.pretrained_mode == 3 and os.path.exists(self.args.model_path):
            print('load pretrained model from %s...' % self.args.model_path)
            self.inference.load_parameters(self.args.model_path, ctx=self.ctx)
            model_name, cur_epoch, cur_iter = (os.path.splitext(os.path.basename(self.args.model_path))[0]).split('-')
            self.args.start_epoch = int(cur_epoch)
            self.cur_iter = int(cur_iter)
            self.cur_epoch = int(cur_epoch)
        elif self.args.model_path.endswith('.params') or not is_train:
            # load from the path
            print('load model from the %s...' % self.args.model_path)
            # self.inference.load_parameters(self.args.model_path, ctx=self.ctx, ignore_extra=True)
            self.inference.load_parameters(self.args.model_path, ctx=self.ctx, ignore_extra=True)
        elif self.args.start_epoch > 0:
            # load from the previous checkpoint
            model_path = os.path.join(self.args.ckpt_dir, '%s-epoch-%d.params' % (self.args.arch, self.args.start_epoch))
            print('load the model for the epoch [%d] from %s...' % (self.args.start_epoch, model_path))
            self.inference.load_parameters(model_path, self.ctx)
        else:
            # random initialize
            print('initialize the model using Xavier initializer...')
            self.inference.initialize(Xavier(rnd_type='gaussian', factor_type='out', magnitude=2), ctx=self.ctx)

    def create_lr_schlr(self):
        """
        Learning rate scheduler
        :return:
        """
        if self.args.lr_schlr == 'msteps':
            iters = [int(iter) for iter in self.args.lr_iters.split(',')]
            self.lr_schlr = msteps_lr_schlr(self.args.lr, iters, self.args.lr_factor)
        else:
            self.lr_schlr = None

    def update_learning_rate(self):
        """
        Update the learning rate
        :return:
        """
        if self.lr_schlr is not None:
            if self.lr_schlr.update(self.trainer, self.cur_iter):
                self.text_logger.add_scalar("Learning rate", self.trainer.learning_rate)

    def create_optim(self):
        """
        Create the optimizer
        :return:
        """
        self.create_lr_schlr()

        if self.args.optim == 'sgd':
            optim_params = {'learning_rate': self.args.lr, 'momentum': self.args.mom, 'wd': self.args.wd}
        else:
            raise NotImplementedError

        self.trainer = trainer.Trainer(self.inference.collect_params(),
                                       optimizer=self.args.optim,
                                       optimizer_params=optim_params)

        if self.args.model_path and self.args.load_optim_states:
            states_path = self.args.model_path.replace('.params', '.states')

            if os.path.exists(states_path):
                print('load the optimizer states from %s' % states_path)
                self.trainer.load_states(states_path)

    def train_model(self):
        """
        Training model
        :return:
        """
        # Build the model
        self.create_model()
        # initialize the model
        self.init_model()
        # create optimizer
        self.create_optim()

        for cur_epoch in range(self.args.start_epoch + 1, self.args.end_epoch + 1):
            # set the metrics
            self.cur_epoch = cur_epoch
            self.reset_metrics()
            # train the model
            self.train_epoch()
            # save the model
            self.save_params()

            if self.cur_epoch >= self.args.eval_min_epoch and self.cur_epoch % self.args.eval_itv == 0:
                self.eval_model()

        # log the best epoch and iter
        self.log_best_records()

    def train_epoch(self):
        """
        Training epoch
        :return:
        """
        self.inference.set_train()
        for data, label in self.tr_loader:
            self.train_batch(data, label)

            if self.args.save_params_itv != 0 and self.cur_iter % self.args.save_params_itv == 0:
                self.save_params()
                self.eval_model(epoch=False)
                self.inference.set_train()

            self.update_learning_rate()

        self.log_epoch()

    def train_batch(self, data, label):
        """
        Train batch
        :param data:
        :param label:
        :return:
        """
        data_lst = split_and_load(data, self.ctx)
        label_lst = split_and_load(label, self.ctx)

        criterion_xent = SoftmaxCELoss()

        with autograd.record():
            losses = []
            for x, y in zip(data_lst, label_lst):
                y_hat = self.inference(x)
                l_xent = criterion_xent(y_hat, y[:, 0])

                loss = l_xent

                losses.append(loss)

                # logging
                self.metrics['Train-Xent'].update(None, [l_xent])
                self.metrics['Train-Acc'].update([y[:, 0]], [y_hat])

            for l in losses:
                l.backward()

        self.trainer.step(data.shape[0])

        self.cur_iter += 1

        if self.args.log_itv != 0 and self.cur_iter % self.args.log_itv == 0:
            self.log_iter()

    def save_params(self):
        """
        Save the parameters
        :return:
        """
        prefix = os.path.join(self.args.ckpt_dir, '%s-%d-%d' % (self.args.arch, self.cur_epoch, self.cur_iter))
        self.inference.save_parameters('%s.params' % prefix)
        self.trainer.save_states('%s.states' % prefix)

    def log_epoch(self):
        self.log_text(epoch=True)
        self.log_summary(epoch=True)

    def log_iter(self):
        self.log_text(epoch=False)
        self.log_summary(epoch=False)

    def log_text(self, epoch=True):
        prefix = 'Epoch [%d]' % self.cur_epoch if epoch else 'Epoch [%d-%d]' % (self.cur_epoch, self.cur_iter)

        self.text_logger.add_scalar('%s: Train-Xent' % prefix, '%.4f' % self.metrics['Train-Xent'].get()[1])
        self.text_logger.add_scalar('%s: Train-Acc' % prefix, '%.2f%%' % (self.metrics['Train-Acc'].get()[1] * 100))

    def log_summary(self, epoch=True):
        prefix = 'Epoch' if epoch else 'Iter'
        cur_step = self.cur_epoch if epoch else self.cur_iter

        if self.summary_logger is not None:
            self.summary_logger.add_scalar('%s-Train-Xent' % prefix, self.metrics['Train-Xent'].get()[1], cur_step)
            self.summary_logger.add_scalar('%s-Train-Acc' % prefix, self.metrics['Train-Acc'].get()[1], cur_step)

    def log_best_records(self):
        test_dbs = self.cfg['eval'].keys()
        for db in list(test_dbs):
            if len(self.records[db]) > 0:
                epoch_acc = np.array(self.records[db])
                ind = np.argmax(epoch_acc[:, 1])
                epoch, acc = epoch_acc[ind]
                self.text_logger.add_str('Best Epoch [%d]: Acc=%.4f' % (epoch, acc))

    def eval_model(self, epoch=True):
        pass

    def test_model(self):
        pass

