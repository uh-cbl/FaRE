import mxnet as mx
from mxnet import gluon, metric, autograd
import logging
import timeit
import math
from datetime import timedelta


def multi_factor_scheduler(lr_step_epochs, num_samples, batch_size, lr_factor, start_epoch):
    num_samples = num_samples
    epoch_size = int(math.ceil(float(num_samples) / batch_size))
    step_epochs = [int(l) - start_epoch for l in lr_step_epochs.split(',')]
    steps = [epoch_size * x for x in step_epochs]

    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(steps, factor=lr_factor)

    return lr_scheduler


def train_cls_network(inference, train_loader, trainer, cur_epoch, ctx, criterion, log_iter=100):
    metric_acc = metric.Accuracy()
    metric_loss = metric.Loss()

    train_loader.reset()

    epoch_start_time = timeit.default_timer()

    for cur_batch, batch in enumerate(train_loader):
        batch_start_time = timeit.default_timer()

        batch_size = batch.data[0].shape[0]

        data = gluon.utils.split_and_load(batch.data[0], ctx)
        label = gluon.utils.split_and_load(batch.label[0], ctx)

        with autograd.record(train_mode=True):
            losses = []
            for x, y in zip(data, label):
                y_hat = inference(x)
                loss = criterion(y_hat, y)
                losses.append(loss)

                metric_loss.update(None, preds=[loss])
                metric_acc.update(preds=[y_hat], labels=[y])

        for loss in losses:
            loss.backward()

        trainer.step(batch_size)

        if cur_batch % log_iter == 0 and cur_batch > 0:
            batch_elpased_time = timeit.default_timer() - batch_start_time
            print('Epoch [%d-%d]: Speed: %.2f samples/s \t Accuracy: %.2f \t Loss: %.4f' %
                  (cur_epoch, cur_batch, batch_elpased_time / batch_size, 100 * metric_acc.get()[1],
                   metric_loss.get()[1]))

    epoch_elapsed_time = timeit.default_timer() - epoch_start_time

    logging.info('Epoch [%d]: Accuracy: %.2f' % (cur_epoch, 100 * metric_acc.get()[1]))
    logging.info('Epoch [%d]: Loss: %.2f' % (cur_epoch, metric_loss.get()[1]))
    logging.info('Epoch [%d]: Elapsed time: %s' % (cur_epoch, str(timedelta(seconds=epoch_elapsed_time))))

    return metric_acc.get()[1]
