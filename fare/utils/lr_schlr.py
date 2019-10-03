

class msteps_lr_schlr(object):
    def __init__(self, base_lr, lr_epochs, lr_factor):

        assert isinstance(lr_epochs, list) and len(lr_epochs) >= 1

        self.cur_lr = base_lr
        self.cur_epochs = lr_epochs
        self.lr_factor = lr_factor

        self.cur_limit_epoch = self.cur_epochs.pop(0)
        self.last_limit = True if len(self.cur_epochs) == 0 else False

    def update(self, trainer, cur_epoch):
        if cur_epoch == self.cur_limit_epoch:

            self.cur_lr = self.cur_lr * self.lr_factor
            trainer.set_learning_rate(self.cur_lr)

            # update limits
            if not self.last_limit:
                self.cur_limit_epoch = self.cur_epochs.pop(0)
                self.last_limit = True if len(self.cur_epochs) == 0 else False

            return True

        return False
