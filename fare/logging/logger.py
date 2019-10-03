"""
Basic Logging Class
"""
import os
import logging
from mxboard import SummaryWriter


class BasicLogger(object):
    def __init__(self, dpath, fname, remove_previous_log=True):
        self.dpath = dpath
        self.fname = fname
        self.fpath = os.path.join(self.dpath, self.fname)

        if self.check_file_exists():
            if remove_previous_log:
                self.remove_log()
        else:
            self.check_dir_exists()
            # self.create_log()

    def check_dir_exists(self):
        if not os.path.exists(self.dpath):
            return os.makedirs(self.dpath)
        else:
            return True

    def check_file_exists(self):
        if not os.path.exists(self.fpath):
            return False
        else:
            return True

    def remove_log(self):
        os.remove(self.fpath)

    def create_log(self):
        f = open(self.fpath, 'w+')
        f.close()

    def add_scalar(self, key, value, global_step=0):
        raise NotImplementedError


class TextLogger(BasicLogger):
    def __init__(self, dpath, fname, remove_previous_log=True):
        super(TextLogger, self).__init__(dpath, fname, remove_previous_log)

        logging.basicConfig(format='%(asctime)s-%(message)s', level=logging.INFO)
        self.logger = logging.getLogger()

        hdlr = logging.FileHandler(self.fpath)
        logger_formatter = logging.Formatter('%(asctime)s-%(message)s')
        hdlr.setFormatter(logger_formatter)
        self.logger.addHandler(hdlr)

    def add_scalar(self, key, value, global_step=0):
        self.logger.info('%s=%s' % (key, value))

    def add_str(self, str_log):
        self.logger.info(str_log)


SummaryLogger = SummaryWriter
