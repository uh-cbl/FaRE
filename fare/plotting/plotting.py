import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
sns.set(context='paper', style='whitegrid', font_scale=1, rc=params)


def plot_rocs(dataset_list, name_list, save_path=None, with_std=True, xlim=(0, 1), ylim=(0, 1),
              xlabel='False Positive Rate', ylabel='True Positive Rate', **kwargs):
    f = plt.figure()

    for i, dataset in enumerate(dataset_list):
        roc_fprs = np.array([protocol.curves['roc']['fpr'] for protocol in dataset.protocols])
        roc_tprs = np.array([protocol.curves['roc']['tpr'] for protocol in dataset.protocols])
        roc_fprs_mean = np.mean(roc_fprs, axis=0)
        roc_tprs_mean = np.mean(roc_tprs, axis=0)
        roc_tprs_std = np.std(roc_tprs, axis=0)
        plt.plot(roc_fprs_mean, roc_tprs_mean, label=name_list[i], **kwargs)
        if with_std:
            plt.fill_between(roc_fprs_mean, roc_tprs_mean - roc_tprs_std, roc_tprs_mean + roc_tprs_std, alpha=0.3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc=4)

    if save_path is not None:
        f.savefig(save_path)

    plt.show()


def plot_cmc(dataset_list, name_list, ranks=10, save_path=None, with_std=True, ylim=(0, 100),
             xlabel='Rank', ylabel='Face Identification Rate (%)', **kwargs):
    f = plt.figure()

    for i, dataset in enumerate(dataset_list):
        cmc = np.array([protocol.curves['cmc'] for protocol in dataset.protocols]) * 100

        cmc_mean = np.mean(cmc, axis=0)
        cmc_std = np.std(cmc, axis=0)

        plt.plot(np.arange(ranks) + 1, cmc_mean[: ranks], label=name_list[i], **kwargs)

        if with_std:
            plt.fill_between(np.arange(ranks) + 1,
                             np.clip(cmc_mean[:ranks] - cmc_std[:ranks], a_min=0, a_max=100),
                             np.clip(cmc_mean[:ranks] + cmc_std[:ranks], a_min=0, a_max=100),
                             alpha=0.3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([1, ranks])
    plt.legend(loc=4)

    if save_path is not None:
        f.savefig(save_path)

    plt.show()
